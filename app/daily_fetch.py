"""
End-to-end daily orchestration for the vol prediction pipeline.

Invocation
----------
    uv run python app/daily_fetch.py            # full daily run
    uv run python app/daily_fetch.py --dry-run  # print plan, do not fetch/train

Designed for a 9:00 AM ET cron entry. Schedule it at 9:05 AM ET so the 09:00
1-min bar has fully printed before the script reads it (the bar is left-
labelled and closes at 09:01).

Pipeline (sequential, hard-fail on any error)
---------------------------------------------
  1. Connect to IB Gateway (live, port 4001 by default).
  2. Backfill data sources for the gap since the last run:
       a. ES  1m bars   → IB
       b. VX  1m bars   → IB
       c. ZN  1m bars   → IB
       d. News articles → Mediastack
       e. Sentiment     → FinBERT + LM Studio Qwen + embeddings + PCA
  3. Settle yesterday's prediction by computing its realized range from
     the now-refreshed ES cache and updating predictions.parquet.
  4. Rebuild every feature parquet with keep_unsettled=True so today's
     row survives the merge.
  5. Retrain the vol model:
        - refit on full data with cached best_params (cheap, daily)
        - full Optuna search if no cached params or > 7 days since last
          search (~25 min, weekly cadence by default)
  6. Build today's 9 AM feature row.
  7. Predict today's realized range.
  8. Append the prediction to predictions.parquet.

Hard-fail semantics
-------------------
Every step raises on failure. Cron should be configured to email the user on
non-zero exit so missed days are surfaced immediately. Soft fallbacks are
intentionally absent — silent degradation would mask drift between the live
pipeline and what the user thinks the model sees.

Prerequisites at runtime
------------------------
- IB Gateway running on 127.0.0.1:4001 with live API enabled.
- LM Studio running locally serving the Tier-2 Qwen model.
- NEWS_API_KEY (Mediastack), HF_TOKEN (HuggingFace) in app/.env.
- All five base parquets already cold-started via the per-source scripts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ib_client import ib_session
from market_data import build_feature_matrix, RAW_PATH as ES_RAW_PATH
from macro_data import (
    build_macro_features,
    update_macro_caches_incremental_async,
    OUTPUT_PATH as MACRO_FEATURES_PATH,
)
from calendar_features import build_calendar_features, OUTPUT_PATH as CAL_PATH
from merge_features import merge_features, OUTPUT_PATH as MODEL_DATA_PATH
import predict_vol as pv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_PROJ_ROOT = Path(__file__).resolve().parent.parent
_DATA = _PROJ_ROOT / "data"
PREDICTIONS_PATH = _DATA / "predictions.parquet"

# Retrain cadence: a full Optuna search runs at most this often.
OPTUNA_REFRESH_DAYS = 7

# Vol-model mode used in production. MAE has been the recommended one since
# the May 2026 vol-model-findings memo; see project_vol_model.md.
MODE = "mae"

# Historical caches that MUST exist before the orchestrator runs. If any of
# these are missing, downstream functions would silently fall back to a full
# cold-start refetch (Databento, Mediastack 5-yr, or a PCA re-fit). The
# orchestrator is for daily incremental work only — never historical loads.
# Cold-start each of these once via its source-specific script before
# scheduling the cron job.
_REQUIRED_CACHES: tuple[tuple[Path, str], ...] = (
    (_DATA / "es_intraday_raw.parquet",          "ES 1m bars — run market_data.fetch_es_futures()"),
    (_DATA / "macro_vix_intraday_raw.parquet",   "VX 1m bars — run macro_data.fetch_series('vix')"),
    (_DATA / "macro_tenY_intraday_raw.parquet",  "ZN 1m bars — run macro_data.fetch_series('tenY')"),
    (_DATA / "sp500_news.parquet",               "raw news — run news_data.fetch_sp500_news_history()"),
    (_DATA / "article_scores_checkpoint.parquet", "sentiment checkpoint — run sentiment-local.py historical scoring"),
    (_DATA / "qwen3_pca.joblib",                  "PCA model — fitted by sentiment-local.py on first historical run"),
)


# ----- Run-time safety window ------------------------------------------------
# The pipeline pins its IB fetch upper-bound to today 09:01 ET (see
# ib_client.today_9am_et_as_utc), which makes the entire run idempotent
# across the rest of the ET day — same forecast whether you run at 09:05,
# 11:00, or 18:00. The time guard therefore only needs to enforce:
#
#   - Lower bound 09:01 ET: today's 09:00 left-labelled bar must be finalised
#     in IB before we ask for it. Running at 09:00:30 risks fetching a partial
#     or absent 09:00 bar (it covers 09:00:00-09:00:59 and only finalises at
#     09:01:00). build_premarket_features keys off strict equality on the
#     09:00 bar, so a missing bar means today's row never enters the features
#     parquet and step 6 crashes.
#
#   - Upper bound 23:59:59 ET: after midnight ET the calendar date rolls over,
#     and the next day's 09:00 bar doesn't exist yet (it would be tomorrow
#     09:01 ET). today_9am_et_as_utc would raise in that window because the
#     newly-rolled "today" hasn't reached 09:01 ET yet.
#
# Bypass with --skip-time-guard for ad-hoc backfills only.
SAFE_WINDOW_START_ET = time(9, 1)
SAFE_WINDOW_END_ET = time(23, 59, 59)
_ET = ZoneInfo("America/New_York")


def _verify_run_time_window() -> None:
    """Hard-fail if current ET time is outside [09:01, 23:59:59] ET."""
    now_et = datetime.now(tz=_ET)
    t = now_et.time()
    if not (SAFE_WINDOW_START_ET <= t <= SAFE_WINDOW_END_ET):
        raise RuntimeError(
            f"Refusing to run at {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')} — "
            f"outside the safe ET window "
            f"[{SAFE_WINDOW_START_ET.strftime('%H:%M')}, "
            f"{SAFE_WINDOW_END_ET.strftime('%H:%M:%S')}]. "
            "Before 09:01 ET today's 09:00 bar may not be finalised in IB. "
            "After midnight ET the calendar day rolls and the next day's 9 AM "
            "bar doesn't exist yet. "
            "Override with --skip-time-guard for ad-hoc backfills."
        )


def _verify_prerequisites() -> None:
    """Hard-fail if any cold-start cache is missing.

    Without these, the daily pipeline would silently trigger a historical
    refetch (Databento bars, 5-year Mediastack pull, or a PCA re-fit that
    would shift the basis vectors and invalidate prior sentiment rows).
    """
    missing = [
        f"  - {path}  → {how}"
        for path, how in _REQUIRED_CACHES
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Daily orchestrator refuses to run: required historical caches missing.\n"
            "Cold-start each one before scheduling the daily cron.\n"
            + "\n".join(missing)
        )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _today_et() -> pd.Timestamp:
    """ET-midnight tz-naive timestamp for today — matches the parquet index."""
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


def _previous_trading_day(latest_ts: pd.Timestamp, ref: pd.Timestamp) -> pd.Timestamp | None:
    """Return the most recent date < ref where a labelled row exists.

    `latest_ts` is the max date present in the realized-range table (or None).
    """
    if latest_ts is None or pd.isna(latest_ts):
        return None
    if latest_ts >= ref:
        # latest_ts could be today if the function is misused — settle that
        # outside, never claim today as a finished day before 4 PM ET.
        return None
    return latest_ts


# ---------------------------------------------------------------------------
# Step 1+2: connect + backfill all data sources
# ---------------------------------------------------------------------------

async def _backfill_market_and_macro_via_ib() -> None:
    """Open one IB session, refresh ES + VX + ZN caches concurrently."""
    log.info("=" * 70)
    log.info("STEP 2a-c: IB incremental fetch for ES, VX, ZN")
    log.info("=" * 70)
    async with ib_session() as ib:
        # ES update is inlined (rather than calling market_data's sync wrapper)
        # so all three symbols share a single IB connection on this session.
        await _update_es_cache_on_session(ib)
        await update_macro_caches_incremental_async(ib)


async def _update_es_cache_on_session(ib) -> None:
    """Mirror market_data.update_es_cache_incremental on a shared IB session.

    Pins fetch `end` to today 09:01 ET (UTC) so the cache stops at today's
    09:00 bar regardless of when the script is invoked. See
    ib_client.today_9am_et_as_utc for the rationale.
    """
    from ib_client import fetch_continuous_bars, today_9am_et_as_utc

    if not os.path.exists(ES_RAW_PATH):
        raise FileNotFoundError(
            f"ES raw cache missing at {ES_RAW_PATH}. Cold-start via "
            "market_data.fetch_es_futures() first (Databento), then re-run."
        )
    cache = pd.read_parquet(ES_RAW_PATH)
    if cache.index.tz is None:
        cache.index = cache.index.tz_localize("UTC")
    cache_max = cache.index.max()
    end = today_9am_et_as_utc()
    start = cache_max + pd.Timedelta(minutes=1)
    if start >= end:
        log.info("[ES] cache already current (max %s, target end %s)", cache_max, end)
        return
    log.info("[ES] fetching incremental %s → %s (pinned 09:01 ET)", start, end)
    new = await fetch_continuous_bars(ib, "ES", start, end)
    if len(new) == 0:
        log.info("[ES] no new bars from IB")
        return
    combined = pd.concat([cache, new]).sort_index()
    n_dup = int(combined.index.duplicated().sum())
    if n_dup:
        log.info("[ES] removing %d duplicate timestamps after append", n_dup)
        combined = combined[~combined.index.duplicated(keep="first")]
    combined.to_parquet(ES_RAW_PATH, engine="pyarrow")
    added = len(combined) - len(cache)
    log.info("[ES] cached %d bars (+%d; through %s)",
             len(combined), added, combined.index.max())


def _backfill_news_and_sentiment() -> None:
    log.info("=" * 70)
    log.info("STEP 2d-e: Mediastack news + sentiment pipeline")
    log.info("=" * 70)
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise EnvironmentError("NEWS_API_KEY not set in environment")
    from news_data import fetch_news_incremental
    fetch_news_incremental(api_key=api_key)

    # sentiment-local.py uses a hyphenated filename and exposes
    # run_sentiment_pipeline; import it via importlib because the dash makes
    # it non-importable as a normal module name.
    import importlib.util
    sent_path = _PROJ_ROOT / "app" / "sentiment-local.py"
    spec = importlib.util.spec_from_file_location("sentiment_local", sent_path)
    sentiment_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sentiment_mod)
    sentiment_mod.run_sentiment_pipeline()


# ---------------------------------------------------------------------------
# Step 3: settle yesterday's prediction
# ---------------------------------------------------------------------------

def _settle_pending_predictions() -> int:
    """Fill in actual_range / abs_error for any logged prediction whose date
    now has a complete RTH session in the refreshed ES cache.

    Returns the number of rows settled this run.
    """
    log.info("=" * 70)
    log.info("STEP 3: settle pending predictions")
    log.info("=" * 70)
    if not PREDICTIONS_PATH.exists():
        log.info("No predictions.parquet yet — nothing to settle.")
        return 0

    preds = pd.read_parquet(PREDICTIONS_PATH)
    if "date" in preds.columns:
        preds = preds.set_index(pd.DatetimeIndex(preds["date"]).rename("date")).drop(columns=["date"])
    # Coerce numeric columns in case a prior run left them as object dtype.
    for col in ("predicted_range", "actual_range", "abs_error"):
        if col in preds.columns:
            preds[col] = pd.to_numeric(preds[col], errors="coerce")
    pending = preds[preds["actual_range"].isna()]
    if pending.empty:
        log.info("All predictions already settled.")
        return 0

    rr = pv.build_realized_range_table()
    rr = rr["realized_range"].dropna()

    settled = 0
    for date in pending.index:
        if date in rr.index:
            actual = float(rr.loc[date])
            predicted = float(preds.at[date, "predicted_range"])
            preds.at[date, "actual_range"] = actual
            preds.at[date, "abs_error"] = abs(predicted - actual)
            settled += 1
            log.info("  settled %s : pred=%.5f actual=%.5f |err|=%.5f",
                     date.date(), predicted, actual, abs(predicted - actual))

    if settled:
        out = preds.reset_index()
        out.to_parquet(PREDICTIONS_PATH, index=False)
        log.info("Settled %d prediction row(s) → %s", settled, PREDICTIONS_PATH)
    else:
        log.info("No settle-able rows yet (RTH bars may not be in cache).")
    return settled


# ---------------------------------------------------------------------------
# Step 4: rebuild feature parquets
# ---------------------------------------------------------------------------

def _rebuild_feature_parquets() -> None:
    log.info("=" * 70)
    log.info("STEP 4: rebuild feature parquets (keep_unsettled=True)")
    log.info("=" * 70)

    df_1m = pd.read_parquet(ES_RAW_PATH)
    if df_1m.index.tz is None:
        df_1m.index = df_1m.index.tz_localize("UTC")
    build_feature_matrix(df_1m, keep_unsettled=True)

    # Macro reads its own raw caches.
    macro_df = build_macro_features()
    MACRO_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    macro_df.to_parquet(MACRO_FEATURES_PATH, engine="pyarrow")
    log.info("macro: %d rows → %s", len(macro_df), MACRO_FEATURES_PATH)

    cal_df = build_calendar_features()
    CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cal_df.to_parquet(CAL_PATH, engine="pyarrow")
    log.info("calendar: %d rows → %s", len(cal_df), CAL_PATH)

    # Sentiment parquet is written by the sentiment pipeline itself in step 2.
    merged = merge_features(keep_unsettled=True)
    MODEL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MODEL_DATA_PATH, engine="pyarrow")
    log.info("model_data: %d rows × %d cols → %s",
             len(merged), merged.shape[1], MODEL_DATA_PATH)


# ---------------------------------------------------------------------------
# Step 5: retrain
# ---------------------------------------------------------------------------

def _should_full_search(meta_path: Path) -> bool:
    """Trigger a full Optuna search when no cached meta or stale by > N days."""
    if not meta_path.exists():
        log.info("No cached model meta — full Optuna search needed.")
        return True
    meta = json.loads(meta_path.read_text())
    last = meta.get("last_optuna_search") or meta.get("trained_at")
    if not last:
        log.info("Cached meta lacks last_optuna_search — full search needed.")
        return True
    last_ts = pd.Timestamp(last)
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    age_days = (pd.Timestamp.utcnow() - last_ts).total_seconds() / 86400.0
    log.info("Last Optuna search: %s (%.1f days ago)", last_ts, age_days)
    return age_days >= OPTUNA_REFRESH_DAYS


def _retrain(mode: str = MODE) -> dict:
    log.info("=" * 70)
    log.info("STEP 5: retrain vol model (mode=%s)", mode)
    log.info("=" * 70)
    meta_path = _PROJ_ROOT / "data" / f"vol_model_{mode}.json"
    force = _should_full_search(meta_path)
    return pv.train_or_refit(mode=mode, force_search=force)


# ---------------------------------------------------------------------------
# Steps 6+7+8: build today's features, predict, log
# ---------------------------------------------------------------------------

def _predict_and_log(model_meta: dict, mode: str = MODE) -> tuple[dict, pd.DataFrame, pd.Timestamp]:
    """Build today's row, predict, log. Returns (logged_row, feature_row, today).

    Returning the feature_row lets step 9 reuse the same single-row DataFrame
    for embed construction rather than reading the parquet again.
    """
    log.info("=" * 70)
    log.info("STEP 6-8: build today's features, predict, log")
    log.info("=" * 70)

    today = _today_et()
    log.info("Target prediction date (ET): %s", today.date())

    features = pv.build_features_for_date(today)
    log.info("Built today's feature row: %d columns", features.shape[1])

    bundle = pv.load_model_bundle(mode)
    pred = pv.predict_one(features, mode=mode, bundle=bundle)
    predicted_range = float(pred.iloc[0])
    log.info("Predicted realized range for %s: %.5f (%.2f bps of open)",
             today.date(), predicted_range, predicted_range * 10000)

    new_row = {
        "date": today,
        "predicted_range": predicted_range,
        # Numeric placeholders must be np.nan (not pd.NA) so the column is
        # written as float64 and later assignments don't silently flip the
        # whole column to object dtype.
        "actual_range": np.nan,
        "abs_error": np.nan,
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_mode": mode,
        "model_trained_at": bundle.get("trained_at"),
        "model_train_rows": bundle.get("train_rows"),
        "last_optuna_search": model_meta.get("last_optuna_search"),
    }

    if PREDICTIONS_PATH.exists():
        existing = pd.read_parquet(PREDICTIONS_PATH)
        # Belt-and-braces: if a legacy run wrote pd.NA, coerce numeric cols
        # back to float64 so .at[...] = float(...) keeps dtype stable.
        for col in ("predicted_range", "actual_range", "abs_error"):
            if col in existing.columns:
                existing[col] = pd.to_numeric(existing[col], errors="coerce")
    else:
        existing = pd.DataFrame(columns=list(new_row.keys()))

    if "date" in existing.columns and today in pd.DatetimeIndex(existing["date"]):
        # Same-day re-run: overwrite prior prediction for today.
        mask = pd.DatetimeIndex(existing["date"]) == today
        for k, v in new_row.items():
            existing.loc[mask, k] = v
        out = existing
        log.info("Overwrote existing prediction row for %s.", today.date())
    else:
        out = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        log.info("Appended new prediction row for %s.", today.date())

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTIONS_PATH, index=False)
    log.info("Predictions log: %d rows → %s", len(out), PREDICTIONS_PATH)
    return new_row, features, today


# ---------------------------------------------------------------------------
# Step 9: Discord forecast notification
# ---------------------------------------------------------------------------

# Trailing window size for the historical distributions that drive the
# embed's color band and news-tone calibration. 252 = roughly one year of
# trading days, the standard choice for vol baselines.
_HISTORY_WINDOW_DAYS = 252


def _send_forecast_notification(
    prediction_row: dict,
    feature_row: pd.DataFrame,
    today: pd.Timestamp,
    model_meta: dict,
) -> None:
    """Build the Discord embed and broadcast it to every forecast channel.

    Hard-fails (raises) if any webhook ultimately fails after retries — see
    notifications.send_forecast. The cron email then surfaces the issue.
    """
    log.info("=" * 70)
    log.info("STEP 9: Discord forecast notification")
    log.info("=" * 70)

    import notifications as notif

    # Trailing 252-day realized-range distribution for the color band.
    rr_table = pv.build_realized_range_table()
    realized = rr_table["realized_range"].dropna()
    realized = realized[realized.index < today]   # exclude today
    historical_ranges = realized.tail(_HISTORY_WINDOW_DAYS).to_numpy()

    # Trailing 252-day weighted_impact distribution for the tone band.
    model_data = pd.read_parquet(MODEL_DATA_PATH)
    if "weighted_impact" in model_data.columns:
        wi_series = model_data["weighted_impact"].dropna()
        wi_series = wi_series[wi_series.index < today]
        historical_wi = wi_series.tail(_HISTORY_WINDOW_DAYS).to_numpy()
    else:
        historical_wi = np.array([])

    # Most-recent settled prediction (any prior date) and rolling 7-day MAE.
    last_settled: dict | None = None
    rolling_mae_7d: float | None = None
    if PREDICTIONS_PATH.exists():
        preds = pd.read_parquet(PREDICTIONS_PATH)
        for col in ("predicted_range", "actual_range", "abs_error"):
            if col in preds.columns:
                preds[col] = pd.to_numeric(preds[col], errors="coerce")
        settled = (
            preds[preds["actual_range"].notna()]
            .copy()
            .assign(_date=lambda d: pd.to_datetime(d["date"]))
            .sort_values("_date")
        )
        if len(settled):
            last = settled.iloc[-1]
            last_settled = {
                "date": last["_date"],
                "predicted_range": float(last["predicted_range"]),
                "actual_range": float(last["actual_range"]),
            }
            rolling_mae_7d = float(settled.tail(7)["abs_error"].mean())

    embed = notif.build_forecast_embed(
        today_date=today,
        predicted_range=prediction_row["predicted_range"],
        feature_row=feature_row,
        historical_ranges=historical_ranges,
        historical_wi=historical_wi,
        last_settled=last_settled,
        rolling_mae_7d=rolling_mae_7d,
        model_meta=model_meta,
    )
    notif.send_forecast(embed)
    log.info("Forecast notification broadcast complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily vol prediction orchestrator.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit; no fetching, training, or writes.")
    parser.add_argument("--force-search", action="store_true",
                        help="Force a full Optuna search even if cached params are fresh.")
    parser.add_argument("--mode", default=MODE, choices=("mae", "aggressive"),
                        help="Vol-model mode (default: mae).")
    parser.add_argument("--skip-time-guard", action="store_true",
                        help="Bypass the safe-window ET check. Use only for "
                             "ad-hoc backfills or manual recovery — running "
                             "outside the safe window risks leakage or crashes.")
    parser.add_argument("--no-notify", action="store_true",
                        help="Skip Discord notifications entirely. Use for "
                             "ad-hoc backfills where you don't want to spam "
                             "the channel or fire spurious error pings.")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN — planned steps:")
        log.info("  0. Verify safe ET window [%s, %s) and prerequisite caches",
                 SAFE_WINDOW_START_ET.strftime("%H:%M"),
                 SAFE_WINDOW_END_ET.strftime("%H:%M"))
        log.info("  1. Connect to IB Gateway %s:%s", os.getenv("IB_HOST", "127.0.0.1"), os.getenv("IB_PORT", "4001"))
        log.info("  2. Backfill ES/VX/ZN bars + news + sentiment")
        log.info("  3. Settle pending predictions")
        log.info("  4. Rebuild feature parquets")
        log.info("  5. Retrain vol model (force_search=%s)", args.force_search)
        log.info("  6. Build today's feature row for %s", _today_et().date())
        log.info("  7. Predict today's realized range")
        log.info("  8. Append to %s", PREDICTIONS_PATH)
        log.info("  9. Broadcast forecast embed to Discord (--no-notify=%s)", args.no_notify)
        return

    started = datetime.now(timezone.utc)
    log.info("Daily orchestrator started at %s", started.isoformat())

    # `step_name` tracks where we are so the error-channel embed can name the
    # failing step. Updated immediately before each meaningful action.
    step_name = "init"
    try:
        step_name = "verify_time_window"
        if args.skip_time_guard:
            log.warning("--skip-time-guard set → bypassing safe-window check.")
        else:
            _verify_run_time_window()

        step_name = "verify_prerequisites"
        _verify_prerequisites()

        step_name = "ib_backfill"
        asyncio.run(_backfill_market_and_macro_via_ib())

        step_name = "news_and_sentiment"
        _backfill_news_and_sentiment()

        step_name = "settle_pending_predictions"
        _settle_pending_predictions()

        step_name = "rebuild_feature_parquets"
        _rebuild_feature_parquets()

        step_name = "retrain"
        if args.force_search:
            log.info("--force-search set → bypassing cadence check.")
            meta = pv.train_or_refit(mode=args.mode, force_search=True)
        else:
            meta = _retrain(mode=args.mode)

        step_name = "predict_and_log"
        prediction_row, feature_row, today = _predict_and_log(meta, mode=args.mode)

        step_name = "discord_notify"
        if args.no_notify:
            log.info("--no-notify set → skipping Discord forecast.")
        else:
            _send_forecast_notification(
                prediction_row=prediction_row,
                feature_row=feature_row,
                today=today,
                model_meta=meta,
            )
    except Exception:
        # Fire the error-channel notification (soft-fail inside) then re-raise
        # so the cron email still goes out and the script exits non-zero.
        import traceback as _tb
        tb_text = _tb.format_exc()
        log.error("Orchestrator failed during '%s':\n%s", step_name, tb_text)
        if not args.no_notify:
            try:
                import notifications as _notif
                _notif.send_error(traceback_text=tb_text, step_name=step_name)
            except Exception as _notify_exc:
                log.error("Error notification raised an unhandled exception: %s", _notify_exc)
        raise

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log.info("Daily orchestrator complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
