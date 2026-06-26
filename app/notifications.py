"""
Discord notification module for the daily vol-prediction orchestrator.

Two outbound channels
---------------------
- forecast_channels — one or more webhooks that receive the same morning
                      forecast embed. Same message broadcast across servers.
- error_channel     — a single webhook on a dedicated ops channel that
                      receives failure tracebacks. Kept private from the
                      forecast audience.

Configuration
-------------
A JSON file at app/discord_webhooks.json (override path via
DISCORD_WEBHOOK_CONFIG env var). Schema:

    {
      "forecast_channels": [
        {"label": "main",   "url": "https://discord.com/api/webhooks/..."},
        {"label": "team",   "url": "https://discord.com/api/webhooks/..."}
      ],
      "error_channel": {
        "label": "ops",
        "url": "https://discord.com/api/webhooks/..."
      }
    }

Failure semantics
-----------------
send_forecast        per-webhook retry with exponential backoff on 5xx/429;
                     per-webhook errors collected; HARD-FAIL at the end if
                     any webhook ultimately failed. Cron email surfaces the
                     issue. Override via --no-notify on daily_fetch.

send_error           soft-fail. If notification of the original error also
                     fails, log loud and return — the cron email is the
                     source-of-truth, Discord is a courtesy.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

_PROJ_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _PROJ_ROOT / "app" / "discord_webhooks.json"

# Embed left-stripe colors. Picked from Discord's official palette so they
# render natively rather than custom hex blobs.
COLOR_CALM = 0x3498DB    # blue
COLOR_NORMAL = 0x57F287  # green
COLOR_ACTIVE = 0xED4245  # red
COLOR_ERROR = 0xED4245   # red — also used for the error channel

MAX_RETRIES = 3
INITIAL_BACKOFF_SEC = 2.0
HTTP_TIMEOUT_SEC = 10.0

# Statuses that indicate permanent client errors. Retrying them is wasteful.
# 429 is excluded — it is transient by definition and honours Retry-After.
_PERMANENT_4XX = frozenset({400, 401, 403, 404, 410, 422})


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _config_path() -> Path:
    return Path(os.getenv("DISCORD_WEBHOOK_CONFIG", DEFAULT_CONFIG_PATH))


def load_config() -> dict:
    """Load the webhook config JSON.

    Raises FileNotFoundError with a remediation hint if the file is missing
    so the cron operator gets a clear message in the email.
    """
    path = _config_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Discord webhook config not found at {path}. Create it from "
            f"{path.with_suffix('.json.example')} or set DISCORD_WEBHOOK_CONFIG."
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Retry-aware HTTP post
# ---------------------------------------------------------------------------

def _post_with_retry(url: str, payload: dict, label: str) -> None:
    """POST a payload to a webhook with retry-with-backoff on transient errors.

    Permanent 4xx (config error) raises immediately; 429 honours Retry-After;
    5xx and network errors exponential-backoff up to MAX_RETRIES.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT_SEC)
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Webhook '{label}' network error after {MAX_RETRIES} attempts: {exc}"
                ) from exc
            delay = INITIAL_BACKOFF_SEC * (2 ** attempt)
            log.warning("Webhook '%s' network error: %s; retrying in %.1fs", label, exc, delay)
            time.sleep(delay)
            continue

        status = resp.status_code
        if 200 <= status < 300:
            return

        if status in _PERMANENT_4XX:
            raise RuntimeError(
                f"Webhook '{label}' permanent {status}: {resp.text[:300]}"
            )

        if status == 429:
            retry_after = resp.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else INITIAL_BACKOFF_SEC * (2 ** attempt)
            except ValueError:
                delay = INITIAL_BACKOFF_SEC * (2 ** attempt)
            log.warning("Webhook '%s' rate-limited; waiting %.1fs", label, delay)
            time.sleep(delay)
            continue

        if 500 <= status < 600:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Webhook '{label}' server error {status} after {MAX_RETRIES} attempts: "
                    f"{resp.text[:300]}"
                )
            delay = INITIAL_BACKOFF_SEC * (2 ** attempt)
            log.warning("Webhook '%s' returned %d; retry in %.1fs", label, status, delay)
            time.sleep(delay)
            continue

        # Unknown status — treat as permanent so the operator investigates.
        raise RuntimeError(
            f"Webhook '{label}' unexpected status {status}: {resp.text[:300]}"
        )

    raise RuntimeError(f"Webhook '{label}' exhausted retries.")


# ---------------------------------------------------------------------------
# Public send functions
# ---------------------------------------------------------------------------

def send_forecast(embed: dict, config: dict | None = None) -> None:
    """Broadcast a forecast embed to every configured forecast channel.

    Per-channel errors are collected; raises at the end iff any channel
    ultimately failed. Cron email surfaces partial failures.
    """
    config = config or load_config()
    channels = config.get("forecast_channels") or []
    if not channels:
        raise ValueError("webhook config has no 'forecast_channels' entries")

    payload = {"embeds": [embed]}
    failures: list[tuple[str, str]] = []
    for ch in channels:
        label = ch.get("label", "unlabeled")
        url = ch.get("url")
        if not url:
            failures.append((label, "missing url"))
            continue
        try:
            _post_with_retry(url, payload, label=label)
            log.info("Forecast posted to webhook '%s'", label)
        except Exception as exc:
            failures.append((label, str(exc)))
            log.error("Forecast failed for webhook '%s': %s", label, exc)

    if failures:
        details = "; ".join(f"{lbl} → {err}" for lbl, err in failures)
        raise RuntimeError(
            f"{len(failures)}/{len(channels)} forecast webhooks failed: {details}"
        )


def send_error(traceback_text: str, step_name: str, config: dict | None = None) -> None:
    """Best-effort error notification to the dedicated error channel.

    Soft-fail: logs but does not raise on failure. The cron email is the
    source-of-truth — we are only trying to push a friendlier surface.
    """
    try:
        config = config or load_config()
    except Exception as exc:
        log.error("Cannot load webhook config to send error: %s", exc)
        return

    err = config.get("error_channel")
    if not err or not err.get("url"):
        log.info("No error_channel configured; skipping error notification.")
        return

    now = datetime.now(timezone.utc)
    embed = {
        "title": "❌ daily_fetch.py failure",
        "description": (
            f"**Failed during:** `{step_name}`\n"
            f"**When:** {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            "```\n"
            f"{traceback_text[-1800:]}"
            "\n```"
        ),
        "color": COLOR_ERROR,
        "footer": {"text": "Investigate via cron email + parquet inspection."},
        "timestamp": now.isoformat(),
    }
    payload = {"embeds": [embed]}
    label = err.get("label", "error")
    try:
        _post_with_retry(err["url"], payload, label=label)
        log.info("Error notification sent to webhook '%s'", label)
    except Exception as exc:
        log.error(
            "Error notification ALSO failed (%s). Rely on the cron email.", exc,
        )


# ---------------------------------------------------------------------------
# Forecast embed construction
# ---------------------------------------------------------------------------

def _classify_band(predicted_range: float, historical_ranges: np.ndarray) -> tuple[str, int]:
    """Bin today's prediction against trailing distribution → (label, color).

    Boundaries are p25 and p75 of the trailing realized_range series. Picked
    on quartiles so the bands are roughly equiprobable rather than tuned to a
    specific volatility regime.
    """
    if len(historical_ranges) < 30:
        return "Normal", COLOR_NORMAL
    p25 = float(np.percentile(historical_ranges, 25))
    p75 = float(np.percentile(historical_ranges, 75))
    if predicted_range < p25:
        return "Calm", COLOR_CALM
    if predicted_range < p75:
        return "Normal", COLOR_NORMAL
    return "Active", COLOR_ACTIVE


def _tone_label(wi_today: float, wi_trailing: np.ndarray) -> str:
    """Calibrated tone band from percentile rank in the trailing 252-day
    weighted_impact distribution.

    Symmetric thresholds around the median: > 95% → extreme, > 80 → strong,
    > 65 → mild; analogous bearish bands below. Neutral covers the central
    ~30 % so noise around zero does not flip the label day to day.
    """
    if pd.isna(wi_today) or len(wi_trailing) < 30:
        return "—"
    pct = float((wi_trailing < wi_today).mean() * 100)
    if pct > 95:
        return "Extreme Bullish"
    if pct > 80:
        return "Strong Bullish"
    if pct > 65:
        return "Mild Bullish"
    if pct >= 35:
        return "Neutral"
    if pct >= 20:
        return "Mild Bearish"
    if pct >= 5:
        return "Strong Bearish"
    return "Extreme Bearish"


def _fmt_bps(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value * 10000:.0f} bps"


def _fmt_pct(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def _safe_get(row: pd.Series, key: str, default=float("nan")):
    val = row.get(key, default)
    if isinstance(val, (np.floating, float)) and pd.isna(val):
        return float("nan")
    return val


def build_forecast_embed(
    today_date: pd.Timestamp,
    predicted_range: float,
    feature_row: pd.DataFrame,
    historical_ranges: np.ndarray,
    historical_wi: np.ndarray,
    last_settled: dict | None,
    rolling_mae_7d: float | None,
    model_meta: dict,
) -> dict:
    """Build the Discord rich embed dict for one forecast.

    feature_row is a single-row DataFrame (the same row used for prediction);
    its columns supply news/calendar/baseline values. historical_ranges and
    historical_wi are trailing 252-day numpy arrays used for binning.
    last_settled is the most-recent settled prediction (any prior date), not
    necessarily yesterday — captures the "did the model nail the last one"
    feedback loop even on a Monday after a long weekend.
    """
    row = feature_row.iloc[0]
    band_name, color = _classify_band(predicted_range, historical_ranges)

    predicted_bps_str = _fmt_bps(predicted_range)
    predicted_pct_str = _fmt_pct(predicted_range, 2)

    # Confidence band uses cached test-set MAE as the ±width. Cached value is
    # ~0.003 (30 bps) on the May 2026 numbers; fall back to that constant if
    # the meta has no test_metrics yet (first cold-start run).
    mae = 0.003
    coverage = None
    try:
        model_metrics = model_meta["test_metrics"]["model"]
        mae = float(model_metrics["mae"])
        cov = model_metrics.get("coverage_at_mae")
        if cov is not None and not pd.isna(cov):
            coverage = float(cov)
    except (KeyError, TypeError, ValueError):
        pass
    low_bps = max(0.0, (predicted_range - mae) * 10000)
    high_bps = (predicted_range + mae) * 10000

    # Coverage is the empirically-measured hit-rate of the ±MAE band on the
    # held-out test set (see predict_vol._metrics). Omit the percentage if an
    # older cache predates the metric rather than printing a stale constant.
    conf_suffix = f" (~{coverage * 100:.0f}% confidence)" if coverage is not None else ""
    description = (
        f"**Predicted range:** {predicted_bps_str} ({predicted_pct_str})\n"
        f"**Read:** {band_name} day expected\n"
        f"Likely range: {low_bps:.0f}–{high_bps:.0f} bps{conf_suffix}"
    )

    fields: list[dict] = []

    # ----- Last settled day --------------------------------------------------
    if last_settled is not None:
        ye_pred = float(last_settled["predicted_range"])
        ye_act = float(last_settled["actual_range"])
        ye_err = abs(ye_pred - ye_act)
        ye_date_str = pd.Timestamp(last_settled["date"]).strftime("%Y-%m-%d")
        ye_lines = [
            f"**{ye_date_str}**",
            f"Pred: {_fmt_bps(ye_pred)}",
            f"Actual: {_fmt_bps(ye_act)}",
            f"Error: {_fmt_bps(ye_err)}",
        ]
        if rolling_mae_7d is not None and not pd.isna(rolling_mae_7d):
            ye_lines.append(f"7d MAE: {_fmt_bps(rolling_mae_7d)}")
        fields.append({
            "name": "Last Settled",
            "value": "\n".join(ye_lines),
            "inline": True,
        })

    # ----- Calendar / macro context -----------------------------------------
    days_to_fomc = _safe_get(row, "days_to_fomc")
    days_to_opex = _safe_get(row, "days_to_opex")
    is_fomc = bool(_safe_get(row, "is_fomc_day", 0) or 0)
    is_qopex = bool(_safe_get(row, "is_quarterly_opex", 0) or 0)
    vix_level = _safe_get(row, "vix_level")
    cal_lines = [f"**{pd.Timestamp(today_date).day_name()}**"]
    if not pd.isna(days_to_fomc):
        suffix = " (today)" if is_fomc else ""
        cal_lines.append(f"Days to FOMC: {int(days_to_fomc)}{suffix}")
    if not pd.isna(days_to_opex):
        suffix = " (quarterly)" if is_qopex else ""
        cal_lines.append(f"Days to OPEX: {int(days_to_opex)}{suffix}")
    if not pd.isna(vix_level):
        cal_lines.append(f"VIX @ 09:00: {float(vix_level):.1f}")
    fields.append({
        "name": "Calendar / Macro",
        "value": "\n".join(cal_lines),
        "inline": True,
    })

    # ----- News block --------------------------------------------------------
    wi = _safe_get(row, "weighted_impact")
    pct_pos = _safe_get(row, "prob_pos_mean")
    pct_neg = _safe_get(row, "prob_neg_mean")
    pct_neu = _safe_get(row, "prob_neu_mean")
    n_articles_raw = _safe_get(row, "article_count", 0)
    n_articles = int(0 if pd.isna(n_articles_raw) else n_articles_raw)
    buzz = _safe_get(row, "buzz_factor")
    momentum = _safe_get(row, "sentiment_momentum")
    ewma = _safe_get(row, "sentiment_ewma_5d")
    is_top_tail = bool(_safe_get(row, "is_top_2pct_weighted_impact", 0) or 0)
    is_bot_tail = bool(_safe_get(row, "is_bot_2pct_weighted_impact", 0) or 0)

    tone = _tone_label(wi, historical_wi)
    news_lines: list[str] = []
    if not pd.isna(wi):
        news_lines.append(f"**Tone: {tone}**  ({wi:+.3f} weighted impact)")
    else:
        news_lines.append(f"**Tone: {tone}**")
    if not pd.isna(buzz):
        news_lines.append(f"Volume: {n_articles} articles ({buzz:.1f}× normal)")
    else:
        news_lines.append(f"Volume: {n_articles} articles")
    if not pd.isna(pct_pos) and not pd.isna(pct_neg) and not pd.isna(pct_neu):
        news_lines.append(
            f"Mix: {pct_pos * 100:.0f}% pos / {pct_neg * 100:.0f}% neg / {pct_neu * 100:.0f}% neu"
        )
    mo_parts = []
    if not pd.isna(momentum):
        mo_parts.append(f"Momentum {momentum:+.3f}")
    if not pd.isna(ewma):
        mo_parts.append(f"5d EWMA {ewma:+.3f}")
    if mo_parts:
        news_lines.append("  •  ".join(mo_parts))
    if is_top_tail:
        news_lines.append("⚠️ Top-2% bullish day (contrarian flag — historically goes UP only ~40% of the time)")
    if is_bot_tail:
        news_lines.append("⚠️ Bottom-2% bearish day (contrarian flag fired)")
    fields.append({
        "name": "News (4 PM yest → 9 AM today)",
        "value": "\n".join(news_lines),
        "inline": False,
    })

    # ----- Baselines ---------------------------------------------------------
    persist = _safe_get(row, "prev_day_range")
    rv5 = _safe_get(row, "rv_lag_5")
    baseline_lines: list[str] = []
    if not pd.isna(persist):
        edge = (predicted_range - persist) * 10000
        baseline_lines.append(
            f"Persistence: {_fmt_bps(persist)}  (model edge {edge:+.0f} bps)"
        )
    if not pd.isna(rv5):
        edge = (predicted_range - rv5) * 10000
        baseline_lines.append(
            f"5-day MA: {_fmt_bps(rv5)}  (model edge {edge:+.0f} bps)"
        )
    if baseline_lines:
        fields.append({
            "name": "Baselines (today's pred vs naive predictors)",
            "value": "\n".join(baseline_lines),
            "inline": False,
        })

    # ----- Model provenance --------------------------------------------------
    trained_at = model_meta.get("trained_at", "?")
    last_search = model_meta.get("last_optuna_search", "?")
    train_rows = model_meta.get("train_rows", "?")
    train_range = model_meta.get("train_date_range") or ["?", "?"]
    fields.append({
        "name": "Model",
        "value": (
            f"Last refit: {trained_at}\n"
            f"Last full search: {last_search}\n"
            f"Trained on {train_rows} days ({train_range[0]} → {train_range[1]})"
        ),
        "inline": False,
    })

    return {
        "title": f"📊 S&P 500 Vol Forecast — {pd.Timestamp(today_date).strftime('%A %Y-%m-%d')}",
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {"text": "daily_fetch.py"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
