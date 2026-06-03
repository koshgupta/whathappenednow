"""
Macro features for the 9 AM snapshot — VIX, 10Y Treasury (futures-proxied).

DXY (Dollar Index) was REMOVED 2026-05-09: `DX.c.0` on `IFUS.IMPACT` does not
roll continuously — Databento returns ~5 weeks of bars per quarterly contract
followed by ~8 weeks of nothing until the next contract becomes front-month,
covering only ~43 % of weekdays. Switching to a different feed (e.g. Alpha
Vantage spot DXY) is tracked separately. The cached raw parquet is left on
disk in case we revisit.

Output: data/macro_features.parquet
  index   : date  (datetime64[ns], midnight-normalised, tz-naive, name="date")
  columns : vix_overnight, tenY_overnight,
            vix_dist_sma_20, tenY_dist_sma_20

ANTI-LOOKAHEAD PROTOCOL
-----------------------
Every value at row T depends ONLY on data observable strictly before 9:00:00 AM
ET on day T. Three sources of values, three guarantees:

  snapshot_9am   = price AT 09:00:00 ET, latest available. If the 09:00 bar
                   exists, its OPEN (price at the first tick of 9:00). Else
                   the CLOSE of the latest bar with timestamp < 09:00 ET
                   (price at the end of that minute, i.e. the start of the
                   next). 1m bars on liquid futures are sometimes empty when
                   no trade prints — this asof rule keeps the snapshot well-
                   defined while never reading a price after 09:00:00.
  prev_close_4pm = close of the 15:59 ET bar from the PRIOR trading day
                   (left-labeled, ends at 16:00 — the official RTH close
                   anchor). Same asof fallback: if 15:59 is empty, use the
                   close of the latest bar with timestamp ≤ 15:59 ET.
  sma_basis      = 20-day rolling mean of 4 PM closes, then SHIFTED FORWARD 1 DAY
                   so today's basis covers T-20..T-1 only. Today's 4 PM close is
                   not knowable at 9 AM regardless, but the explicit shift(1)
                   makes the guarantee mechanical and auditable.

There is no path by which a value from on-or-after 9:00 AM ET on day T can flow
into row T's features.

STATIONARITY
------------
All six features are stationary by construction (percent change or ratio − 1).
Raw price levels are intentionally OMITTED:
  - DXY trends for years (~70 in 2008, ~120 in 1985) — train/test ranges may not
    overlap, breaking generalisation.
  - 10Y yield trended 15% → 0.5% over 40 years then jumped back to 5%.
  - VIX is the only one of the three that is approximately stationary; we still
    skip the raw level for symmetry — the SMA-distance form captures the same
    "elevated vs recent norm" signal without committing to level scale.

10Y YIELD PROXY VIA ZN PRICE
----------------------------
The 10Y "yield" features are derived from ZN (10Y Note) futures PRICE, since
yield is not directly traded. Bond price moves INVERSELY to yield: a price gain
overnight means yields fell. We sign-flip the ZN features so the published
columns read in the natural yield direction:

    tenY_overnight    = -((zn_9am / zn_prev_4pm) - 1)
                      → POSITIVE means yields rose overnight.
    tenY_dist_sma_20  = -((zn_9am / zn_sma20_lagged) - 1)
                      → POSITIVE means today's yield is above the 20d norm.

Magnitudes are NOT in basis points — they are unit-less price-percent equivalents.
For interpretation, yield_change_bps ≈ tenY_overnight × duration_yrs × 10000,
duration ≈ 7.5 yr for the on-the-run 10Y. XGBoost only needs ordering, so the
unit difference does not affect modelling.

DATABENTO COVERAGE
------------------
  ZN  (10Y Note futures, CME)              → GLBX.MDP3   CONFIRMED, fixed-point.
  VX  (VIX futures, Cboe Futures Exchange) → XCBF.PITCH  CONFIRMED on first fetch
                                                          (continuous-symbol roll
                                                          overlap dedup'd at fetch).
  DX  (US Dollar Index futures, ICE US)    → IFUS.IMPACT REJECTED — c.0 fails to
                                                          roll, ~57 % weekday gap.

If you swap a series for an alternate dataset/symbol pair, everything downstream
(anchors, features) is provider-agnostic and only requires a UTC-indexed 1m
OHLCV DataFrame.

JOINABILITY
-----------
Index name 'date', dtype datetime64[ns], midnight-normalised, tz-naive — same
shape as features_9am_snapshot.parquet and calendar_features.parquet. Merge
with an inner join on date so that rows missing in any source (holidays in a
given exchange, fetch gaps) are dropped safely.

NEVER EXECUTED BY CLAUDE — see CLAUDE.md. The user runs this script.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict

import databento as db
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

load_dotenv()

_PROJ_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJ_ROOT / "data"
OUTPUT_PATH = _DATA_DIR / "macro_features.parquet"

LOOKBACK_YEARS = 5          # match market_data.py training window
WARMUP_YEARS = 1            # extra year fetched so SMA-20 has clean history at the cutoff
SMA_WINDOW = 20             # one trading month
SCHEMA = "ohlcv-1m"

# GLBX returns int64 fixed-point at 1e9; non-GLBX feeds are typically already float.
PRICE_FACTOR = 1e9


# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceCfg:
    dataset: str        # Databento dataset code
    symbol: str         # continuous-front-month convention varies by venue
    fixed_point: bool   # True if prices come back as int64 scaled by 1e9 (GLBX)
    invert_sign: bool   # True for ZN (price ↔ inverse of yield)


SOURCES: Dict[str, SourceCfg] = {
    "vix": SourceCfg(
        dataset="XCBF.PITCH",
        symbol="VX.c.0",
        fixed_point=False,
        invert_sign=False,
    ),
    "tenY": SourceCfg(
        dataset="GLBX.MDP3",
        symbol="ZN.c.0",
        fixed_point=True,
        invert_sign=True,         # ZN price up ⇒ yield down — flip so feature reads as yield direction
    ),
}

# Mapping from macro source name → IB ContFuture symbol (used for incremental
# updates only; the cold-start path stays on Databento per SOURCES above).
IB_SYMBOL_FOR_SOURCE: Dict[str, str] = {
    "vix": "VX",
    "tenY": "ZN",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch (per-series cache)
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> Path:
    return _DATA_DIR / f"macro_{name}_intraday_raw.parquet"


def _dedupe_keep_highest_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate-timestamp bars to the highest-volume row per minute.

    Databento's continuous-symbol resolution (`stype_in="continuous"`, e.g.
    `VX.c.0`) can return bars from BOTH the front and the next-front contract
    on the same minute when multiple maturities trade simultaneously. The
    higher-volume row is the active front month — `c.0`'s intended meaning.
    No-op when the index is already unique (e.g. GLBX's ZN).
    """
    if not df.index.duplicated().any():
        return df
    name = df.index.name or "index"
    reset = df.reset_index()
    keep = reset.groupby(name, sort=False)["volume"].idxmax()
    return reset.loc[keep].set_index(name).sort_index()


def fetch_series(name: str, cfg: SourceCfg) -> pd.DataFrame:
    """
    Fetch (LOOKBACK_YEARS + WARMUP_YEARS) of 1 min OHLCV bars for one macro
    series and cache to parquet. Returns a UTC-indexed DataFrame with columns
    open/high/low/close/volume (low/high/volume not used downstream but kept for
    parity with market_data.py and any future extension).

    Self-heals legacy caches that contain duplicate timestamps from
    continuous-symbol contract overlap: dedupe in memory and rewrite the
    parquet so future loads are O(clean).
    """
    cache = _cache_path(name)
    if cache.exists():
        log.info("[%s] loading cached raw bars from %s", name, cache)
        df = pd.read_parquet(cache)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if df.index.duplicated().any():
            n_before = len(df)
            df = _dedupe_keep_highest_volume(df)
            log.info("[%s] cache had duplicate timestamps: %d → %d rows; rewriting clean cache", name, n_before, len(df))
            df.to_parquet(cache, engine="pyarrow")
        return df

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise EnvironmentError("DATABENTO_API_KEY not set in environment")

    end = datetime.now() - timedelta(days=1)
    start = end - relativedelta(years=LOOKBACK_YEARS + WARMUP_YEARS)

    log.info(
        "[%s] fetching %s %s from %s to %s via Databento (%s)...",
        name, cfg.symbol, SCHEMA, start.date(), end.date(), cfg.dataset,
    )

    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset=cfg.dataset,
        symbols=[cfg.symbol],
        schema=SCHEMA,
        start=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end=end.strftime("%Y-%m-%dT%H:%M:%S"),
        stype_in="continuous",
    )

    df = data.to_df()
    df.index = pd.to_datetime(df.index, utc=True)

    if cfg.fixed_point:
        for col in ("open", "high", "low", "close"):
            if col in df.columns and df[col].dtype == np.int64:
                df[col] = df[col] / PRICE_FACTOR

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].sort_index()
    df = _dedupe_keep_highest_volume(df)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, engine="pyarrow")
    log.info("[%s] %d bars cached → %s", name, len(df), cache)

    return df


# ---------------------------------------------------------------------------
# Incremental update via IB (used by daily_fetch.py)
# ---------------------------------------------------------------------------

async def update_one_cache_async(ib, source_name: str) -> pd.DataFrame:
    """Extend one macro series's local 1m cache to ~now via IB Gateway.

    Hard-fails if no cache yet. The Databento cold-start path is intentionally
    not auto-triggered: daily orchestrator runs must never silently refetch
    multi-year history. Cold-start once via fetch_series() (Databento),
    then this update path is safe to schedule.
    """
    if source_name not in IB_SYMBOL_FOR_SOURCE:
        raise ValueError(f"Unknown macro source {source_name!r}")
    cache_path = _cache_path(source_name)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"[{source_name}] no cache at {cache_path}. Cold-start once via "
            "Databento (call fetch_series(name, SOURCES[name])) before "
            "scheduling the daily orchestrator."
        )

    cache = pd.read_parquet(cache_path)
    if cache.index.tz is None:
        cache.index = cache.index.tz_localize("UTC")

    # Local import keeps the cold-start Databento path independent of ib_async.
    from ib_client import fetch_continuous_bars, today_9am_et_as_utc

    cache_max = cache.index.max()
    end = today_9am_et_as_utc()
    start = cache_max + pd.Timedelta(minutes=1)
    if start >= end:
        log.info("[%s] cache already current (max %s, target end %s)",
                 source_name, cache_max, end)
        return cache

    log.info("[%s] fetching IB incremental %s → %s (pinned 09:01 ET)",
             source_name, start, end)
    new = await fetch_continuous_bars(ib, IB_SYMBOL_FOR_SOURCE[source_name], start, end)
    if len(new) == 0:
        log.info("[%s] no new bars from IB", source_name)
        return cache

    combined = pd.concat([cache, new]).sort_index()
    combined = _dedupe_keep_highest_volume(combined)
    combined.to_parquet(cache_path, engine="pyarrow")
    added = len(combined) - len(cache)
    log.info(
        "[%s] cached %d bars (+%d; range %s → %s) → %s",
        source_name, len(combined), added,
        combined.index.min(), combined.index.max(), cache_path,
    )
    return combined


async def update_macro_caches_incremental_async(ib) -> Dict[str, pd.DataFrame]:
    """Extend all macro raw caches in parallel on a shared IB connection."""
    results = await asyncio.gather(
        *(update_one_cache_async(ib, name) for name in IB_SYMBOL_FOR_SOURCE)
    )
    return dict(zip(IB_SYMBOL_FOR_SOURCE.keys(), results))


def update_macro_caches_incremental() -> Dict[str, pd.DataFrame]:
    """Sync wrapper: open IB connection, refresh all macro caches, close.

    Use the async variant directly when sharing a connection with other
    fetchers (e.g. the daily_fetch orchestrator).
    """
    from ib_client import ib_session

    async def _run():
        async with ib_session() as ib:
            return await update_macro_caches_incremental_async(ib)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Anchors and features
# ---------------------------------------------------------------------------

def _to_et(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a US/Eastern DatetimeIndex.

    Critical for anti-lookahead: filtering on ET clock-times AFTER conversion
    means '9:00 AM' is the actual New York 9:00 AM — never a UTC 9 AM (which
    is 4 AM ET in winter and would silently insert future data).
    """
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    return df


def _asof_last_per_date(
    df_et: pd.DataFrame,
    cutoff: time,
) -> pd.DataFrame:
    """Return the latest bar per ET-date with timestamp ≤ cutoff (inclusive).

    Uses pd.between_time which is inclusive on both ends, so a bar exactly at
    cutoff is kept. Result is indexed by the bar's original ET timestamp.
    """
    window = df_et.between_time(time(0, 0), cutoff)
    return window.groupby(window.index.normalize(), sort=False).tail(1)


def build_daily_anchors(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    From 1 min UTC bars, extract per-session anchors using an as-of rule that
    tolerates empty 1m bars (no print in that minute):

      snapshot_9am  = OPEN of the 09:00 ET bar if present (price at 09:00:00),
                      else CLOSE of the latest bar with timestamp < 09:00 ET
                      (price at the end of that minute).
      close_4pm     = CLOSE of the latest bar with timestamp ≤ 15:59 ET. The
                      15:59 bar's close is at 16:00:00 — the official RTH
                      close anchor when present.

    Both anchors are anti-lookahead-safe: only bars whose timestamp is at or
    before the cutoff are eligible.

    Returns a date-indexed DataFrame (weekdays only). Days where the futures
    market had no bars at all in the morning/afternoon window are absent —
    the downstream merge's inner join will drop them.
    """
    df_et = _to_et(df_1m)

    morning = _asof_last_per_date(df_et, time(9, 0))
    is_exact_9 = morning.index.time == time(9, 0)
    snap_values = np.where(is_exact_9, morning["open"].values, morning["close"].values)
    snap = pd.Series(
        snap_values,
        index=pd.DatetimeIndex(morning.index.date),
        name="snapshot_9am",
    )

    afternoon = _asof_last_per_date(df_et, time(15, 59))
    close = pd.Series(
        afternoon["close"].values,
        index=pd.DatetimeIndex(afternoon.index.date),
        name="close_4pm",
    )

    daily = pd.concat([snap, close], axis=1).sort_index()
    daily = daily[daily.index.dayofweek < 5]
    return daily


def build_features_for_series(
    daily: pd.DataFrame,
    name: str,
    invert_sign: bool,
    sma_window: int = SMA_WINDOW,
) -> pd.DataFrame:
    """
    Compute the two stationary features for a single series.

    overnight   = (snapshot_9am / prev_close_4pm) − 1      (* −1 if invert_sign)
    dist_sma_N  = (snapshot_9am / sma_of_N_prior_closes) − 1  (* −1 if invert_sign)

    Anti-lookahead: BOTH bases (prev 4 PM close, SMA20 of past closes) are
    explicitly lagged by one trading day before the snapshot is divided into
    them. Even if today's 4 PM close were somehow present in the input (it
    isn't — we don't have it at 9 AM), the shift(1) on close_4pm guarantees
    it cannot enter the calculation.
    """
    snap = daily["snapshot_9am"]
    prev_close = daily["close_4pm"].shift(1)
    sma_basis = daily["close_4pm"].rolling(sma_window).mean().shift(1)

    sign = -1.0 if invert_sign else 1.0
    overnight = sign * ((snap / prev_close) - 1.0)
    dist_sma = sign * ((snap / sma_basis) - 1.0)

    return pd.DataFrame({
        f"{name}_overnight": overnight,
        f"{name}_dist_sma_{sma_window}": dist_sma,
    })


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_macro_features() -> pd.DataFrame:
    """
    Pipeline: per-source fetch → daily anchors → stationary features → join.
    Output is restricted to the LOOKBACK_YEARS training window (warm-up
    discarded) but rows with NaN are KEPT so the merge step can decide how to
    treat per-source holiday gaps.
    """
    parts: list[pd.DataFrame] = []
    for name, cfg in SOURCES.items():
        bars = fetch_series(name, cfg)
        anchors = build_daily_anchors(bars)
        feats = build_features_for_series(anchors, name, cfg.invert_sign)
        log.info(
            "[%s] %d daily rows | nan overnight=%d | nan dist_sma_%d=%d",
            name, len(feats),
            int(feats[f"{name}_overnight"].isna().sum()),
            SMA_WINDOW,
            int(feats[f"{name}_dist_sma_{SMA_WINDOW}"].isna().sum()),
        )
        parts.append(feats)

    df = pd.concat(parts, axis=1, join="outer").sort_index()
    df.index.name = "date"

    cutoff = pd.Timestamp(datetime.now() - relativedelta(years=LOOKBACK_YEARS)).normalize()
    df = df[df.index >= cutoff]

    return df
