"""
Merge market + macro + sentiment + calendar feature parquets into model_data.parquet.

INPUTS (date-aligned at 9 AM ET; calendar is a date superset):
  data/features_9am_snapshot.parquet  market: ES technicals + intraday + y target
  data/macro_features.parquet         VIX + 10Y Treasury (futures-proxied)
  data/sentiment_features.parquet     daily news aggregates + lags + PCA
  data/calendar_features.parquet      DOW one-hots + FOMC/OPEX distances

OUTPUT:
  data/model_data.parquet             inner join on date, ready for predictor.py

JOIN POLICY
-----------
Inner join on a tz-naive midnight `date` index. Sentiment carries 'Date' as a
column (not the index) — we promote it to the index here. Every other source
already exposes `date` as the named DatetimeIndex.

The market frame is the canonical base because it carries the `y` target. The
final output drops rows with NaN `y` so the training set is clean. Once the
predictor moves to scoring at 9 AM (no `y` for "today"), this should switch to
a left-join on market with `y` left as NaN — see TODO.

CUTOFF
------
Optionally cap the output at a date — useful for "last completed RTH day"
slicing when a partial-current-day row would otherwise pollute the tail. The
inner join already drops anything not present in every source, so the cap is
mostly defensive.

NEVER FETCHES — pure local read/merge/write. Safe to run anytime.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parent.parent
_DATA = _PROJ_ROOT / "data"

MARKET_PATH = _DATA / "features_9am_snapshot.parquet"
MACRO_PATH = _DATA / "macro_features.parquet"
SENTIMENT_PATH = _DATA / "sentiment_features.parquet"
CALENDAR_PATH = _DATA / "calendar_features.parquet"
OUTPUT_PATH = _DATA / "model_data.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _check_date_index(df: pd.DataFrame, source: str) -> None:
    if df.index.name != "date":
        raise ValueError(f"{source}: expected index.name == 'date', got {df.index.name!r}")
    if df.index.dtype.kind != "M":
        raise ValueError(f"{source}: expected datetime64 index, got {df.index.dtype}")
    if df.index.tz is not None:
        raise ValueError(f"{source}: expected tz-naive index, got tz={df.index.tz}")
    if not df.index.is_unique:
        raise ValueError(f"{source}: index has {int(df.index.duplicated().sum())} duplicate dates")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{source}: index is not sorted")


def _load_market() -> pd.DataFrame:
    df = pd.read_parquet(MARKET_PATH)
    _check_date_index(df, "market")
    if "y" not in df.columns:
        raise ValueError("market frame missing 'y' target column")
    log.info("market   : %d rows %s → %s", len(df), df.index.min().date(), df.index.max().date())
    return df


def _load_macro() -> pd.DataFrame:
    df = pd.read_parquet(MACRO_PATH)
    _check_date_index(df, "macro")
    log.info("macro    : %d rows %s → %s", len(df), df.index.min().date(), df.index.max().date())
    return df


def _load_sentiment() -> pd.DataFrame:
    df = pd.read_parquet(SENTIMENT_PATH)
    if "Date" not in df.columns:
        raise ValueError("sentiment frame missing 'Date' column to promote to index")
    df = df.set_index(pd.DatetimeIndex(df["Date"]).rename("date")).drop(columns=["Date"])
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    _check_date_index(df, "sentiment")
    log.info("sentiment: %d rows %s → %s", len(df), df.index.min().date(), df.index.max().date())
    return df


def _load_calendar() -> pd.DataFrame:
    df = pd.read_parquet(CALENDAR_PATH)
    _check_date_index(df, "calendar")
    log.info("calendar : %d rows %s → %s", len(df), df.index.min().date(), df.index.max().date())
    return df


def merge_features(cutoff: pd.Timestamp | None = None, keep_unsettled: bool = False) -> pd.DataFrame:
    """Merge all source parquets on date.

    keep_unsettled
    --------------
    False (default): drop rows with NaN target — the standard training set.
    True           : keep rows whose only missing piece is the target (e.g.
                     today's unsettled row, used by daily_fetch.py to score
                     today's 9 AM snapshot).
    """
    market = _load_market()
    macro = _load_macro()
    sentiment = _load_sentiment()
    calendar = _load_calendar()

    overlaps = (
        set(market.columns) & set(macro.columns)
        | set(market.columns) & set(sentiment.columns)
        | set(market.columns) & set(calendar.columns)
        | set(macro.columns) & set(sentiment.columns)
        | set(macro.columns) & set(calendar.columns)
        | set(sentiment.columns) & set(calendar.columns)
    )
    if overlaps:
        raise ValueError(f"column name collisions across sources: {sorted(overlaps)}")

    df = market.join(macro, how="inner").join(sentiment, how="inner").join(calendar, how="inner")
    log.info("inner-join: %d rows × %d cols  %s → %s",
             len(df), df.shape[1], df.index.min().date(), df.index.max().date())

    if cutoff is not None:
        df = df[df.index <= pd.Timestamp(cutoff).normalize()]
        log.info("after cutoff %s: %d rows", cutoff, len(df))

    if not keep_unsettled:
        n_before = len(df)
        df = df[df["y"].notna()]
        if n_before != len(df):
            log.info("dropped %d rows with NaN y (training requires a label)", n_before - len(df))
    else:
        n_unsettled = int(df["y"].isna().sum())
        if n_unsettled:
            log.info("keeping %d unsettled rows (y is NaN) for prediction use", n_unsettled)

    nan_per_col = df.isna().sum()
    nan_cols = nan_per_col[nan_per_col > 0]
    if len(nan_cols):
        log.info("NaN counts per column (non-zero only):")
        for col, n in nan_cols.items():
            log.info("  %-25s %d", col, n)
    else:
        log.info("no NaNs in any feature or target")

    return df
