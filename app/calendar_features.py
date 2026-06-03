"""
Calendar / structural features for the 9 AM snapshot feature matrix.

Output : data/calendar_features.parquet
  index   : date  (datetime64[ns], normalised to midnight, name="date")
  columns :
    dow_mon, dow_tue, dow_wed, dow_thu, dow_fri  (uint8 one-hot)
    days_to_fomc       (float64, NaN beyond the last known scheduled meeting)
    is_fomc_day        (uint8)
    days_to_opex       (int32, days until next 3rd-Friday monthly equity opex,
                        0 on the opex day itself)
    is_quarterly_opex  (uint8, 1 if today is the 3rd Friday of Mar/Jun/Sep/Dec
                        — "triple/quad witching")

ANTI-LOOKAHEAD PROTOCOL
-----------------------
Every value in this table is a deterministic function of the calendar date T
alone — no future prices, returns, or events touch any feature. Each row's
values are knowable months to years before T:
  - day_of_week       : determined by the date itself (since 1582).
  - days_to_fomc      : the Fed publishes its schedule ~12 months in advance;
                        for any date covered here, the next scheduled meeting
                        was already public on or before T. Emergency /
                        unscheduled meetings are intentionally EXCLUDED — they
                        carried no anticipatory effect at the time, so feeding
                        them here would be retroactive knowledge insertion.
  - days_to_opex      : 3rd Friday of each Gregorian month — fixed centuries
                        ahead.
  - is_*_*            : binary indicators of the same date facts.

Therefore there is no path by which a future label can leak into a past row.
The merge step that joins this table into the model feature matrix must do an
inner join on date so dates outside the trading-day index (weekends already
excluded by bdate_range; market holidays excluded by the futures index) are
dropped naturally.

JOINABILITY
-----------
Index name   : "date"   (matches features_9am_snapshot.parquet)
Index dtype  : datetime64[ns], midnight-normalised, tz-naive
Index values : every Mon-Fri in [DEFAULT_START, DEFAULT_END]; market holidays
               are NOT removed here because we can't know them without a
               holiday calendar dependency. The merge step's inner join on the
               trading-day index will drop them.

KNOWN EDGE CASES
----------------
- If a 3rd Friday falls on Good Friday (e.g. April 2025), monthly equity opex
  effectively settles on the prior trading day. This script reports the pure
  calendar 3rd Friday; the actual settlement-day adjustment happens at most
  ~once a year and is left to a follow-up if it proves material.
- FOMC dates beyond 2026 are unknown until the Fed publishes the 2027 schedule
  (~mid-2026). days_to_fomc is NaN for any date after the last entry in
  FOMC_ANNOUNCEMENT_DATES; XGBoost handles NaN natively. Update the constant
  annually — see the URL below.

FOMC dates source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = _PROJ_ROOT / "data" / "calendar_features.parquet"

# Cover well before the model's 5-yr training window starts and a few years
# forward of present so daily prediction has lookahead headroom on opex /
# day-of-week features (FOMC will tail off into NaN past the last known date).
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2027-12-31"

# ---------------------------------------------------------------------------
# Scheduled FOMC ANNOUNCEMENT dates (day 2 of each 2-day meeting).
# Verify against https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# Excludes EMERGENCY meetings (e.g. 2020-03-03 and 2020-03-15 surprise cuts).
# Update this list each year when the Fed publishes the next year's calendar.
# ---------------------------------------------------------------------------
FOMC_ANNOUNCEMENT_DATES: tuple[str, ...] = (
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (scheduled meetings only)
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 — verify against the Fed's published 2026 calendar
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    # 2027 — INTENTIONALLY OMITTED. The Fed publishes the next year's calendar
    # around mid-summer of the prior year; add 2027 dates when they appear.
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Return the 3rd Friday of (year, month) — equity options monthly expiry."""
    first = pd.Timestamp(year=year, month=month, day=1)
    days_to_first_friday = (4 - first.weekday()) % 7   # Friday = weekday 4
    return first + pd.Timedelta(days=days_to_first_friday + 14)


def _monthly_opex_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """All monthly opex (3rd Friday) dates needed to cover [start, end].

    Extends one month past `end` so days_to_opex is well-defined for index
    rows that fall after the last 3rd Friday inside the range.
    """
    first_month = start.normalize().replace(day=1)
    last_month_plus_one = end.normalize().replace(day=1) + pd.offsets.MonthBegin(1)
    months = pd.date_range(first_month, last_month_plus_one, freq="MS")
    return pd.DatetimeIndex([_third_friday(d.year, d.month) for d in months])


def _days_to_next(target_dates: pd.DatetimeIndex, idx: pd.DatetimeIndex) -> np.ndarray:
    """
    For each date in `idx`, return the number of calendar days until the next
    date in `target_dates` that is >= the idx date. Returns NaN if no future
    target date exists in the list (i.e. idx date past the last known target).
    """
    sorted_targets = np.sort(target_dates.values.astype("datetime64[ns]"))
    idx_vals = idx.values.astype("datetime64[ns]")
    # side="left" → if idx_vals[i] equals a target, returns that target's pos,
    # giving days_to_next = 0 on the event day itself.
    pos = np.searchsorted(sorted_targets, idx_vals, side="left")
    out = np.full(len(idx), np.nan, dtype=np.float64)
    in_range = pos < len(sorted_targets)
    deltas = (
        sorted_targets[pos[in_range]] - idx_vals[in_range]
    ).astype("timedelta64[D]").astype(np.int64)
    out[in_range] = deltas
    return out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_calendar_features(
    start: str | pd.Timestamp = DEFAULT_START,
    end: str | pd.Timestamp = DEFAULT_END,
    fomc_dates: tuple[str, ...] = FOMC_ANNOUNCEMENT_DATES,
) -> pd.DataFrame:
    """Build the calendar feature matrix indexed by every Mon-Fri in [start, end]."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    idx = pd.bdate_range(start_ts, end_ts, name="date")
    df = pd.DataFrame(index=idx)

    # --- Day-of-week one-hot (only Mon..Fri exist in a bdate_range)
    dow = idx.dayofweek
    for i, name in enumerate(["mon", "tue", "wed", "thu", "fri"]):
        df[f"dow_{name}"] = (dow == i).astype(np.uint8)

    # --- FOMC features
    fomc_idx = pd.DatetimeIndex(pd.to_datetime(list(fomc_dates))).sort_values()
    df["days_to_fomc"] = _days_to_next(fomc_idx, idx)        # float64 with NaN tail
    df["is_fomc_day"] = idx.isin(fomc_idx).astype(np.uint8)

    # --- Options expiry features
    opex_idx = _monthly_opex_dates(start_ts, end_ts)
    days_to_opex = _days_to_next(opex_idx, idx)
    if np.isnan(days_to_opex).any():
        raise ValueError("days_to_opex has NaN — opex generation gap; check date range")
    df["days_to_opex"] = days_to_opex.astype(np.int32, copy=False)

    quarterly_opex = pd.DatetimeIndex([d for d in opex_idx if d.month in (3, 6, 9, 12)])
    df["is_quarterly_opex"] = idx.isin(quarterly_opex).astype(np.uint8)

    return df


