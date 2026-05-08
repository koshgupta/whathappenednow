"""
ES continuous futures feature pipeline for S&P 500 intraday direction prediction.

Prediction point : 9:00 AM ET — after 8:30 AM macro noise settles, before cash open.
Target           : buffered binary — 1 if RTH close (4 PM) > RTH open (9:30 AM) * 1.0002.
Anti-lookahead   : daily technical indicators are shifted forward by 1 trading day so
                   each 9 AM row only sees values knowable from yesterday's close.
                   Target uses the exact 9:30 AM open (09:30 bar) and 4 PM close
                   (15:59 bar) — both only available after the fact, used for training only.
"""
import os
from datetime import datetime, time, timedelta

import databento as db
import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401  (registers .ta accessor)
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

load_dotenv()

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET = "GLBX.MDP3"
SYMBOL = "ES.c.0"
SCHEMA = "ohlcv-1m"
LOOKBACK_YEARS = 5

RAW_PATH = os.path.join(_PROJ_ROOT, "data", "es_intraday_raw.parquet")
FEATURES_PATH = os.path.join(_PROJ_ROOT, "data", "features_9am_snapshot.parquet")

# Databento stores prices as fixed-point int64: actual_price = int_value / PRICE_FACTOR
PRICE_FACTOR = 1e9


# ---------------------------------------------------------------------------
# 1. DATA FETCH
# ---------------------------------------------------------------------------

def fetch_es_futures() -> pd.DataFrame:
    """Fetch 5 years of ES continuous futures 1 min bars from Databento and cache locally."""
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise EnvironmentError("DATABENTO_API_KEY not set in environment")

    end = datetime.now() - timedelta(days=1)
    start = end - relativedelta(years=LOOKBACK_YEARS + 1)

    print(f"Fetching {SYMBOL} ({SCHEMA}) from {start.date()} to {end.date()} via Databento (6 yr for SMA-200 warm-up, ~3M bars expected)...")

    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[SYMBOL],
        schema=SCHEMA,
        start=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end=end.strftime("%Y-%m-%dT%H:%M:%S"),
        stype_in="continuous",
    )

    df = data.to_df()
    df.index = pd.to_datetime(df.index, utc=True)

    # Databento returns prices as int64 fixed-point; convert to float if needed
    for col in ("open", "high", "low", "close"):
        if col in df.columns and df[col].dtype == np.int64:
            df[col] = df[col] / PRICE_FACTOR

    df = df[["open", "high", "low", "close", "volume"]].sort_index()

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_parquet(RAW_PATH, engine="pyarrow")
    print(f"  {len(df):,} bars saved → {RAW_PATH}")

    return df


# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------

def _to_et(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a US/Eastern DatetimeIndex."""
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    return df


def _daily_resample(df_et: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1 min ET bars to daily OHLCV over the RTH window (09:30–15:59).
    The 15:59 bar is left-labeled and closes at 16:00 ET — the official 4 PM close.
    """
    rth = df_et.between_time("09:30", "15:59")
    daily = (
        rth.groupby(rth.index.date)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily[(daily.index.dayofweek < 5) & (daily["close"] > 0)]
    return daily


# ---------------------------------------------------------------------------
# 3. FEATURE BUILDERS
# ---------------------------------------------------------------------------

def build_daily_technicals(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily technical indicators, convert all price-based values to
    stationary percentage distances ((indicator / price) − 1), then shift
    the entire frame forward by 1 trading day to prevent lookahead.

    Indicators computed:
        Scale-free (no transformation): RSI-14, ADX-14, +DI-14, -DI-14,
                                        Stochastic %K/%D, relative volume
        Price-based → (x / close) − 1: SMA-50, SMA-200, ATR-14, BB %B,
                                        MACD line, histogram, signal
        Normalized ratios (already stationary): prev_rth_momentum,
                                                prev_day_range
    """
    print("Computing daily technical indicators...")

    df_et = _to_et(df_1m)
    daily = _daily_resample(df_et)
    close = daily["close"]

    # SMA distances: (price / SMA) − 1
    daily.ta.sma(length=50, append=True)
    daily.ta.sma(length=200, append=True)
    if "SMA_50" in daily.columns:
        daily["dist_sma_50"] = (close / daily["SMA_50"]) - 1
    if "SMA_200" in daily.columns:
        daily["dist_sma_200"] = (close / daily["SMA_200"]) - 1

    # RSI-14: already bounded 0–100, kept as-is
    daily.ta.rsi(length=14, append=True)
    if "RSI_14" in daily.columns:
        daily["rsi_14"] = daily["RSI_14"]

    # ATR as fraction of price: ATR / price
    # pandas_ta column name varies by version: prefer ATRr_14, fall back to ATR_14
    daily.ta.atr(length=14, append=True)
    atr_col = next((c for c in ("ATRr_14", "ATR_14") if c in daily.columns), None)
    if atr_col:
        daily["pct_atr"] = daily[atr_col] / close
    else:
        print("  WARNING: ATR column not found — pct_atr feature will be missing")

    # Bollinger Bands %B: (price − lower) / (upper − lower)
    daily.ta.bbands(length=20, append=True)
    bbl = next((c for c in daily.columns if c.startswith("BBL_")), None)
    bbu = next((c for c in daily.columns if c.startswith("BBU_")), None)
    if bbl and bbu:
        bb_range = daily[bbu] - daily[bbl]
        daily["bb_pct_b"] = np.where(
            bb_range == 0, 0.5, (close - daily[bbl]) / bb_range
        )

    # Relative volume: today's volume / 20-day rolling mean
    daily["relative_volume"] = daily["volume"] / daily["volume"].rolling(20).mean()

    # ADX-14: trend strength and directional indicators (all 0–100, scale-free)
    daily.ta.adx(length=14, append=True)
    adx_col = next((c for c in daily.columns if c.startswith("ADX_")), None)
    dmp_col = next((c for c in daily.columns if c.startswith("DMP_")), None)
    dmn_col = next((c for c in daily.columns if c.startswith("DMN_")), None)
    if adx_col:
        daily["adx_14"] = daily[adx_col]
    else:
        print("  WARNING: ADX column not found — adx_14 feature will be missing")
    if dmp_col:
        daily["dmp_14"] = daily[dmp_col]
    if dmn_col:
        daily["dmn_14"] = daily[dmn_col]

    # Stochastic Oscillator: %K and %D (both 0–100, scale-free)
    daily.ta.stoch(append=True)
    stoch_k_col = next((c for c in daily.columns if c.startswith("STOCHk_")), None)
    stoch_d_col = next((c for c in daily.columns if c.startswith("STOCHd_")), None)
    if stoch_k_col:
        daily["stoch_k"] = daily[stoch_k_col]
    else:
        print("  WARNING: STOCHk column not found — stoch_k feature will be missing")
    if stoch_d_col:
        daily["stoch_d"] = daily[stoch_d_col]

    # MACD: price-denominated → divide by close for stationarity
    daily.ta.macd(append=True)
    macd_line = next((c for c in daily.columns if c.startswith("MACD_")), None)
    macd_hist = next((c for c in daily.columns if c.startswith("MACDh_")), None)
    macd_signal = next((c for c in daily.columns if c.startswith("MACDs_")), None)
    if macd_line:
        daily["pct_macd"] = daily[macd_line] / close
    else:
        print("  WARNING: MACD column not found — pct_macd feature will be missing")
    if macd_hist:
        daily["pct_macd_hist"] = daily[macd_hist] / close
    if macd_signal:
        daily["pct_macd_signal"] = daily[macd_signal] / close

    # Previous day RTH momentum: (RTH close − RTH open) / RTH open
    daily["prev_rth_momentum"] = (daily["close"] - daily["open"]) / daily["open"]

    # Previous day total price swing: (high − low) / open — normalized range
    daily["prev_day_range"] = (daily["high"] - daily["low"]) / daily["open"]

    feature_cols = [
        "dist_sma_50", "dist_sma_200", "rsi_14",
        "pct_atr", "bb_pct_b", "relative_volume",
        "adx_14", "dmp_14", "dmn_14",
        "stoch_k", "stoch_d",
        "pct_macd", "pct_macd_hist", "pct_macd_signal",
        "prev_rth_momentum", "prev_day_range",
    ]
    out = daily[[c for c in feature_cols if c in daily.columns]].copy()

    # ANTI-LOOKAHEAD: each row now holds the indicator values from the prior trading day's close
    out = out.shift(1)

    return out


def build_premarket_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-market features locked at 9:00 AM ET (all knowable before cash open):

    overnight_gap    = (9 AM open / prev-day 4 PM close) − 1
    premarket_trend  = (9 AM open / prev-evening 6 PM futures open) − 1
    macro_vol_830    = (8:30 bar high − low) / open  — magnitude of macro data reaction
    macro_dir_830    = (8:30 bar close − open) / open — direction of macro data reaction

    The 8:30 AM bar covers 8:30–8:59 AM and is fully complete at 9:00 AM, capturing
    the initial market reaction to macro releases (NFP, CPI, PPI, retail sales, etc.).
    The 6 PM bar on date T is re-indexed to T+1 before joining (Sunday→Monday safe).
    """
    print("Computing pre-market features...")

    df_et = _to_et(df_1m)

    # 9:00 AM open — the prediction snapshot price (bar opens at 9:00 AM, knowable at 9:00 AM)
    s_9am = df_et[df_et.index.time == time(9, 0)]["open"].copy()
    s_9am.index = pd.to_datetime(s_9am.index.date)
    s_9am.name = "price_9am"

    # 4:00 PM close — close of the 15:59 bar (left-labeled, ends at 16:00 ET)
    s_4pm = df_et[df_et.index.time == time(15, 59)]["close"].copy()
    s_4pm.index = pd.to_datetime(s_4pm.index.date)
    s_4pm.name = "close_4pm"

    # 6:00 PM overnight open — open of the 18:00 bar, mapped one calendar day forward
    # so Sunday's 6 PM bar aligns with Monday's 9 AM row, Monday's with Tuesday's, etc.
    s_6pm = df_et[df_et.index.time == time(18, 0)]["open"].copy()
    s_6pm.index = pd.to_datetime(s_6pm.index.date) + pd.Timedelta(days=1)
    s_6pm.name = "overnight_open_6pm"

    # 8:30–8:59 AM window — aggregated across all 30 one-minute bars to capture the full
    # macro data release reaction window (a single time(8,30) bar with 1m resolution covers
    # only 60 seconds; we need the full 8:30–9:00 spread for macro_vol/dir_830)
    _s830_range = df_et.between_time("08:30", "08:59")
    s_830 = (
        _s830_range.groupby(_s830_range.index.date)
        .agg(
            open_830=("open", "first"),
            high_830=("high", "max"),
            low_830=("low", "min"),
            close_830=("close", "last"),
        )
    )
    s_830.index = pd.to_datetime(s_830.index)

    df = (
        s_9am.to_frame()
        .join(s_4pm, how="left")
        .join(s_6pm, how="left")
        .join(s_830, how="left")
    )

    # Shift 4 PM close by 1 row to get the *previous* trading day's close
    df["prev_close_4pm"] = df["close_4pm"].shift(1)

    df["overnight_gap"] = (df["price_9am"] / df["prev_close_4pm"]) - 1
    df["premarket_trend"] = (df["price_9am"] / df["overnight_open_6pm"]) - 1

    # 8:30 AM volatility: normalized range — how big was the macro reaction
    df["macro_vol_830"] = (df["high_830"] - df["low_830"]) / df["open_830"]

    # 8:30 AM direction: signed return — which way did the macro reaction go
    df["macro_dir_830"] = (df["close_830"] - df["open_830"]) / df["open_830"]

    return df[["overnight_gap", "premarket_trend", "macro_vol_830", "macro_dir_830"]]


def build_target(df_1m: pd.DataFrame) -> pd.Series:
    """
    Buffered binary target:
        y = 1  if  close_4pm > open_930am * 1.0002
        y = 0  otherwise  (flat or down, including small up moves ≤ 0.02 %)

    With 1 min bars both prices are exact discrete bar boundaries — no proxy needed.
    close_4pm  = close of the 15:59 bar (left-labeled, ends at 16:00 ET).
    open_930am = open of the 09:30 bar (left-labeled, the 9:30 AM cash open tick).
    Neither value is available at the 9:00 AM prediction point; both are only used
    to label historical rows during training.
    """
    print("Building buffered binary target...")

    df_et = _to_et(df_1m)

    # Left-labeled 15:59 bar → closes at 16:00 ET = official 4 PM RTH close
    close_4pm = df_et[df_et.index.time == time(15, 59)]["close"].copy()
    close_4pm.index = pd.to_datetime(close_4pm.index.date)

    # Left-labeled 09:30 bar → opens at exactly 9:30 AM ET = cash open tick
    open_930 = df_et[df_et.index.time == time(9, 30)]["open"].copy()
    open_930.index = pd.to_datetime(open_930.index.date)

    aligned = close_4pm.rename("close_4pm").to_frame().join(
        open_930.rename("open_930am"), how="inner"
    )

    y = pd.Series(
        np.where(aligned["close_4pm"] > aligned["open_930am"] * 1.0002, 1, 0),
        index=aligned.index,
        name="y",
        dtype=np.int8,
    )
    return y


# ---------------------------------------------------------------------------
# 4. MAIN PIPELINE
# ---------------------------------------------------------------------------

def build_feature_matrix(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-market features, shifted daily technicals, and target into a
    single Date-indexed DataFrame of 9:00 AM snapshots ready for XGBoost.
    No raw OHLC prices are included in the output.
    """
    print("\nBuilding feature matrix...")

    df_pm = build_premarket_features(df_1m)
    df_tech = build_daily_technicals(df_1m)
    y = build_target(df_1m)

    df = df_pm.join(df_tech, how="left").join(y, how="left")

    # Restrict to the 5-year window (indicator warm-up excluded by dropna below)
    cutoff = pd.Timestamp(datetime.now() - relativedelta(years=LOOKBACK_YEARS)).normalize()
    df = df[df.index >= cutoff]

    df.dropna(inplace=True)
    df.index.name = "date"

    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    df.to_parquet(FEATURES_PATH, engine="pyarrow")

    print(f"\n  {len(df):,} trading days | {df.shape[1]} columns → {FEATURES_PATH}")
    print(f"  Features : {[c for c in df.columns if c != 'y']}")
    print(f"  Target   : {df['y'].mean():.1%} up days (y=1)")

    return df


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if os.path.exists(RAW_PATH):
        print(f"Loading cached raw data from {RAW_PATH}...")
        df_1m = pd.read_parquet(RAW_PATH)
        if df_1m.index.tz is None:
            df_1m.index = df_1m.index.tz_localize("UTC")
    else:
        df_1m = fetch_es_futures()

    build_feature_matrix(df_1m)
