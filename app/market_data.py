import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE')
TICKER = 'SPY' 
OUTPUT_FILE = 'sp500_5yr_indicators.parquet'

def fetch_and_calculate_sp500():
    print(f"Fetching full historical data for {TICKER}...")
    
    # 1. Fetch data from Alpha Vantage (1 Request = ~25 years of data)
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": TICKER,
        "outputsize": "full", 
        "apikey": API_KEY,
        "datatype": "json"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "Time Series (Daily)" not in data:
        print("Error fetching data. Check your API key or connection.")
        return
        
    # 2. Format into Pandas DataFrame
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
    df = df.astype(float)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = pd.to_datetime(df.index)
    df = df.sort_index() 
    
    # 3. Calculate Indicators on the ENTIRE history first
    # (We must do this before filtering so the 200-SMA has enough past data to calculate properly)
    print("Calculating technical indicators...")
    df.ta.sma(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=2, append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.bbands(length=20, append=True) 
    df.ta.atr(length=14, append=True)
    df.ta.vwma(length=20, append=True)
    df.ta.obv(append=True)
    
    # 4. Filter for exactly the last 5 years
    print("Filtering down to the last 5 years...")
    five_years_ago = datetime.now() - relativedelta(years=5)
    df_5yr = df[df.index >= five_years_ago].copy()
    
    # Clean up any potential NaNs and Save
    df_5yr.dropna(inplace=True)
    df_5yr.to_parquet(OUTPUT_FILE, engine='pyarrow')
    
    print(f"✅ Success! 5 years of data saved to {OUTPUT_FILE}")
    print(f"Total trading days captured: {len(df_5yr)}")
    
def preprocess_for_xgboost(input_file='sp500_5yr_indicators.parquet', output_file='sp500_xgboost_ready.parquet'):
    print(f"Loading raw data from {input_file}...")
    df = pd.read_parquet(input_file)
    
    # Ensure data is sorted chronologically (oldest to newest)
    df = df.sort_index()

    print("Engineering OHLC features...")
    # ---------------------------------------------------------
    # 1. OHLC Transformations (Stationary / Scale-Free)
    # ---------------------------------------------------------
    # Daily Return
    df['daily_return'] = (df['close'] / df['close'].shift(1)) - 1
    
    # Overnight Gap
    df['overnight_gap'] = (df['open'] / df['close'].shift(1)) - 1
    
    # Daily Range
    df['daily_range'] = (df['high'] - df['low']) / df['open']
    
    # Wicks (Using np.maximum/minimum is much faster than standard max/min on DataFrames)
    max_open_close = np.maximum(df['open'], df['close'])
    min_open_close = np.minimum(df['open'], df['close'])
    
    df['upper_wick'] = (df['high'] - max_open_close) / df['open']
    df['lower_wick'] = (min_open_close - df['low']) / df['open']

    print("Engineering Volume & Indicator features...")
    # ---------------------------------------------------------
    # 2. Volume Preprocessing
    # ---------------------------------------------------------
    # Relative Volume (Current volume vs 20-day average)
    df['relative_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # ---------------------------------------------------------
    # 3. Indicator Transformations (Price-bound to Percentage)
    # ---------------------------------------------------------
    # Moving Average Distance
    if 'SMA_200' in df.columns:
        df['dist_SMA_200'] = (df['close'] / df['SMA_200']) - 1
        
    # VWMA Distance (Functioning as your VWAP proxy)
    if 'VWMA_20' in df.columns:
        df['dist_VWMA_20'] = (df['close'] / df['VWMA_20']) - 1

    # Bollinger Bands %B (0 to 1 scale)
    if all(col in df.columns for col in ['BBL_20_2.0', 'BBU_20_2.0']):
        bb_range = df['BBU_20_2.0'] - df['BBL_20_2.0']
        # np.where prevents division by zero if bands ever pinch to exactly 0
        df['BB_pct_B'] = np.where(bb_range == 0, 0, (df['close'] - df['BBL_20_2.0']) / bb_range)

    # Average True Range (Percentage Volatility)
    if 'ATRr_14' in df.columns:
        df['pct_ATR_14'] = df['ATRr_14'] / df['close']
        
    # OBV Fix (OBV is cumulative and non-stationary. We must look at its daily change relative to volume)
    if 'OBV' in df.columns:
        df['OBV_momentum'] = df['OBV'].diff() / df['volume'].rolling(window=20).mean()

    # Special Case: MACD (If you end up adding it to your initial fetcher)
    macd_cols = [c for c in df.columns if c.startswith('MACD_')]
    for col in macd_cols:
        df[f'pct_{col}'] = df[col] / df['close']

    print("Setting up target variables and cleaning data...")
    # ---------------------------------------------------------
    # 4. Target Variable Creation
    # ---------------------------------------------------------
    # To train XGBoost for NEXT-DAY prediction, you need to shift the daily return back by 1 day.
    # This aligns today's indicators with tomorrow's actual return.
    df['target_next_day_return'] = df['daily_return'].shift(-1)
    
    # Optional: Create a binary classification target (1 if up, 0 if down)
    df['target_next_day_direction'] = np.where(df['target_next_day_return'] > 0, 1, 0)

    # ---------------------------------------------------------
    # 5. Clean Up (Drop Raw/Non-Stationary Columns)
    # ---------------------------------------------------------
    # We leave RSI_14, RSI_2, and MFI_14 alone since they are already stationary oscillators (0-100)
    cols_to_drop = [
        'open', 'high', 'low', 'close', 'volume', 
        'SMA_200', 'VWMA_20', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 
        'ATRr_14', 'OBV'
    ] + macd_cols

    # Drop the raw columns
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # Drop NaN rows created by the rolling averages and shifts
    df.dropna(inplace=True)

    # Save to Parquet
    df.to_parquet(output_file, engine='pyarrow')
    print(f"✅ Success! XGBoost-ready dataset saved to {output_file}")
    
    # Display the final features
    print("\nFinal Model Features:")
    print(df.columns.tolist())
    
    return df

if __name__ == "__main__":
    #fetch_and_calculate_sp500()
    # processed_df = preprocess_for_xgboost()