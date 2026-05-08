import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta


load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Grab news data into dataframe

def fetch_sp500_news_history(
    api_key: str, 
    years: int = 5, 
    save_path: str = os.path.join(_PROJ_ROOT, "data", "sp500_news.parquet")
):
    """
    Fetches historical news for S&P 500 modeling.
    Expands scope to include healthcare, general macro/politics, and energy-producing nations.
    Saves directly to your ZFS pool in Parquet format.
    """
    
    # Use the historical endpoint (Requires Mediastack Paid Plan)
    BASE_URL = "http://api.mediastack.com/v1/news"
    
    # S&P 500 Country Anchors:
    # us (US) - Primary market
    # cn (China) - Manufacturing & supply chain
    # sa (Saudi Arabia), ca (Canada) - Energy & Commodities
    COUNTRY_LIST = 'us,cn,sa,ca'

    # S&P 500 Category Set:
    # business, technology (core sector signal)
    # health (captures the ~13% S&P 500 Healthcare weighting)
    CATEGORY_LIST = 'business,technology,health'

    end_date = datetime.now()
    start_date = end_date - relativedelta(years=years)

    # Query day-by-day. Mediastack caps results-per-query at 10,000; with sort=
    # published_desc, larger chunks return only the most-recent N articles, which
    # collapses to a few days per chunk. Daily queries sidestep that.
    date_ranges = pd.date_range(start=start_date, end=end_date, freq='D')

    # Pagination cap to fit the 10k/month API quota. 5 pages × 100 = 500 articles/day
    # ceiling. Across 1,825 days this is ~9,125 calls in the worst case (every day
    # exceeds the cap). sort=published_desc keeps the latest articles when truncated.
    MAX_PAGES_PER_DAY = 5
    LIMIT = 100

    all_articles = []

    print(f"📡 Initializing fetch for S&P 500 broad macro signal...")
    print(f"Targeting: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({len(date_ranges)} days)")

    for current_date in date_ranges:
        date_str = current_date.strftime('%Y-%m-%d')
        day_count = 0

        for page in range(MAX_PAGES_PER_DAY):
            params = {
                'access_key': api_key,
                'categories': CATEGORY_LIST,
                'countries': COUNTRY_LIST,
                'languages': 'en',
                'date': date_str,
                'limit': LIMIT,
                'offset': page * LIMIT,
                'sort': 'published_desc'
            }

            try:
                response = requests.get(BASE_URL, params=params, timeout=15)

                if response.status_code == 429:
                    print("⚠️ Rate limit reached. Waiting 10 seconds...")
                    time.sleep(10)
                    response = requests.get(BASE_URL, params=params, timeout=15)

                if response.status_code != 200:
                    print(f"❌ API Error {response.status_code} on {date_str}: {response.text}")
                    break

                data = response.json()
                articles = data.get('data', [])

                if not articles:
                    break

                all_articles.extend(articles)
                day_count += len(articles)

                pagination = data.get('pagination', {})
                total_found = pagination.get('total', 0)
                if (page + 1) * LIMIT >= total_found:
                    break

                time.sleep(0.15)

            except requests.exceptions.RequestException as e:
                print(f"⚠️ Connection error on {date_str}: {e}. Skipping rest of day.")
                break

        print(f"✅ {date_str}: {day_count} articles | Total: {len(all_articles)}")

    # Data Processing & Storage
    if all_articles:
        df = pd.DataFrame(all_articles)
        
        # 1. Deduplicate by title and date
        df = df.drop_duplicates(subset=['title', 'published_at'])
        
        # 2. Enforce Datetime types for XGBoost Sliding Window
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        # 3. Sort by date (oldest first)
        df = df.sort_values('published_at').reset_index(drop=True)
        
        # 4. Save to your ZFS dataset
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df.to_parquet(save_path, compression='snappy', index=False)
        print(f"\n🚀 SUCCESS: {len(df)} articles stored at {save_path}")
        return df
    else:
        print("🛑 No data retrieved. Verify your API key and plan tier.")
        return None
        
def align_and_prepare_news(
    input_parquet_path: str, 
    output_parquet_path: str, 
    date_col: str = 'published_at',
    title_col: str = 'title',
    desc_col: str = 'description'
):
    """
    Reads raw news data, time-boxes it to the 4:00 PM - 8:59 AM window, 
    adjusts for weekends, combines text fields, and saves a ready-to-score Parquet file.
    """
    print(f"Reading raw data from {input_parquet_path}...")
    
    # 1. Read the raw parquet file
    df = pd.read_parquet(input_parquet_path)
    initial_rows = len(df)
    
    # 2. Combine Title and Description for Hugging Face
    # We use .fillna('') to ensure missing descriptions don't wipe out the titles
    # We also add a period and space between them for natural sentence flow
    df['news_text'] = (
        df[title_col].fillna('').str.strip() + 
        ". " + 
        df[desc_col].fillna('').str.strip()
    )
    
    # Clean up any weird double periods (e.g., if the title already ended in a period)
    df['news_text'] = df['news_text'].str.replace('.. ', '. ', regex=False)
    
    # Optional: Drop the old columns if you want to save memory/storage
    # df = df.drop(columns=[title_col, desc_col])
    
    # 3. Convert timestamps to US/Eastern
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert('US/Eastern')
    
    # 4. Extract the base date and the hour for vectorized logic
    df['base_date'] = df[date_col].dt.normalize()
    df['hour'] = df[date_col].dt.hour
    
    # 5. Apply custom time-boxing logic
    conditions = [
        (df['hour'] < 9),       # Midnight to 8:59 AM -> Belongs to TODAY
        (df['hour'] >= 16)      # 4:00 PM to Midnight -> Belongs to TOMORROW
    ]
    
    choices = [
        df['base_date'],
        df['base_date'] + pd.Timedelta(days=1)
    ]
    
    # Anything between 9:00 AM and 3:59 PM is assigned NaT
    df['Trading_Date'] = np.select(conditions, choices, default=pd.NaT)
    
    # 6. Drop the intraday news
    df = df.dropna(subset=['Trading_Date'])
    filtered_rows = len(df)
    
    # 7. The Weekend Trap Fix
    day_of_week = df['Trading_Date'].dt.dayofweek
    shift_days = np.where(day_of_week == 5, 2, np.where(day_of_week == 6, 1, 0))
    df['Trading_Date'] = df['Trading_Date'] + pd.to_timedelta(shift_days, unit='D')

    # 8. Strip timezone and normalize to tz-naive midnight.
    # .normalize() is applied first to guard against DST-induced hour drift from the
    # timedelta shift above (e.g. a Saturday midnight ET + 2 days over a DST boundary
    # could land at 01:00 AM instead of midnight before stripping).
    # The result matches market_data.py's tz-naive date index exactly.
    df['Trading_Date'] = df['Trading_Date'].dt.normalize().dt.tz_localize(None)

    # 9. Slim to only the columns needed for HuggingFace sentiment scoring.
    # Date      — trading day label; used as groupby key during aggregation and
    #             as the join key when merging sentiment features with market features.
    # date_col  — original publish timestamp; preserved for intra-day ordering so
    #             articles within a trading day are processed chronologically.
    # news_text — combined title + description fed to FinBERT / Llama3-FinSenti.
    df = (
        df[['Trading_Date', date_col, 'news_text']]
        .rename(columns={'Trading_Date': 'Date'})
        .sort_values(by=['Date', date_col])
        .reset_index(drop=True)
    )

    # 10. Save
    df.to_parquet(output_parquet_path, index=False)

    print(f"Combined titles and descriptions into 'news_text' column.")
    print(f"Dropped {initial_rows - filtered_rows} out-of-window articles.")
    print(f"Saved {len(df)} aligned articles to {output_parquet_path}.")

    return df

if __name__ == "__main__":    
    # --- Usage Example ---
    df_ready_for_nlp = align_and_prepare_news(
        input_parquet_path="/mnt/storage/data/code/whathappenednow/data/sp500_news.parquet",
        output_parquet_path="/mnt/storage/data/code/whathappenednow/data/aligned_premarket_news.parquet",
        date_col="published_at" # Change this if your Mediastack column is named differently
    )
    # fetch_sp500_news_history(
    #     api_key=NEWS_API_KEY,
    #     years=5,
    #     save_path=os.path.join(_PROJ_ROOT, "data", "sp500_news.parquet"))
