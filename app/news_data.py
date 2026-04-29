import os
from dotenv import load_dotenv
import pandas as pd
import requests
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta


load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Grab news data into dataframe

def fetch_sp500_news_history(
    api_key: str, 
    years: int = 5, 
    save_path: str = "../data/sp500_news.parquet"
):
    """
    Fetches historical news for S&P 500 modeling.
    Expands scope to include healthcare, general macro/politics, and energy-producing nations.
    Saves directly to your ZFS pool in Parquet format.
    """
    
    # Use the historical endpoint (Requires Mediastack Paid Plan)
    BASE_URL = "http://api.mediastack.com/v1/news"
    
    # S&P 500 Country Anchors: 
    # us (US), gb (UK), de (Germany), cn (China), jp (Japan) - Core Macro
    # sa (Saudi Arabia), ca (Canada) - Energy & Commodities
    COUNTRY_LIST = 'us,gb,de,cn,jp,sa,ca'
    
    # S&P 500 Category Expansion:
    # business, technology (Core)
    # general (Fed policy, geopolitical events)
    # health (Captures the ~13% S&P 500 Healthcare weighting)
    CATEGORY_LIST = 'business,technology,general,health'
    
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=years)
    
    # Create monthly chunks to avoid hitting the 10,000 pagination limit per request
    date_ranges = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    all_articles = []
    
    print(f"📡 Initializing fetch for S&P 500 broad macro signal...")
    print(f"Targeting: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    for i in range(len(date_ranges) - 1):
        chunk_start = date_ranges[i].strftime('%Y-%m-%d')
        chunk_end = date_ranges[i+1].strftime('%Y-%m-%d')
        
        offset = 0
        limit = 100 # Maximum allowed per request
        
        while True:
            params = {
                'access_key': api_key,
                'categories': CATEGORY_LIST,
                'countries': COUNTRY_LIST,
                'languages': 'en',
                'date': f"{chunk_start},{chunk_end}",
                'limit': limit,
                'offset': offset,
                'sort': 'published_desc'
            }
            
            try:
                response = requests.get(BASE_URL, params=params, timeout=15)
                
                if response.status_code == 429:
                    print("⚠️ Rate limit reached. Waiting 10 seconds...")
                    time.sleep(10)
                    continue
                    
                if response.status_code != 200:
                    print(f"❌ API Error {response.status_code}: {response.text}")
                    break
                    
                data = response.json()
                articles = data.get('data', [])
                
                if not articles:
                    break # End of this month's data
                    
                all_articles.extend(articles)
                
                # Check pagination progress
                pagination = data.get('pagination', {})
                total_found = pagination.get('total', 0)
                offset += limit
                
                if offset >= total_found:
                    break
                    
                # Small sleep to prevent TCP congestion
                time.sleep(0.15)
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Connection error: {e}. Retrying...")
                time.sleep(5)

        print(f"✅ Chunk Completed: {chunk_start} | Total so far: {len(all_articles)}")

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
        


# To run:
df = fetch_sp500_news_history(api_key=NEWS_API_KEY)

# Feed it to classifier
# Get the previous days financial data
# Make a dataframe with the news and the financial data
# Predict the next day movement of the stock price
# Save predictions to the database
# Save model state 
