from pathlib import Path

import requests
import pandas as pd

def fetch_daily_data(api_url: str) -> dict:
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

def process_data(data: dict | list | str | Path) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        return pd.read_json(data)
    return pd.DataFrame(data)

def clean_news_data(df: pd.DataFrame) -> pd.DataFrame:
    return df

test = process_data("data/test_data.json")
print(test.head())