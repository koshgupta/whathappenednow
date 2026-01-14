from pathlib import Path
import json

import requests
import pandas as pd

def fetch_daily_data(api_url: str) -> dict:
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

def process_data(data: dict | list | str | Path) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        data_path = Path(data)
        with data_path.open() as data_file:
            data = json.load(data_file)
    if isinstance(data, dict) and "feed" in data:
        new_data = pd.json_normalize(
            data,
            record_path="feed",
            meta=[
                "items",
                "sentiment_score_definition",
                "relevance_score_definition",
            ],
        )
        return new_data.drop(columns=["url", "authors", "banner_image","source","category_within_source","source_domain"])
    return pd.json_normalize(data)