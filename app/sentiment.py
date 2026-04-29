import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
API_KEY = os.getenv('HF_TOKEN')

def probabilities(HF_TOKEN: str):
    client = InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN,
    )
    df = pd.read_parquet("../data/sp500_news.parquet")
    for 
    result = client.text_classification(
        ,
        model="ProsusAI/finbert",
    )