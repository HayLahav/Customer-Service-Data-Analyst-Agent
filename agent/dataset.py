import os
import pandas as pd
from functools import lru_cache

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


@lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    """Load the customer support dataset, using a local CSV cache after the first download."""
    if os.path.exists(_CACHE_PATH):
        return pd.read_csv(_CACHE_PATH)

    print("Downloading dataset from HuggingFace (first run only, ~10 seconds)...")
    from datasets import load_dataset

    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    df = ds.to_pandas()
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    df.to_csv(_CACHE_PATH, index=False)
    print("Dataset saved to local cache.")
    return df
