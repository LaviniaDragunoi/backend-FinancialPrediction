import pandas as pd
from .connector import MarketAPIConnector
from typing import Dict, Any

def get_raw_data(connector: MarketAPIConnector, ticker: str, interval: str) -> pd.DataFrame:
    raw_dict: Dict[str, Any] = connector.fetch_data(ticker, interval)
    time_series_key = next((key for key in raw_dict.keys() if "Time Series" in key), None)
    if not time_series_key or not raw_dict.get(time_series_key):
        raise ValueError("Could not find time series data in the API response.")

    df = pd.DataFrame.from_dict(raw_dict[time_series_key], orient="index")

    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    })

    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.sort_index(inplace=True)
    return df
