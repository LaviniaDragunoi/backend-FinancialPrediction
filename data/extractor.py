import pandas as pd
from typing import Dict, Any

class DataExtractor:
    def __init__(self, raw_data: Dict[str, Any]):
        self.raw_data = raw_data

    def extract_time_series_to_dataframe(self) -> pd.DataFrame:
        time_series_key = next((key for key in self.raw_data.keys() if "Time Series" in key), None)
        if not time_series_key or not self.raw_data.get(time_series_key):
            raise ValueError("Could not find time series data in the API response.")

        df = pd.DataFrame.from_dict(self.raw_data[time_series_key], orient="index")

        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
