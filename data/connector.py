import requests
import pandas as pd
from typing import Dict, Any

class AlphaVantageConnector:
    BASE_URL = "https://www.alphavantage.co"

    def __init__(self, api_key: str, use_local_data: bool = False, local_data_path: str = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.use_local_data = use_local_data
        self.local_data_path = local_data_path

    def fetch_time_series_intraday(self, ticker: str, interval: str) -> Dict[str, Any]:
        if self.use_local_data and self.local_data_path:
            print(f"--- Loading local data from: {self.local_data_path} ---")
            df = pd.read_csv(self.local_data_path, index_col="timestamp")
            # Convert the DataFrame to the dictionary format expected by the extractor
            time_series_data = df.to_dict(orient="index")
            return {"Time Series (60min)": time_series_data} # Mock the API response structure

        params = {
            'function': "TIME_SERIES_INTRADAY",
            'symbol': ticker,
            'interval': interval,
            'apikey': self.api_key
        }
        return self._make_request('query', params)

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error for {url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Exception for {url}: {e}")
            raise

    def close(self):
        self.session.close()
