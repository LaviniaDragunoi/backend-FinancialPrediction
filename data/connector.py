import requests
from typing import Dict, Any, Optional

class MarketAPIConnector:
    BASE_URL = "https://www.alphavantage.co"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        query_params = params.copy() if params else {}
        query_params['apikey'] = self.api_key

        try:
            response = self.session.get(url, params=query_params, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error for {url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Exception for {url}: {e}")
            raise

    def fetch_data(self, ticker: str, interval: str, function: str = "TIME_SERIES_INTRADAY") -> Dict[str, Any]:
        params = {
            'function': function,
            'symbol': ticker,
            'interval': interval
        }

        raw_data = self._make_request('query', params)

        return raw_data
