import requests
from typing import Dict, Any, Optional

class MarketAPIConnector:
    def __init__(self, api_key: str, base_url:str):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()

    def _make_request(self, endpoint, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
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
        endpoint = f"ticker/{ticker}/{interval}"
        params = {
            'function': function,
            'symbol': ticker,
            'interval': interval
        }

        raw_data = self._make_request(endpoint, params)

        return raw_data