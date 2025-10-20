import pytest
import pandas as pd
from datetime import datetime, timedelta

@pytest.fixture(scope="session")
def raw_data() -> pd.DataFrame:
    dates = [datetime.now() - timedelta(days=i) for i in range(50, 0, -1)]
    data = {
        'Open': [100 + i for i in range(50)],
        'High': [102 + i for i in range(50)],
        'Low': [98 + i for i in range(50)],
        'Close':[101 + i * 1.1 for i in range(50)],
        'Volume': [1000 + i * 1000 for i in range(50)]
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

@pytest.fixture(scope="session")
def raw_data_with_nans(raw_ohlcv_data) -> pd.DataFrame:
    df = raw_ohlcv_data.copy()
    df.loc[df.index[5], 'Close'] = None
    df.loc[df.index[10], 'Close'] = None
    return df