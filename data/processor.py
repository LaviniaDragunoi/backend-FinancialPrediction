import pandas as pd
from typing import Tuple


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.fillna(method='ffill')

    return df_cleaned.dropna()


def create_sequences(df: pd.DataFrame, window_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i: i + window_size].values)
        y.append(df['Close'].iloc[i + window_size])

    return pd.DataFrame(X.reshape(len(X), -1)), pd.DataFrame(y)