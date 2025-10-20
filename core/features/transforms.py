import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs)).rename('RSI')

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:

    df['RSI'] = calculate_rsi(df)
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    return df.dropna()

def scale_features(df: pd.DataFrame, scaler: MinMaxScaler = None) -> Tuple[pd.DataFrame, MinMaxScaler]:

    if scaler is None:
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        return df_scaled, scaler

    else:
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
        return df_scaled, scaler

