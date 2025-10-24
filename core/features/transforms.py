import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI) for a given DataFrame."""
    close_prices = df['Close']
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).rename('RSI')

def add_technical_features(df: pd.DataFrame, ma_window: int = 20) -> pd.DataFrame:
    """
    Adds technical features to the DataFrame.
    
    Args:
        df: The input DataFrame.
        ma_window: The window for the moving average.
        
    Returns:
        A new DataFrame with the added features.
    """
    df = df.copy()
    df['RSI'] = calculate_rsi(df)
    df['MA'] = df['Close'].rolling(window=ma_window).mean()
    return df.dropna()

def scale_features(df: pd.DataFrame, scaler: MinMaxScaler = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scales the features of the DataFrame.
    
    Args:
        df: The input DataFrame.
        scaler: The scaler to use. If None, a new scaler is fitted.
        
    Returns:
        A tuple containing the scaled DataFrame and the scaler.
    """
    if scaler is None:
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    else:
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    return df_scaled, scaler
