import pandas as pd
import numpy as np
from typing import Tuple


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values and drops any remaining NaNs."""
    # Forward-fill handles missing values in the middle of the data
    df_cleaned = df.fillna(method='ffill')
    # Drop any NaNs that might exist at the very beginning of the dataset
    return df_cleaned.dropna()


def create_sequences(df: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sequences of data for model training."""
    X, y = [], []
    for i in range(len(df) - window_size):
        # Extract a window of features
        sequence = df.iloc[i:i + window_size].values
        X.append(sequence)
        
        # The target is the 'close' price right after the window
        target = df['close'].iloc[i + window_size]
        y.append(target)

    return np.array(X), np.array(y)
