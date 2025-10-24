import pandas as pd
import numpy as np
from typing import Tuple

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean_data(self) -> pd.DataFrame:
        df_cleaned = self.df.ffill()
        return df_cleaned.dropna()

    @staticmethod
    def create_sequences(df: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(df) - window_size):
            sequence = df.iloc[i:i + window_size].values
            X.append(sequence)
            target = df['close'].iloc[i + window_size]
            y.append(target)
        return np.array(X), np.array(y)
