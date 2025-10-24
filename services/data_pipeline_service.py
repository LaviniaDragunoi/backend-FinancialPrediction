import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import Tuple

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data.connector import AlphaVantageConnector
from data.extractor import DataExtractor
from data.processor import DataProcessor

# Load environment variables from .env file
load_dotenv()

class DataPipelineService:
    def __init__(self, api_key: str, use_local_data: bool = False, local_data_path: str = None):
        if not use_local_data and (not api_key or api_key == "YOUR_API_KEY"):
            raise ValueError("API key not found. Please set the ALPHA_VANTAGE_API_KEY in your .env file.")
        self.connector = AlphaVantageConnector(api_key, use_local_data, local_data_path)

    def run(self, ticker: str, interval: str, window_size: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        try:
            print(f"Fetching raw data for {ticker}...")
            raw_data_dict = self.connector.fetch_time_series_intraday(ticker, interval)
            extractor = DataExtractor(raw_data_dict)
            raw_df = extractor.extract_time_series_to_dataframe()
            print("Raw data fetched and extracted successfully.")

            print("Cleaning and processing data...")
            processor = DataProcessor(raw_df)
            cleaned_df = processor.clean_data()
            print("Data cleaned successfully.")

            print("Creating sequences...")
            X, y = DataProcessor.create_sequences(cleaned_df, window_size)
            print("Sequences created successfully.")

            return X, y, cleaned_df
        finally:
            self.connector.close()

if __name__ == '__main__':
    TICKER = "IBM"
    INTERVAL = "60min"
    WINDOW_SIZE = 10
    USE_LOCAL_DATA = True # Set to True to use the sample CSV
    LOCAL_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv')

    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        pipeline = DataPipelineService(api_key, use_local_data=USE_LOCAL_DATA, local_data_path=LOCAL_DATA_PATH)
        features, targets, cleaned_df = pipeline.run(TICKER, INTERVAL, WINDOW_SIZE)

        print(f"\nPipeline finished successfully!")
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Cleaned data head:\n{cleaned_df.head()}")
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}")
