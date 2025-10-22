import os
import sys
import pandas as pd
import numpy as np # Import numpy
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data.connector import MarketAPIConnector
from data.extractor import get_raw_data
from data.processor import clean_data, create_sequences

# Load environment variables from .env file
load_dotenv()

def run_data_pipeline(ticker: str, interval: str, window_size: int) -> (np.ndarray, np.ndarray, pd.DataFrame):
    """
    Orchestrates the data processing pipeline.

    1. Fetches raw data using the MarketAPIConnector.
    2. Cleans the raw data.
    3. Creates sequences for machine learning.

    Args:
        ticker: The stock ticker symbol (e.g., "AAPL").
        interval: The time interval for the data (e.g., "60min").
        window_size: The size of the sliding window for creating sequences.

    Returns:
        A tuple containing the feature sequences (X), target sequences (y), and the cleaned DataFrame.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY":
        raise ValueError("API key not found. Please set the ALPHA_VANTAGE_API_KEY in your .env file.")

    connector = MarketAPIConnector(api_key)

    # 1. Extract
    print(f"Fetching raw data for {ticker}...")
    raw_data = get_raw_data(connector, ticker, interval)
    print("Raw data fetched successfully.")

    # 2. Process
    print("Cleaning data...")
    cleaned_data = clean_data(raw_data)
    print("Data cleaned successfully.")

    # 3. Create Sequences
    print("Creating sequences...")
    X, y = create_sequences(cleaned_data, window_size)
    print("Sequences created successfully.")

    return X, y, cleaned_data

if __name__ == '__main__':
    # Example usage of the data pipeline service
    TICKER = "IBM"
    INTERVAL = "60min"
    WINDOW_SIZE = 10

    try:
        features, targets, cleaned_df = run_data_pipeline(TICKER, INTERVAL, WINDOW_SIZE)
        print(f"\nPipeline finished successfully!")
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Cleaned data head:\n{cleaned_df.head()}")
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}")
