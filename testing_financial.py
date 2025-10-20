import unittest
import pandas as pd

from models import FinancialData

class MyTestCase(unittest.TestCase):
    def test_financial_data(self):
        path = "date_istorice.csv"
        column = "Datorii"

        print(f"------ Test FinancialData class on {path} -----")

        data_manager = FinancialData(path)
        initial_df = data_manager._load_data

        assert not initial_df, "Failed test: DataFrame wasn't loaded"
        print("Test 1(_load_data): Data loaded successful")

        assert isinstance(initial_df.index, pd.DatetimeIndex), "test failed: Index is not of type DateTimeIndex"

    if __name__ == "__main__":
            test_financial_data()