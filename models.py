import pandas as pd

class FinancialData:
    def __init__(self, file_path):
        self._file_path = file_path
        self._initial_data = self._load_data(file_path)

    def _load_data(self, file_path):
        try:
            df = pd.read_csv(
                file_path,
                sep=',',
                parse_dates=True,
                )
            return df
        except FileNotFoundError:
            print(f"Error: '{file_path}' was not found.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error while loading: {e}")
            return pd.DataFrame()

    def cleanup_data(self, new_entry_df: pd.DataFrame) -> pd.DataFrame:
        clean_df = new_entry_df.copy()
        column_date = clean_df.columns[0]

        try:
            clean_df[column_date] = pd.to_datetime(clean_df[column_date])

            clean_df = clean_df.set_index(column_date)

        except Exception as e:
            print(f"Time convertion/set index error: {e}")
            return pd.DataFrame()

        clean_df = clean_df.fillna(clean_df.mean())
        return clean_df

    def extract_date_series(self, clean_df: pd.DataFrame, column: str) -> pd.Series:
        try:
            date_series = clean_df[column].copy()

            if not date_series.index.is_monotonic_increasing:
                date_series = date_series.sort_index()
                print("Warnning: Index was sorted for cronological order")
            return date_series

        except KeyError:
            print(f"Column '{column} was noty found. Error")
            return pd.Series(dtype=float)
        except Exception as e:
            print(f"Error while extracting date series: {e}")
            return pd.Series(dtype=float)