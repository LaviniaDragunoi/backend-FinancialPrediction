from data.processor import clean_data, create_sequences


def test_data_cleaning(raw_data_with_nans):
    original_nan_count = raw_data_with_nans.isnull().sum().sum()
    assert original_nan_count > 0

    df_cleaned = clean_data(raw_data_with_nans)

    assert df_cleaned.isnull().sum().sum() == 0
    assert len(df_cleaned) <= len(raw_data_with_nans)


def test_sequence_creation(raw_ohlcv_data):
    WINDOW_SIZE = 5
    X, y = create_sequences(raw_ohlcv_data, window_size=WINDOW_SIZE)

    expected_samples = len(raw_ohlcv_data) - WINDOW_SIZE
    assert len(X) == expected_samples
    assert len(y) == expected_samples

    n_features = len(raw_ohlcv_data.columns)
    assert X.shape[1] == WINDOW_SIZE * n_features