import pytest
import pandas as pd
from scripts.data_processing import load_raw

def test_raw_data_shape():
    try:
        df = load_raw()
    except FileNotFoundError:
        pytest.skip("Raw training data not available in CI environment")

    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] > 0
