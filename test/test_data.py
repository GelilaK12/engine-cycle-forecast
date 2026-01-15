from scripts import data_processing

def test_raw_data_shape():
    df = data_processing.load_raw()
    assert df.shape[1] > 0
