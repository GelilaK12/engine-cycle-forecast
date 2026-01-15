from scripts import train_model
def test_train_runs():
    model = train_model.train()
    assert hasattr(model, "predict")
