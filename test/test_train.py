 #No need to test as pipeline uses pkl of an already trained model

'''
  from scripts import train_model

def test_train_runs():
    model = train_model.train()
    assert hasattr(model, "predict")

'''