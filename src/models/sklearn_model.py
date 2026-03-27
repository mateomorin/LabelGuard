import joblib
from .model_interface import BaseModel


class SklearnModel(BaseModel):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        model = joblib.load(path)
        return cls(model)
