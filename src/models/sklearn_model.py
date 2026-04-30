from sklearn.base import BaseEstimator

from .model_interface import BaseModel


class SklearnModel(BaseModel):

    def __init__(self, model: BaseEstimator = None):
        self.model = model
        self.metrics = {}

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        self.model.fit(X_train, y_train)

    def _predict_internal(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

    def get_params(self):
        return self.model.get_params()

    def get_metrics(self):
        return self.metrics
