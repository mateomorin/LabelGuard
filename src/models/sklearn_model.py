import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator

from .model_interface import BaseModel


class SklearnModel(BaseModel):

    def __init__(self, model: BaseEstimator):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

    def get_params(self):
        return self.model.get_params()

    def save(self, name: str = "model"):
        """
        Upload to MLFlow.
        """

        mlflow.sklearn.log_model(
            sk_model=self.model,
            name=name,
        )

    @classmethod
    def load(cls, model_uri: str):
        """
        Download from MLFlow.
        """

        model = mlflow.sklearn.load_model(model_uri)

        return model
