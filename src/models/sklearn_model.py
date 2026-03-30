import mlflow
import mlflow.sklearn

from .model_interface import BaseModel


class SklearnModel(BaseModel):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

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
