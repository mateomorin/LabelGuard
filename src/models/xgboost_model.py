import mlflow
import mlflow.xgboost

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

    def save(self, name: str = "model"):
        """
        Upload to MLFlow.
        """

        mlflow.xgboost.log_model(
            sk_model=self.model,
            name=name,
        )

    @classmethod
    def load(cls, model_uri: str):
        """
        Download from MLFlow.
        """

        model = mlflow.xgboost.load_model(model_uri)

        return model
