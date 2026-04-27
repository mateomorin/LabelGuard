import os
from abc import ABC, abstractmethod
import mlflow.pyfunc


class BaseModel(ABC, mlflow.pyfunc.PythonModel):

    @abstractmethod
    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        pass

    def load_context(self, context):
        pass

    def predict(self, model_input, context=None):
        """For MLFlow (context is never used)"""
        return self._predict_internal(model_input)

    @abstractmethod
    def _predict_internal(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def get_metrics(self):
        pass

    def save(self, name: str = "model"):
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"
        mlflow.pyfunc.log_model(
            name=name,
            python_model=self
        )

    @classmethod
    def load(cls, model_uri: str):
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"
        return mlflow.pyfunc.load_model(model_uri)
