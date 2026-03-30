import mlflow
import mlflow.xgboost
import xgboost as xgb

from .model_interface import BaseModel


class XGBoostModel(BaseModel):

    def __init__(
            self,
            n_estimators: int = 10,
            learning_rate: float = 0.5,
            max_depth: int = 2,
            min_split_loss: float = 0,
            subsample: float = 0.7,
            colsample_bytree: float = 0.3,
            random_state: int = None
            ):

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            eta=learning_rate,
            gamma=min_split_loss,
            max_depth=max_depth,
            seed=random_state,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )

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
