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
            learning_rate=learning_rate,
            gamma=min_split_loss,
            max_depth=max_depth,
            seed=random_state,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric="logloss"
        )
        self.metrics = {}

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        # ====================================
        #               Train
        # ====================================
        eval_set = [(X_train, y_train)]
        if X_eval is not None:
            eval_set.append((X_eval, y_eval))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )

        # ====================================
        #           Loss evolution
        # ====================================
        evals_result = self.model.evals_result()
        self.metrics["train_logloss"] = evals_result["validation_0"]["logloss"]

        if X_eval is not None:
            self.metrics["eval_logloss"] = evals_result["validation_1"]["logloss"]

    def _predict_internal(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

    def get_params(self):
        return self.model.get_params()

    def get_metrics(self):
        return self.metrics
