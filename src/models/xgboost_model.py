import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):

        # ====================================
        #               Train
        # ====================================
        eval_set = [(X_train, y_train)]
        eval_names = ["train"]

        if X_eval is not None:
            eval_set.append((X_eval, y_eval))
            eval_names.append("eval")

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

        self.metrics = {}
        self.metrics["train_logloss"] = evals_result["validation_0"]["logloss"]

        if X_eval is not None:
            self.metrics["eval_logloss"] = evals_result["validation_1"]["logloss"]

        # ====================================
        #           Final metrics
        # ====================================
        y_train_pred = self.model.predict(X_train)

        train_cfm = confusion_matrix(y_train, y_train_pred)

        t_n, f_p, f_n, t_p = train_cfm.ravel()

        self.metrics.update({
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "train_t_n": t_n,
            "train_f_p": f_p,
            "train_t_p": t_p,
            "train_f_n": f_n,
        })

        if X_eval is not None:
            y_eval_pred = self.model.predict(X_eval)

            eval_cfm = confusion_matrix(y_eval, y_eval_pred)

            t_n, f_p, f_n, t_p = eval_cfm.ravel()
            self.metrics.update({
                "eval_accuracy": accuracy_score(y_eval, y_eval_pred),
                "eval_f1": f1_score(y_eval, y_eval_pred),
                "eval_t_n": t_n,
                "eval_f_p": f_p,
                "eval_t_p": t_p,
                "eval_f_n": f_n,
            })

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

    def get_params(self):
        return self.model.get_params()

    def get_metrics(self):
        return self.metrics

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
