from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .model_interface import BaseModel


class SklearnModel(BaseModel):

    def __init__(self, model: BaseEstimator = None):
        self.model = model
        self.metrics = {}

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        # ====================================
        #               Train
        # ====================================
        self.model.fit(X_train, y_train)

        # ====================================
        #               Eval
        # ====================================
        y_train_pred = self.model.predict(X_train)
        train_cfm = confusion_matrix(y_train, y_train_pred)
        t_n, f_p, f_n, t_p = train_cfm.ravel()

        self.metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "train_t_n": int(t_n),
            "train_f_p": int(f_p),
            "train_t_p": int(t_p),
            "train_f_n": int(f_n),
        }

        if X_eval is not None:
            y_eval_pred = self.model.predict(X_eval)
            eval_cfm = confusion_matrix(y_eval, y_eval_pred)
            t_n, f_p, f_n, t_p = eval_cfm.ravel()
            self.metrics.update({
                "eval_accuracy": accuracy_score(y_eval, y_eval_pred),
                "eval_f1": f1_score(y_eval, y_eval_pred),
                "eval_t_n": int(t_n),
                "eval_f_p": int(f_p),
                "eval_t_p": int(t_p),
                "eval_f_n": int(f_n),
            })

    def _predict_internal(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

    def get_params(self):
        return self.model.get_params()

    def get_metrics(self):
        return self.metrics
