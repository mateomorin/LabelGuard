import os
from abc import ABC, abstractmethod
import mlflow.pyfunc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class BaseModel(ABC, mlflow.pyfunc.PythonModel):

    @abstractmethod
    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        pass

    def load_context(self, context):
        pass

    def predict(self, context, model_input=None):
        """For MLFlow (context is never used)"""
        if model_input is None:
            return self._predict_internal(context)
        return self._predict_internal(model_input)

    def final_eval(self, X_train, y_train, X_eval, y_eval):
        # ====================================
        #               Eval
        # ====================================
        y_train_pred = self._predict_internal(X_train)
        train_cfm = confusion_matrix(y_train, y_train_pred)
        t_n, f_p, f_n, t_p = train_cfm.ravel()

        self.metrics.update({
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "train_t_n": int(t_n),
            "train_f_p": int(f_p),
            "train_t_p": int(t_p),
            "train_f_n": int(f_n),
            "train_fpr": f_p/(f_p + t_n)
        })

        if X_eval is not None:
            y_eval_pred = self._predict_internal(X_eval)
            eval_cfm = confusion_matrix(y_eval, y_eval_pred)
            t_n, f_p, f_n, t_p = eval_cfm.ravel()
            self.metrics.update({
                "eval_accuracy": accuracy_score(y_eval, y_eval_pred),
                "eval_f1": f1_score(y_eval, y_eval_pred),
                "eval_t_n": int(t_n),
                "eval_f_p": int(f_p),
                "eval_t_p": int(t_p),
                "eval_f_n": int(f_n),
                "eval_fpr": f_p/(f_p + t_n)
            })

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
