from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from hydra.utils import instantiate

from .sklearn_model import SklearnModel
from .torch_model import TorchMLPClassifier
from .xgboost_model import XGBoostModel


def build_model(cfg):

    name = cfg["model"]["name"]

    if name == "logreg":
        logreg_cfg = cfg["model"]["logreg"]
        model = SklearnModel(
            LogisticRegression(
                C=logreg_cfg["C"] or 1.0,
                random_state=cfg["random_state"],
                solver=logreg_cfg["solver"] or "sag",
                max_iter=logreg_cfg["max_iter"] or 100
            )
        )

        return model

    elif name == "svm":
        svm_cfg = cfg["model"]["svm"]
        model = SklearnModel(
            SVC(
                C=svm_cfg["C"] or 1.0,
                random_state=cfg["random_state"],
                max_iter=svm_cfg["max_iter"] or 1000,
                kernel="linear",
                probability=True
            )
        )

        return model

    elif name == "mlp":
        mlp_cfg = cfg["model"]["mlp"]
        model = TorchMLPClassifier(
            input_dim=mlp_cfg["input_dim"],
            hidden_layers=mlp_cfg["hidden_layers"],
            loss_fn=instantiate(mlp_cfg["loss_fn"]),
            activation=instantiate(mlp_cfg["activation"]),
            lr=mlp_cfg["lr"],
            epochs=mlp_cfg["epochs"],
            batch_size=mlp_cfg["batch_size"]
        )

        return model

    elif name == "xgboost":
        xgb_cfg = cfg["model"]["xgboost"]
        model = XGBoostModel(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            min_split_loss=xgb_cfg["min_split_loss"],
            max_depth=xgb_cfg["max_depth"],
            seed=xgb_cfg["random_state"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            random_state=cfg["random_state"]
        )

    raise ValueError("Unknown model")
