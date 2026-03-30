from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .sklearn_model import SklearnModel
from .torch_model import TorchMLPClassifier


def build_model(cfg):

    name = cfg["name"]

    if name == "logreg":
        logreg_cfg = cfg["logreg"]
        model = SklearnModel(
            LogisticRegression(
                C=logreg_cfg["C"] or 1.0,
                random_state=logreg_cfg["random_state"],
                solver=logreg_cfg["solver"] or "sag",
                max_iter=logreg_cfg["max_iter"] or 100
            )
        )

        return model

    if name == "svm":
        svm_cfg = cfg["svm"]
        model = SklearnModel(
            LinearSVC(
                C=svm_cfg["C"] or 1.0,
                random_state=logreg_cfg["random_state"],
                max_iter=logreg_cfg["max_iter"] or 1000
            )
        )

        return model

    if name == "mlp":
        mlp_cfg = cfg["mlp"]
        model = TorchMLPClassifier(
            input_dim=mlp_cfg["input_dim"],
            hidden_layers=mlp_cfg["hidden_layers"],
            loss_fn=mlp_cfg["loss_fn"],
            optimizer_cls=mlp_cfg["optimizer_cls"],
            lr=mlp_cfg["lr"],
            device=mlp_cfg["device"],
        )

        return model

    raise ValueError("Unknown model")
