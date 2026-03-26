from sklearn.linear_model import LogisticRegression
from .sklearn_model import SklearnModel
from .torch_model import TorchModel


def build_torch():
    return TorchModel()


def build_LogReg():
    return SklearnModel(LogisticRegression())


def build_model(config):

    name = config["name"]

    if name == "logreg":
        pass

    if name == "torch_mlp":
        pass

    raise ValueError("Unknown model")
