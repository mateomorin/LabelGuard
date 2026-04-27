from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import torch

from models.sklearn_model import SklearnModel
from models.torch_model import TorchMLPClassifier
from models.xgboost_model import XGBoostModel

X, y = load_breast_cancer(return_X_y=True)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, train_size=0.7, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)


def test_logreg():
    model_1 = SklearnModel(
        LogisticRegression(
            C=1.0,
            random_state=1,
            max_iter=100
        )
    )
    model_2 = SklearnModel(
        LogisticRegression(
            C=1.0,
            random_state=1,
            max_iter=100
        )
    )

    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)

    y_pred_1 = model_1.predict(X_eval)
    y_pred_2 = model_2.predict(X_eval)
    assert np.array_equal(y_pred_1, y_pred_2), "Issue on RNG training: predict"
    y_pred_1 = model_1.predict_proba(X_eval)
    y_pred_2 = model_2.predict_proba(X_eval)
    assert np.array_equal(y_pred_1, y_pred_2), "Issue on RNG training: predict proba"

    model = SklearnModel(
        LogisticRegression(
            C=1.0,
            random_state=1,
            max_iter=100
        )
    )
    model.fit(X_train, y_train, X_eval, y_eval)

    model.get_params()
    model.get_metrics()


def test_svm():
    model_1 = SklearnModel(
        SVC(
            C=1.0,
            random_state=1,
            max_iter=2000,
            kernel="linear",
            probability=True
        )
    )

    model_2 = SklearnModel(
        SVC(
            C=1.0,
            random_state=1,
            max_iter=2000,
            kernel="linear",
            probability=True
        )
    )

    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)

    y_pred_1 = model_1.predict(X_eval)
    y_pred_2 = model_2.predict(X_eval)
    assert np.array_equal(y_pred_1, y_pred_2), "Issue on RNG training: predict"
    y_pred_1 = model_1.predict_proba(X_eval)
    y_pred_2 = model_2.predict_proba(X_eval)
    assert np.array_equal(y_pred_1, y_pred_2), "Issue on RNG training: predict proba"

    model = SklearnModel(
        SVC(
            C=1.0,
            random_state=1,
            max_iter=2000,
            kernel="linear",
            probability=True
        )
    )
    model.fit(X_train, y_train, X_eval, y_eval)

    model.get_params()
    model.get_metrics()


def test_mlp():
    torch.manual_seed(1)
    model_1 = TorchMLPClassifier(
        input_dim=X.shape[1],
        hidden_layers=[12, 6],
        lr=1e-3,
        activation=torch.nn.ReLU(),
        epochs=10,
        batch_size=32
    )
    model_1.fit(X_train, y_train)
    y_pred_1 = model_1.predict(X_eval)
    y_proba_1 = model_1.predict_proba(X_eval)

    torch.manual_seed(1)
    model_2 = TorchMLPClassifier(
        input_dim=X.shape[1],
        hidden_layers=[12, 6],
        lr=1e-3,
        activation=torch.nn.ReLU(),
        epochs=10,
        batch_size=32
    )
    model_2.fit(X_train, y_train)
    y_pred_2 = model_2.predict(X_eval)
    y_proba_2 = model_2.predict_proba(X_eval)

    assert np.array_equal(y_pred_1, y_pred_2), "Issue on RNG training: predict"
    assert np.array_equal(y_proba_1, y_proba_2), "Issue on RNG training: predict proba"

    model = TorchMLPClassifier(
        input_dim=X.shape[1],
        hidden_layers=[12, 6],
        lr=1e-3,
        activation=torch.nn.ReLU(),
        epochs=10,
        batch_size=32
    )

    model.fit(X_train, y_train, X_eval, y_eval)

    model.get_params()
    model.get_metrics()


def test_xgb():
    model_1 = XGBoostModel(
        n_estimators=10,
        learning_rate=0.5,
        max_depth=2,
        min_split_loss=0,
        subsample=0.7,
        colsample_bytree=0.3,
        random_state=1
    )

    model_2 = XGBoostModel(
        n_estimators=10,
        learning_rate=0.5,
        max_depth=2,
        min_split_loss=0,
        subsample=0.7,
        colsample_bytree=0.3,
        random_state=1
    )

    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)

    y_pred_1 = model_1.predict(X_eval)
    y_pred_2 = model_2.predict(X_eval)
    assert np.array_equal(y_pred_1, y_pred_2), "Issue on RNG training: predict"
    y_pred_1 = model_1.predict_proba(X_eval)
    y_pred_2 = model_2.predict_proba(X_eval)
    assert np.array_equal(y_pred_1, y_pred_2), "Issue on RNG training: predict proba"

    model = XGBoostModel(
        n_estimators=10,
        learning_rate=0.5,
        max_depth=2,
        min_split_loss=0,
        subsample=0.7,
        colsample_bytree=0.3
    )
    model.fit(X_train, y_train, X_eval, y_eval)

    model.get_params()
    model.get_metrics()
