import logging

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_X_y(
        df_real: pd.DataFrame,
        df_synth: pd.DataFrame
        ) -> tuple[np.ndarray]:

    X = pd.concat(df_real["embeddings"], df_synth["embeddings"]).to_numpy()

    y = np.concatenate(np.zeros(len(df_real)), np.ones(len(df_synth)))

    return X, y


def create_train_test(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float,
        random_state: int
        ) -> tuple[np.ndarray]:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
