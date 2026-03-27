import logging

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_train_test(
        df_real: pd.DataFrame,
        df_synth: pd.DataFrame,
        train_size: float,
        random_state: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train and test dataset from the combination of orginal and synthetic ones.

    Args:
        df_real (pd.DataFrame): the original dataset
        df_synth (pd.DataFrame): the synthetic dataset
        train_size (float): the proportion of elements in the training dataset
        random_state (int): seed for the random splitting
    """

    X = pd.concat([df_real["embedding"], df_synth["embedding"]]).values
    X = np.vstack(X)
    y = np.append(np.zeros(len(df_real)), np.ones(len(df_synth))).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
