import pandas as pd
import numpy as np
from datasets import data_preprocessing

DF_REAL = pd.DataFrame({
    "code": [i for i in range(5)],
    "label": [str(i) for i in range(5)],
    "embedding": [[i, i] for i in range(5)]
})
DF_SYNTH = pd.DataFrame({
    "code": [i for i in range(5, 10)],
    "label": [str(i) for i in range(5, 10)],
    "embedding": [[i, i] for i in range(5, 10)]
})


def test_create_train_test_rng():
    X_train_1, X_test_1, y_train_1, y_test_1, indices_train_1, indices_test_1 = data_preprocessing.create_train_test(
        df_real=DF_REAL,
        df_synth=DF_SYNTH,
        train_size=0.7,
        random_state=1
    )

    X_train_2, X_test_2, y_train_2, y_test_2, indices_train_2, indices_test_2 = data_preprocessing.create_train_test(
        df_real=DF_REAL,
        df_synth=DF_SYNTH,
        train_size=0.7,
        random_state=1
    )

    assert np.array_equal(X_train_1, X_train_2), "Issue on RNG for X_train"
    assert np.array_equal(y_train_1, y_train_2), "Issue on RNG for y_train"
    assert np.array_equal(indices_train_1, indices_train_2), "Issue on RNG for indices_train"
    assert np.array_equal(X_test_1, X_test_2), "Issue on RNG for X_test"
    assert np.array_equal(y_test_1, y_test_2), "Issue on RNG for y_test"
    assert np.array_equal(indices_test_1, indices_test_2), "Issue on RNG for indices_test"


def test_create_train_test_valid():
    X_train, X_test, y_train, y_test, indices_train, indices_test = data_preprocessing.create_train_test(
        df_real=DF_REAL,
        df_synth=DF_SYNTH,
        train_size=0.7
    )

    X = pd.concat([DF_REAL["embedding"], DF_SYNTH["embedding"]]).values
    X = np.vstack(X)
    y = np.append(np.zeros(len(DF_REAL)), np.ones(len(DF_SYNTH)))

    assert np.array_equal(y_train, y[indices_train]), "Issue on permutations for y"
    assert np.array_equal(y_test, y[indices_test]), "Issue on permutations for y"
    assert X_train.shape == (7, 2), "Issue on X_train shape"
    assert X_test.shape == (3, 2), "Issue on X_test shape"
    assert y_train.shape == (7,), "Issue on y_train shape"
    assert y_test.shape == (3,), "Issue on X_test shape"
