
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
import uuid
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


def get_payloads(points: list):
    return [point.payload for point in points]


def get_vectors(points: list):
    return np.vstack([point.vector for point in points])


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def create_train_test(
    points_real: list,
    points_synth: list,
    train_size: float,
    random_state: int = None
) -> list:
    """
    Create train and test dataset from the combination of orginal and synthetic points.
    Standard scaling fit on X_train and only applied on X_test.
    """
    rng = np.random.default_rng(random_state)

    all_points = points_real + points_synth

    logger.info("Collecting vectors and payloads...")
    X = get_vectors(all_points)
    payloads = get_payloads(all_points)

    # Add labels for supervised learning
    logger.info("Creating targets...")
    y = np.append(np.zeros(len(points_real)), np.ones(len(points_real)))
    y.astype(np.float32)
    for payload, is_synth in zip(payloads, y):
        payload["is_synth"] = is_synth

    # Select random indices
    logger.info("Random splitting of indices...")
    indices = np.arange(len(X))
    indices_train = rng.choice(indices, size=int(train_size*len(X)), replace=False)
    indices_test = np.delete(indices, indices_train)
    rng.shuffle(indices_test)

    # Split and scale
    logger.info("Spliting and scaling data")
    X_train, X_test = scale_data(X[indices_train], X[indices_test])

    # Split
    payloads_train = [payloads[i] for i in indices_train]
    payloads_test = [payloads[i] for i in indices_test]

    logger.info("Creating training points...")
    train_points = [PointStruct(
        id=str(uuid.uuid4()),
        vector=X,
        payload=payload
    ) for X, payload in zip(X_train, payloads_train)]

    logger.info("Creating test points")
    test_points = [PointStruct(
        id=str(uuid.uuid4()),
        vector=X,
        payload=payload
    ) for X, payload in zip(X_test, payloads_test)]

    return train_points, test_points
