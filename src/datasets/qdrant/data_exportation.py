import logging
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def export_points(
    client: QdrantClient,
    train_points: list,
    test_points: list,
    collection_train: str,
    collection_test: str
) -> None:
    logger.info("Exporting train dataset...")
    client.upsert(
        collection_name=collection_train,
        points=train_points
    )

    logger.info("Exporting test dataset...")
    client.upsert(
        collection_name=collection_test,
        points=test_points
    )
