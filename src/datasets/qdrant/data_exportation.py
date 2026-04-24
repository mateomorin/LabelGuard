import logging
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)
BATCH_SIZE = 100


def export_points(
    client: QdrantClient,
    points: list,
    collection_name: str
) -> None:
    total = len(points)
    logger.info(f"Starting export to '{collection_name}' ({total} points)...")

    for i in range(0, total, BATCH_SIZE):
        batch = points[i: i + BATCH_SIZE]
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=False          # 'False' accélère l'envoi, Qdrant indexera en arrière-plan
        )
        if i % (BATCH_SIZE * 10) == 0:
            logger.info(f"Uploaded {i}/{total} points to {collection_name}...")
