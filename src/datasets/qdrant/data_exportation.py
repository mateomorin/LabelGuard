import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)
BATCH_SIZE = 100


def create_collection_if_not_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 4096
):
    """Vérifie l'existence d'une collection et la crée si nécessaire."""
    # On récupère la liste des collections existantes
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        logger.info(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE         # Ou DOT / EUCLID selon votre modèle
            ),
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")


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
