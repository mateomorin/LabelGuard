"""
Script done by ChatGPT.
"""

import os
from dotenv import load_dotenv
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import s3fs
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from tqdm import tqdm
import uuid

load_dotenv(override=True)
# =========================
# CONFIG
# =========================
S3_PATH = "s3://mateom/graal/embeddings/NAF2025/original_train_cleaned.parquet"
COLLECTION_NAME = "original_cleaned"

QDRANT_URL = "http://qdrant:6333"
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

BATCH_SIZE = 100

# =========================
# INIT S3 + PARQUET
# =========================
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"]
)
parquet_file = pq.ParquetFile(S3_PATH, filesystem=fs)

# =========================
# INIT QDRANT
# =========================
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE),
)


# =========================
# UPSERT BATCH
# =========================
def upload_batch(batch_rows):
    points = []

    for row in batch_rows:
        try:
            vector = np.array(row["embedding"], dtype=np.float32)

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={
                        "code": row["code"],
                        "label": row["label"]
                    }
                )
            )
        except Exception as e:
            print(f"Erreur parsing ligne: {e}")

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )


# =========================
# STREAMING PAR ROW GROUP
# =========================
print(f"Nombre de row groups: {parquet_file.num_row_groups}")

for rg_idx in tqdm(range(parquet_file.num_row_groups)):

    table: pa.Table = parquet_file.read_row_group(rg_idx, columns=["code", "label", "embedding"])

    # Convertir en python natif sans exploser la RAM
    batch = table.to_pylist()

    # =========================
    # ENVOI PAR SOUS-BATCH
    # =========================
    for start in range(0, len(batch), BATCH_SIZE):
        sub_batch = batch[start:start + BATCH_SIZE]

        try:
            upload_batch(sub_batch)
        except Exception as e:
            print(f"Erreur batch RG {rg_idx} - offset {start}: {e}")

# =========================
# INDEXING
# =========================

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="code",
    field_schema=models.PayloadSchemaType.KEYWORD,
)