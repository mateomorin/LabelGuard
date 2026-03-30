"""
Correct the format of the .parquet dataset so that it can be read by duckdb.
"""

import os

import pyarrow.parquet as pq
import pyarrow as pa
import s3fs


def main():

    OLD_PATH = 's3://mateom/graal/embeddings/NAF2025/original_train.parquet'
    NEW_PATH = 's3://mateom/graal/embeddings/NAF2025/original_train_corrected.parquet'

    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"]
    )

    with fs.open(OLD_PATH) as f:
        reader = pq.ParquetFile(f)

        # On définit une taille de groupe raisonnable (ex: 50 000 lignes)
        writer = None

        # iter_batches permet de ne lire qu'une fraction du Row Group à la fois
        # même s'il n'y a qu'un seul Row Group physique.
        for batch in reader.iter_batches(batch_size=50000, columns=['code', 'label', 'embedding']):
            if writer is None:
                writer = pq.ParquetWriter(NEW_PATH, batch.schema, compression='snappy')

            writer.write_table(pa.Table.from_batches([batch]))
            print("50 000 lines have been treated...")

        if writer:
            writer.close()


if __name__ == "__main__":
    main()
