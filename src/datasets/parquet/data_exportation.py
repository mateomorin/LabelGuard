import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
import numpy as np


def export_data(
    fs: s3fs.S3FileSystem,
    path: str,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray
):
    """
    Export to S3 in Parquet format.
    """
    df = pd.DataFrame({
        "embedding": list(X),
        "label": y,
        "original_index": indices
    })

    table = pa.Table.from_pandas(df)

    with fs.open(path, 'wb') as f:
        pq.write_table(table, f, compression='snappy')
