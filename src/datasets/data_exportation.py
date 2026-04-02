import s3fs
import numpy as np
import pandas as pd


def export_data(
    fs: s3fs.S3FileSystem,
    path: str,
    texts: list[str],
    prediction: np.ndarray
) -> bool:

    df = pd.Dataframe(
        {
            "label": texts,
            "prediction": prediction
        }
    )

    df.to_parquet(path, filesystem=fs)
