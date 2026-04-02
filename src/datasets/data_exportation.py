import s3fs
import numpy as np
import pandas as pd


def save_results(
    fs: s3fs.S3FileSystem,
    path: str,
    texts: list[str],
    prediction: np.ndarray,
    true_labels: np.ndarray
) -> bool:

    df = pd.Dataframe(
        {
            "texts": texts,
            "prediction": prediction,
            "true_labels": true_labels
        }
    )

    df.to_parquet(path, filesystem=fs)
