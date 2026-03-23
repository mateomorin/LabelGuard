import logging

import s3fs
import polars as pl
import pandas as pd


def fetch_original_data(
        fs: s3fs.S3FileSystem,
        path: str,
        n_samples: int
        ) -> pd.DataFrame:

    with fs.open(path, "rb") as f:
        lf = pl.scan_parquet(f)

        # TODO: assert exhaustivity 

        # TODO: sample n_samples rows without replacement

    # TODO: collect

    pass


def select_synthetic_data(
        fs: s3fs.S3FileSystem,
        path: str,
        code_list: list[str]
        ):

    with fs.open(path, "rb") as f:
        lf = pl.scan_parquet(f)

        # TODO: sample with replacement using codes from code_list

    # TODO: collect

    pass
