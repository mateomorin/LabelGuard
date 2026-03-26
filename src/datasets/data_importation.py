import duckdb
import pandas as pd
import numpy as np
from collections import Counter


def fetch_original_data(
        path: str,
        n_samples: int,
        random_state: int = None
        ) -> pd.DataFrame:
    """
    Fetch the original parquet dataset with columns ["code", "embedding"] from S3.
    Each code is sampled at least one.

    args:
        fs (s3fs.S3FileSystem): filesystem (for S3)
        path (str): path of the dataset in S3
        n_samples (int): number of samples to draw from the original data
            n_samples >= #distinct codes
        random_state (int): seed for the random sampling

    returns:
        pd.DataFrame: a dataframe with columns code and embedding.
    """

    con = duckdb.connect()
    rng = np.random.default_rng(seed=random_state)

    # ===============================
    #          Check n_sample
    # ===============================
    code_len_query = f"""
    SELECT COUNT(DISTINCT(code))
    FROM read_parquet('{path}')
    """

    n_codes = con.execute(code_len_query).fetchone()[0]

    assert n_samples >= n_codes, f"Please provide n_sample >= {n_codes}"

    # ===============================
    #       Exhaustive sampling
    # ===============================

    code_row_count = con.execute(f"""
    SELECT code, COUNT(*) AS n_rows
    FROM read_parquet('{path}')
    GROUP BY code
    ORDER BY code
    """).df()

    code_row_count["k"] = code_row_count["n_rows"].apply(
        lambda n: rng.integers(1, n + 1)
    )
    code_row_count["offset"] = code_row_count["n_rows"].cumsum() - code_row_count["n_rows"]
    code_row_count["row_id"] = code_row_count["offset"] + code_row_count["k"]

    selected_ids = code_row_count["row_id"].to_numpy()

    # ===============================
    #       Remaining sampling
    # ===============================

    remaining = n_samples - n_codes

    # Adjust if remaining is too high
    n_rows = code_row_count["n_rows"].sum()
    remaining = min(n_rows, remaining)

    # Sample the remaining codes
    if remaining > 0:
        remaining_ids = np.delete(np.arange(1, n_rows + 1), selected_ids - 1)
        random_ids = rng.choice(remaining_ids, size=remaining, replace=False)
        final_ids = np.concatenate([selected_ids, random_ids])
    else:
        final_ids = selected_ids

    con.register("selected_ids", {"row_id": final_ids})

    query = f"""
    SELECT t.code, t.embedding
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (ORDER BY code) AS row_id
        FROM read_parquet('{path}')
    ) t
    JOIN selected_ids s
    USING (row_id)
    """

    sampled_df = con.execute(query).df()

    return sampled_df


def select_synthetic_data(
        path: str,
        code_list: list[str],
        random_state: int
        ) -> pd.DataFrame:
    """
    Fetch the synthetic parquet dataset with columns ["code", "embedding"] from s3.
    Sample it with replacement according to the distribution of codes in code_list.

    args:
        fs (s3fs.S3FileSystem): filesystem (for S3)
        path (str): path of the dataset
        code_list (list): codes to sample (can contain duplictes)
        random_state (int): seed for the random sampling

    returns:
        pd.DataFrame: a dataframe with columns code and embedding.
    """
    con = duckdb.connect()
    rng = np.random.default_rng(seed=random_state)

    # ===============================
    #    Counting codes to sample
    # ===============================
    code_counts = Counter(code_list)

    # Pick a random row_id for each code
    code_row_count = con.execute(f"""
    SELECT code, COUNT(*) AS n_rows
    FROM read_parquet('{path}')
    GROUP BY code
    ORDER BY code
    """).df()

    code_row_count["k"] = code_row_count.apply(
        lambda x: rng.integers(1, x["n_rows"] + 1, size=code_counts[x["code"]]),
        axis=1
    )
    code_row_count["offset"] = code_row_count["n_rows"].cumsum() - code_row_count["n_rows"]
    code_row_count["row_id"] = code_row_count["offset"] + code_row_count["k"]

    selected_ids = code_row_count["row_id"].to_numpy()

    con.register("selected_ids", {"row_id": code_list})

    query = f"""
    SELECT t.code, t.embedding
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (ORDER BY code) AS row_id
        FROM read_parquet('{path}')
    ) t
    JOIN selected_ids s
    USING (row_id)
    """

    sampled_df = con.execute(query).df()

    return sampled_df
