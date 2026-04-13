from collections import Counter
import duckdb
import pandas as pd
import numpy as np


def get_n_codes(con, path):
    """
    Get the number of distinct codes in the database.
    """

    code_len_query = f"""
    SELECT COUNT(DISTINCT(code))
    FROM read_parquet('{path}')
    """

    return con.execute(code_len_query).fetchone()[0]


def exhaustive_sampling(con, path, rng):
    """
    Sample one row per code for every code.
    Returns the selected row ids and the total number of rows.
    """

    # Retrieving the number of labels per codes
    code_row_count = con.execute(f"""
    SELECT code, COUNT(*) AS n_rows
    FROM read_parquet('{path}')
    GROUP BY code
    ORDER BY code
    """).df()

    # Sampling
    code_row_count["row_id"] = code_row_count["n_rows"].apply(
        lambda n: rng.integers(1, n + 1)
    )
    code_row_count["offset"] = code_row_count["n_rows"].cumsum() - code_row_count["n_rows"]
    code_row_count["row_id"] = code_row_count["offset"] + code_row_count["row_id"]

    return code_row_count["row_id"].to_numpy(), code_row_count["n_rows"].sum()


def remaining_sampling(n_samples, n_codes, n_rows, selected_ids, rng):
    """
    Sample the remaining codes uniformly to get n_samples samples.
    selected_ids are the ids that have already been selected and that should not be selected again.
    """
    remaining = n_samples - n_codes

    # Adjust if remaining is too high
    remaining = min(n_rows, remaining)

    # Sample the remaining codes
    if remaining > 0:
        remaining_ids = np.delete(np.arange(1, n_rows + 1), selected_ids - 1)
        random_ids = rng.choice(remaining_ids, size=remaining, replace=False)
        final_ids = np.concatenate([selected_ids, random_ids])
    else:
        final_ids = selected_ids

    return final_ids


def retrieve_matching_rows(con, path, selected_ids_table_name):
    """
    Retrieve the rows from the database sharing the same ids as in selected_ids.
    """

    query = f"""
    SELECT t.code, t.label, t.embedding
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (ORDER BY (code, label)) AS row_id
        FROM read_parquet('{path}')
    ) t
    JOIN {selected_ids_table_name} s
    USING (row_id)
    ORDER BY t.row_id
    """

    return con.execute(query).df()


def fetch_original_data(
    path: str,
    n_samples: int,
    random_state: int = None
) -> pd.DataFrame:
    """
    Fetch the original parquet dataset with columns ["code", "label", "embedding"] from S3.
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

    n_codes = get_n_codes(con, path)
    assert n_samples >= n_codes, f"Please provide n_sample >= {n_codes}"

    selected_ids, n_rows = exhaustive_sampling(con, path, rng)

    final_ids = remaining_sampling(n_samples, n_codes, n_rows, selected_ids, rng)

    selected_ids = pd.DataFrame({"row_id": final_ids})
    con.register("selected_ids", selected_ids)

    sampled_df = retrieve_matching_rows(con, path, "selected_ids")

    return sampled_df


def sample_code_equivalents(con, path, rng, code_counts):
    """
    Sample row ids so that there is still the same counts for code as in code_counts.
    """
    # Retrieving number of labels per codes
    code_row_count = con.execute(f"""
    SELECT code, COUNT(*) AS n_rows
    FROM read_parquet('{path}')
    GROUP BY code
    ORDER BY code
    """).df()

    # Selecting the correct number of random labels
    code_row_count["row_id"] = code_row_count.apply(
        lambda x: rng.integers(1, x["n_rows"] + 1, size=code_counts[x["code"]]),
        axis=1
    )

    # Converting into global row id for join
    code_row_count["offset"] = code_row_count["n_rows"].cumsum() - code_row_count["n_rows"]
    code_row_count["row_id"] = code_row_count["offset"] + code_row_count["row_id"]

    return code_row_count["row_id"].explode().to_frame()


def select_synthetic_data(
    path: str,
    code_list: list[str],
    random_state: int
) -> pd.DataFrame:
    """
    Fetch the synthetic parquet dataset with columns ["code", "label", "embedding"] from s3.
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

    code_counts = Counter(code_list)

    all_row_ids = sample_code_equivalents(con, path, rng, code_counts)

    con.register("selected_ids", all_row_ids[["row_id"]])

    sampled_df = retrieve_matching_rows(con, path, "selected_ids")

    return sampled_df
