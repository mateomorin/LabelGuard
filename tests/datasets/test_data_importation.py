from collections import Counter
import duckdb
import numpy as np
import pandas as pd
from datasets import data_importation

CON = duckdb.connect()
RNG = np.random.default_rng()
REAL_PATH = "s3://mateom/graal/embeddings/NAF2025/original_train_cleaned.parquet"
SYNTH_PATH = "s3://mateom/graal/embeddings/NAF2025/2026-03-16_gemma_synth.parquet"
CODE_LIST = CON.execute(
    f"""
    SELECT DISTINCT(code)
    FROM read_parquet('{REAL_PATH}')
    ORDER BY code
    """
).df()["code"].to_list()


def test_get_n_codes():
    data_importation.get_n_codes(CON, REAL_PATH)
    data_importation.get_n_codes(CON, SYNTH_PATH)


def test_original_data_sampling():
    n_codes = data_importation.get_n_codes(CON, REAL_PATH)
    n_samples = n_codes + 10

    rng = np.random.default_rng(1)
    selected_ids_1, n_rows_1 = data_importation.exhaustive_sampling(CON, REAL_PATH, rng)
    final_ids_1 = data_importation.remaining_sampling(n_samples, n_codes, n_rows_1, selected_ids_1, rng)

    selected_ids_1 = pd.DataFrame({"row_id": final_ids_1})
    CON.register("selected_ids_1", selected_ids_1)
    sampled_df_1 = data_importation.retrieve_matching_rows(CON, REAL_PATH, "selected_ids_1")

    rng = np.random.default_rng(1)
    selected_ids_2, n_rows_2 = data_importation.exhaustive_sampling(CON, REAL_PATH, rng)
    final_ids_2 = data_importation.remaining_sampling(n_samples, n_codes, n_rows_2, selected_ids_2, rng)
    selected_ids_2 = pd.DataFrame({"row_id": final_ids_2})
    CON.register("selected_ids_2", selected_ids_2)
    sampled_df_2 = data_importation.retrieve_matching_rows(CON, REAL_PATH, "selected_ids_2")

    # Attributes
    assert "code" in sampled_df_1.columns, "No column code in final sample"
    assert "label" in sampled_df_1.columns, "No column label in final sample"
    assert "embedding" in sampled_df_1.columns, "No column embedding in final sample"

    # Exhaustivity
    code_list_1 = np.sort(sampled_df_1["code"].unique())
    assert np.array_equal(code_list_1, np.array(CODE_LIST)), "No exhaustivity in final sample"

    # RNG
    assert n_rows_1 == n_rows_2, "Issue on row counting"
    assert np.array_equal(selected_ids_1, selected_ids_2), "Issue on RNG for exhaustive sampling"
    assert np.array_equal(final_ids_1, final_ids_2), "Issue on RNG for remaining sampling"
    assert sampled_df_1[["code", "label"]].equals(sampled_df_2[["code", "label"]]), "Matched samples are not the same"


def test_sample_code_equivalents():
    # Single sampling
    code_counts = Counter(CODE_LIST)

    rng = np.random.default_rng(1)
    all_row_ids_1 = data_importation.sample_code_equivalents(CON, SYNTH_PATH, rng, code_counts)

    rng = np.random.default_rng(1)
    all_row_ids_2 = data_importation.sample_code_equivalents(CON, SYNTH_PATH, rng, code_counts)

    assert all_row_ids_1.equals(all_row_ids_2), "Issue on RNG for code equivalent single sampling"

    # Multiple sampling
    code_counts = Counter(CODE_LIST * 3)

    rng = np.random.default_rng(1)
    all_row_ids_1 = data_importation.sample_code_equivalents(CON, SYNTH_PATH, rng, code_counts)

    rng = np.random.default_rng(1)
    all_row_ids_2 = data_importation.sample_code_equivalents(CON, SYNTH_PATH, rng, code_counts)

    assert all_row_ids_1.equals(all_row_ids_2), "Issue on RNG for code equivalent sampling"
