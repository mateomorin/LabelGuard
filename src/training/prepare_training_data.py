import os
import logging
from collections import Counter

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig
import pandas as pd
from qdrant_client import QdrantClient

from src.datasets.qdrant import data_importation, data_preprocessing, data_exportation


logger = logging.getLogger(__name__)
load_dotenv(override=True)


@hydra.main(version_base=None, config_path="", config_name="training_data")
def main(cfg: DictConfig):
    client = QdrantClient(
        url="http://qdrant:6333",
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=30
    )

    logger.info("Creating training dataset...")

    df_real = data_importation.fetch_original_points(
        client=client,
        collection_name=cfg["qdrant"]["original_collection"],
        size=cfg["data"]["n_samples"],
        min_size_per_code=cfg["data"]["min_size_per_code"],
        random_state=cfg["random_state"]
    )

    code_list = Counter(df_real["code"].to_list())

    df_synth = data_importation.select_synthetic_data(
        client=client,
        collection_name=cfg["qdrant"]["synth_collection"],
        code_list=code_list,
        random_state=cfg["random_state"]
    )

    X_train, X_eval, y_train, y_eval, indices_train, indices_eval = data_preprocessing.create_train_test(
        df_real=df_real,
        df_synth=df_synth,
        train_size=cfg["data"]["train_size"],
        random_state=cfg["random_state"]
    )

    logger.info("Exporting data...")
    
    df_train = 