import os
import logging
from collections import Counter

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig
from qdrant_client import QdrantClient

from src.datasets.qdrant import data_importation, data_preprocessing, data_exportation


logger = logging.getLogger(__name__)
load_dotenv(override=True)


@hydra.main(version_base=None, config_path="", config_name="training_data")
def main(cfg: DictConfig):
    client = QdrantClient(
        url="http://qdrant:6333",
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=120
    )

    logger.info("Importing original dataset...")

    original_points = data_importation.fetch_original_points(
        client=client,
        collection_name=cfg["qdrant"]["original_collection"],
        size=cfg["data"]["n_samples"],
        min_size_per_code=cfg["data"]["min_size_per_code"],
        random_state=cfg["random_state"]
    )

    payloads = data_preprocessing.get_payloads(original_points)
    code_list = Counter([payload["code"] for payload in payloads])

    logger.info("Matching synthetic dataset...")

    synthetic_points = data_importation.select_synthetic_points(
        client=client,
        collection_name=cfg["qdrant"]["synth_collection"],
        code_list=code_list,
        random_state=cfg["random_state"]
    )

    logger.info("Creating train test...")
    train_points, test_points = data_preprocessing.create_train_test(
        points_real=original_points,
        points_synth=synthetic_points,
        train_size=cfg["data"]["train_size"],
        random_state=cfg["random_state"]
    )

    logger.info("Exporting data...")

    data_exportation.export_points(
        client=client,
        train_points=train_points,
        test_points=test_points,
        collection_train=cfg["qdrant"]["collection_train"],
        collection_test=cfg["qdrant"]["collection_test"]
    )


if __name__ == "__main__":
    main()
