import logging
import os

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig
import mlflow
import torch
from qdrant_client import QdrantClient

from src.models import model_factory
from src.datasets.qdrant import data_importation


logger = logging.getLogger(__name__)
load_dotenv(override=True)


@hydra.main(version_base=None, config_path="", config_name="training")
def main(cfg: DictConfig):

    if cfg["random_state"]:
        torch.manual_seed(cfg["random_state"])

    mlflow.set_experiment("Discriminator")

    client = QdrantClient(
        url="http://qdrant:6333",
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=120
    )

    X_train, y_train = data_importation.fetch_training_data(
        client=client,
        collection_name=cfg["qdrant"]["collection_train"]
    )

    X_eval, y_eval = data_importation.fetch_training_data(
        client=client,
        collection_name=cfg["qdrant"]["collection_test"]
    )

    # ==============================
    #             Model
    # ==============================
    logger.info("Building the desired model...")

    model = model_factory.build_model(cfg)

    # ==============================
    #            Training
    # ==============================
    logger.info("Training...")

    model.fit(X_train, y_train, X_eval, y_eval)

    # ==============================
    #        MLFlow logging
    # ==============================
    logger.info("Logging to MLFlow...")

    # model
    model.save(name=cfg["model"]["name"])
    mlflow.log_params(params=model.get_params())

    # data
    mlflow.log_params(params=cfg["data"])

    # seed
    mlflow.log_param("random_state", cfg["random_state"])

    # metrics
    metrics = model.get_metrics()
    for k, v in metrics.items():
        print(k, v)
        # curves
        if isinstance(v, list):
            for step, loss in enumerate(v):
                mlflow.log_metric(k, loss, step=step)

        # scalars
        else:
            mlflow.log_metric(k, v)


if __name__ == "__main__":
    main()
