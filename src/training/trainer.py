import os
import logging
from collections import Counter

from dotenv import load_dotenv
import s3fs
import hydra
from omegaconf import DictConfig
import mlflow
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


from src.datasets import data_importation, data_preprocessing, data_exportation
from src.embeddings import offline_embedder
from src.models import model_factory


logger = logging.getLogger(__name__)
load_dotenv(override=True)


@hydra.main(version_base=None, config_path="../configs", config_name="training")
def main(cfg: DictConfig):
    # ==============================
    #             Data
    # ==============================
    logger.info("Creating training dataset...")

    df_real = data_importation.fetch_original_data(
        path=cfg["data"]["real_path"],
        n_samples=cfg["data"]["n_samples"],
        random_state=cfg["random_state"]
    )

    code_list = Counter(df_real["code"].to_list())

    df_synth = data_importation.select_synthetic_data(
        path=cfg["data"]["synth_path"],
        code_list=code_list,
        random_state=cfg["random_state"]
    )

    X_train, X_eval, y_train, y_eval = data_preprocessing.create_train_test(
        df_real=df_real,
        df_synth=df_synth,
        train_size=cfg["data"]["train_size"],
        random_state=cfg["random_state"]
    )

    # ==============================
    #             Model
    # ==============================
    logger.info("Building the desired model...")
    with mlflow.start_run():
        mlflow.autolog()

        model = model_factory.build_model(cfg)

        # ==============================
        #            Training
        # ==============================
        logger.info("Training...")

        model.fit(X_train, y_train)

        # ==============================
        #           Evaluation
        # ==============================
        logger.info("Evaluation:")
        y_train_pred = model.predict(X_train, y_train)
        y_eval_pred = model.predict(X_eval, y_eval)

        training_accuracy = accuracy_score(y_true=y_train, y_pred=y_train_pred)
        training_f1 = f1_score(y_true=y_train, y_pred=y_train_pred)
        training_cfm = confusion_matrix(y_true=y_train, y_pred=y_train_pred)

        logger.info("Training dataset:")
        logger.info(f"Accuracy: {training_accuracy}")
        logger.info(f"F1 score: {training_f1}")
        logger.info(f"Confusion Matrix: {training_cfm}")

        eval_accuracy = accuracy_score(y_true=y_eval, y_pred=y_eval_pred)
        eval_f1 = f1_score(y_true=y_eval, y_pred=y_eval_pred)
        eval_cfm = confusion_matrix(y_true=y_eval, y_pred=y_eval_pred)

        logger.info("Evaluation dataset:")
        logger.info(f"Accuracy: {eval_accuracy}")
        logger.info(f"F1 score: {eval_f1}")
        logger.info(f"Confusion Matrix: {eval_cfm}")

