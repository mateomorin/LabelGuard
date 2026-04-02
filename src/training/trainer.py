import os
import logging
from collections import Counter

from dotenv import load_dotenv
import s3fs
import hydra
from omegaconf import DictConfig
import mlflow
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd

from src.datasets import data_importation, data_preprocessing, data_exportation
from src.models import model_factory


logger = logging.getLogger(__name__)
load_dotenv(override=True)


@hydra.main(version_base=None, config_path="../configs", config_name="training")
def main(cfg: DictConfig):
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"]
    )

    with mlflow.start_run():
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

        X_train, X_eval, y_train, y_eval, indices_train, indices_eval = data_preprocessing.create_train_test(
            df_real=df_real,
            df_synth=df_synth,
            train_size=cfg["data"]["train_size"],
            random_state=cfg["random_state"]
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

        model.fit(X_train, y_train)

        # ==============================
        #           Evaluation
        # ==============================
        logger.info("Evaluation:")
        y_train_pred = model.predict(X_train)
        y_eval_pred = model.predict(X_eval)

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

        # ==============================
        #        MLFlow logging
        # ==============================

        model.save(name=cfg["name"])
        mlflow.log_params(params=model.get_params())

        # ==============================
        #       Data Exportation
        # ==============================
        if cfg["export"]["save_results"]:
            data_exportation.export_data(
                fs=fs,
                path=cfg["export"]["train_path"],
                texts=pd.concat([df_real["label"], df_synth["label"]]).iloc[indices_train].to_list()
            )
            data_exportation.export_data(
                fs=fs,
                path=cfg["export"]["eval_path"],
                texts=pd.concat([df_real["label"], df_synth["label"]]).iloc[indices_eval].to_list()
            )
