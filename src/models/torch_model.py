import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
import mlflow
import mlflow.pytorch

from .model_interface import BaseModel


class LitMLP(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        lr: float,
        activation=nn.ReLU
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            # robustness to nn.ReLU or nn.ReLU()
            if isinstance(activation, type):
                print(f"Instantiating {str(activation)} to {str(activation())}")
                layers.append(activation())
            else:
                print(f"Giving {str(activation)} directly")
                layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_acc(torch.sigmoid(logits), y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.sigmoid(logits)
        self.val_acc(preds, y)
        self.val_f1(preds, y)

        # used for early stopping
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class TorchMLPClassifier(BaseModel):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list = [64, 32],
        lr=1e-3,
        activation=nn.ReLU,
        epochs: int = 10,
        batch_size: int = 32,
        patience: int = 5   # Early stopping
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.model = LitMLP(input_dim, hidden_layers, lr, activation)
        self.trainer = None
        self.metrics = {}

        self.params = {
            "input_dim": input_dim,
            "hidden_layers": str(hidden_layers),
            "loss_fn": "BCEWithLogitsLoss",
            "optimizer_cls": "Adam",
            "activation": str(activation),
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size
        }

    def _prepare_dataloader(self, X, y, shuffle=False):
        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float().unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        train_loader = self._prepare_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._prepare_dataloader(X_eval, y_eval) if X_eval is not None else None

        # Early stopping
        callbacks = []
        if val_loader:
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                mode="min",
                verbose=True
            )
            callbacks.append(early_stop)

        # Training
        self.trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks,
            accelerator="auto",
            devices=1,
            enable_checkpointing=True
        )

        self.trainer.fit(self.model, train_loader, val_loader)

        self.metrics = {k: v.item() if torch.is_tensor(v) else v
                        for k, v in self.trainer.logged_metrics.items()}

    def get_params(self):
        return self.params

    def get_metrics(self):
        return self.metrics

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        X_t = torch.from_numpy(X).float()
        logits = self.model(X_t)
        return (torch.sigmoid(logits) > 0.5).long().numpy().squeeze()

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.from_numpy(X).float()
        logits = self.model(X_t)
        return torch.sigmoid(logits).long().numpy().squeeze()

    def save(self, name: str = "model"):
        # Pour éviter les soucis de configurations trop lourdes
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"
        os.environ["MLFLOW_CLIENT_HTTP_TIMEOUT"] = "600"
        os.environ["MLFLOW_DISABLE_ENV_MANAGER_CONDA"] = "True"
        mlflow.pytorch.log_model(self.model, name)
        mlflow.log_params({
            "hidden_layers": self.hidden_layers,
            "lr": self.lr,
            "batch_size": self.batch_size
        })

    @classmethod
    def load(cls, path: str):
        """
        Télécharge le modèle depuis MLflow et reconstruit l'objet TorchMLPClassifier.
        """

        loaded_lit_model = mlflow.pytorch.load_model(path)
        obj = cls.__new__(cls)
        hparams = loaded_lit_model.hparams

        obj.input_dim = hparams.input_dim
        obj.hidden_layers = hparams.hidden_layers
        obj.lr = hparams.lr
        obj.activation = hparams.activation

        obj.epochs = 10
        obj.batch_size = 32
        obj.patience = 5

        obj.model = loaded_lit_model
        obj.trainer = pl.Trainer(accelerator="auto", devices=1)
        obj.metrics = {}

        return obj
