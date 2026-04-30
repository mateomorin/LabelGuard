import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from .model_interface import BaseModel


class LitMLP(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout_layers: list[float],
        lr: float,
        activation=nn.ReLU
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_dim = input_dim
        for h, p in zip(hidden_layers, dropout_layers):
            layers.append(nn.Linear(in_dim, h))
            # robustness to nn.ReLU or nn.ReLU()
            if isinstance(activation, type):
                layers.append(activation())
            else:
                layers.append(activation)
            layers.append(nn.Dropout(p))
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
        hidden_layers: list = [1024, 256],
        dropout_layers: list = [0.4, 0.2],
        lr=1e-3,
        activation=nn.ReLU,
        epochs: int = 10,
        batch_size: int = 32,
        patience: int = 5
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_layers = dropout_layers
        self.lr = lr
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.model = LitMLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout_layers=dropout_layers,
            lr=lr,
            activation=activation
        )
        self.trainer = None
        self.metrics = {}
        self.params = {
            "input_dim": input_dim,
            "hidden_layers": str(hidden_layers),
            "dropout_layers": str(dropout_layers),
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

        active_run = mlflow.active_run()

        if active_run:
            mlflow_logger = MLFlowLogger(
                experiment_name="LabelGuard",
                tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
                run_id=active_run.info.run_id
            )

        else:
            mlflow_logger = MLFlowLogger(
                experiment_name="LabelGuard",
                tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
            )

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
            logger=mlflow_logger,
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
    def _predict_internal(self, X):
        self.model.eval()
        X_t = torch.from_numpy(X).float() if not isinstance(X, torch.Tensor) else X.float()
        logits = self.model(X_t)
        return (torch.sigmoid(logits) > 0.5).long().numpy().squeeze()

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.from_numpy(X).float()
        logits = self.model(X_t)
        return torch.sigmoid(logits).long().numpy().squeeze()
