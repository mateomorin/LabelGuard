import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .model_interface import BaseModel


class MLP(nn.Module):
    """
    Dynamicaly configurable MLP with output_dim=1.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        activation=nn.ReLU
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchMLPClassifier(BaseModel):

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list = [64, 32],
        loss_fn=None,
        lr=1e-3,
        activation=nn.ReLU(),
        device=None,
        epochs: int = 10,
        batch_size: int = 32
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # model
        self.model = MLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            activation=activation or nn.ReLU()
        ).to(self.device)

        # loss
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()

        # optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=lr,
        )

        self.params = {
            "input_dim": input_dim,
            "hidden_layers": str(hidden_layers),
            "loss_fn": str(loss_fn),
            "optimizer_cls": "Adam",
            "activation": str(activation),
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size
        }

        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):

        self.model.train()

        # =========================
        #          Device
        # =========================
        X_train = torch.from_numpy(X_train).to(self.device)
        y_train = torch.from_numpy(y_train).float().unsqueeze(1).to(self.device)

        if X_eval is not None:
            X_eval = torch.from_numpy(X_eval).to(self.device)
            y_eval = torch.from_numpy(y_eval).float().unsqueeze(1).to(self.device)

        # =========================
        #     Dataset / Loader
        # =========================
        train_dataset = TensorDataset(X_train, y_train)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # =========================
        #     Metrics container
        # =========================
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
        }

        # =========================
        # Training loop
        # =========================
        for epoch in range(self.epochs):

            self.model.train()
            epoch_loss = 0.0

            for xb, yb in train_loader:
                self.optimizer.zero_grad()

                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(train_loader.dataset)
            self.metrics["train_loss"].append(epoch_loss)

            # -------------------------
            #        Eval loss
            # -------------------------
            eval_loss = None

            if X_eval is not None:
                self.model.eval()
                with torch.no_grad():
                    logits_eval = self.model(X_eval)
                    eval_loss = self.loss_fn(logits_eval, y_eval).item()

                self.metrics["eval_loss"].append(eval_loss)

            print(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"- train_loss: {epoch_loss:.4f}"
                + (f" - eval_loss: {eval_loss:.4f}" if eval_loss else "")
            )

        # =========================
        # Final metrics computation
        # =========================
        self.model.eval()

        def compute_metrics(X, y):
            with torch.no_grad():
                logits = self.model(X)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()

            y_true = y.cpu().numpy().ravel()
            y_pred = preds.cpu().numpy().ravel()

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cfm = str(confusion_matrix(y_true, y_pred))

            return acc, f1, cfm

        # ---- train metrics ----
        train_acc, train_f1, train_cfm = compute_metrics(X_train, y_train)
        t_n, f_p, f_n, t_p = train_cfm.ravel()

        self.metrics.update({
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "train_t_n": t_n,
            "train_f_p": f_p,
            "train_t_p": t_p,
            "train_f_n": f_n,
        })

        if X_eval is not None:
            eval_acc, eval_f1, eval_cfm = compute_metrics(X_eval, y_eval)

            t_n, f_p, f_n, t_p = eval_cfm.ravel()
            self.metrics.update({
                "eval_accuracy": eval_acc,
                "eval_f1": eval_f1,
                "eval_t_n": t_n,
                "eval_f_p": f_p,
                "eval_t_p": t_p,
                "eval_f_n": f_n,
            })

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()

        X = torch.from_numpy(X).to(self.device)
        logits = self.model(X)

        probs = torch.sigmoid(logits)
        return (probs > 0.5).long().squeeze(1)

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()

        X = torch.from_numpy(X).to(self.device)
        logits = self.model(X)

        probs = torch.sigmoid(logits)
        return torch.cat([probs], dim=1)

    def get_params(self):
        return self.params

    def get_metrics(self):
        return self.metrics

    def save(self, artifact_path: str = "model"):
        """
        Upload to MLFlow.
        """

        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path=artifact_path,
        )

        # sauver aussi la config nécessaire au reload
        mlflow.log_params({
            "input_dim": self.model.network[0].in_features,
            "output_dim": self.model.network[-1].out_features,
        })

    # ======================
    # LOAD
    # ======================
    @classmethod
    def load(cls, model_uri: str, device=None):
        """
        Download from MLFlow.
        """

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = mlflow.pytorch.load_model(model_uri)

        obj = cls.__new__(cls)  # bypass __init__
        obj.model = model.to(device)
        obj.device = device
        obj.model.eval()

        # placeholders (pas nécessaires pour inference)
        obj.optimizer = None
        obj.loss_fn = None

        return obj
