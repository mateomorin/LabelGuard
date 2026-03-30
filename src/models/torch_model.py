import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch

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
            layers.append(activation())
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchMLPClassifier(BaseModel):

    def __init__(
            self,
            input_dim: int,
            hidden_layers=(64, 32),
            loss_fn=None,
            optimizer_cls=torch.optim.Adam,
            lr=1e-3,
            device=None,
            ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # modèle configurable
        self.model = MLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)

        # loss modifiable
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()

        # optimizer modifiable
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=lr,
        )

    # ======================
    # TRAIN
    # ======================

    def fit(self, X, y, epochs=10, batch_size=32):
        self.model.train()

        # ---- device ----
        X = X.to(self.device)
        y = y.float().to(self.device)

        # ---- dataset ----
        dataset = TensorDataset(X, y)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # ---- training loop ----
        for epoch in range(epochs):
            epoch_loss = 0.0

            for xb, yb in loader:
                self.optimizer.zero_grad()

                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(loader.dataset)
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")

    # ======================
    # PREDICT
    # ======================
    @torch.no_grad()
    def predict(self, X):
        self.model.eval()

        X = X.to(self.device)
        logits = self.model(X)

        probs = torch.sigmoid(logits)
        return (probs > 0.5).long().squeeze(1)

    # ======================
    # PREDICT PROBA
    # ======================
    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()

        X = X.to(self.device)
        logits = self.model(X)

        probs = torch.sigmoid(logits)
        return torch.cat([probs], dim=1)

    # ======================
    # SAVE
    # ======================
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
