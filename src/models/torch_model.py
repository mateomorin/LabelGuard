from .base_model import BaseModel


class TorchModel(BaseModel):

    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, X, y):
        for epoch in range(10):
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict(self, X):
        self.model.eval()
        return self.model(X).argmax(dim=1)
