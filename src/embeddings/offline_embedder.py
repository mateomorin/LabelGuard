from .embedding_interface import Embedder


class OfflineEmbedder(Embedder):
    """
    Made for training (directly using an embedding dataset).
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed(self, texts):
        raise RuntimeError("Offline embedder used only for datasets")
