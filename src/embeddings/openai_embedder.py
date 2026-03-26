from openai import OpenAI
import numpy as np

from .embedding_interface import Embedder


class OpenAIEmbedder(Embedder):
    """
    Made for infering.
    """

    def __init__(self,
                 base_url: str,
                 model: str = None,
                 api_key: str = ""
                 ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model

    def embed(self, texts: np.ndarray) -> np.ndarray:

        embeddings = []

        for t in texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=t
            )
            embeddings.append(response.data[0].embedding)

        return np.array(embeddings)
