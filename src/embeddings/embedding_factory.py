import os

from .basic_embedder import BasicEmbedder
from .async_embedder import AsyncEmbedder


def build_embedding(cfg):

    embed_type = cfg["embed"]["type"]
    model = cfg["embed"]["model"]

    if embed_type == "async":
        embedder = AsyncEmbedder(
            base_url=os.environ["EMBEDDING_API_BASE_URL"],
            model=model,
            api_key=os.environ["EMBEDDING_API_KEY"]
        )

        return embedder

    elif embed_type == "basic":
        embedder = BasicEmbedder(
            base_url=os.environ["EMBEDDING_API_BASE_URL"],
            model=model,
            api_key=os.environ["EMBEDDING_API_KEY"]
        )

        return embedder

    raise ValueError("Unknown embedding type")
