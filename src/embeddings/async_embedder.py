import asyncio

from openai import AsyncOpenAI
import numpy as np

from .embedding_interface import Embedder


class AsyncEmbedder(Embedder):
    """
    Made for infering.
    """

    def __init__(self,
                 base_url: str,
                 model: str = None,
                 api_key: str = "",
                 max_concurrency: int = 15
                 ):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.max_concurrency = max_concurrency

    async def embed_async(self, texts: list[str]) -> list:

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def limited_call(text):
            async with semaphore:
                return await self.client.embeddings.create(
                        model=self.model,
                        input=text
                )

        tasks = [limited_call(t) for t in texts]
        return await asyncio.gather(*tasks)

    def embed(self, texts: list[str]) -> np.ndarray:

        embeddings = asyncio.run(self.embed_async(texts))

        return np.array(embeddings)
