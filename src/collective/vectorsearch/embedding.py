import numpy as np


class EmbeddingBase:
    """Base class for embedding."""

    meta_type = None

    def __init__(self, model, chunk_size=500, prefix_query=None):
        self.model = model
        self.chunk_size = chunk_size
        self.prefix_query = prefix_query

    def embed(self, text: str, query=False) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingBase):
    """Sentence Transformer Embedding."""

    meta_type = "SentenceTransformerEmbedding"

    def embed(self, text: str, query=False) -> np.ndarray:
        if query:
            text = self.prefix_query + text
        texts = [
            text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
        ]
        embeddings = self.model.encode(texts)
        return embeddings
