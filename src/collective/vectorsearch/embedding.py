import numpy as np


class EmbeddingBase:
    """Base class for embedding."""

    meta_type = None

    def __init__(self, model, chank_size=500, prefix_query=None):
        self.model = model
        self.chank_size = chank_size
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
            text[i : i + self.chank_size] for i in range(0, len(text), self.chank_size)
        ]
        embeddings = self.model.encode(texts)
        return embeddings
