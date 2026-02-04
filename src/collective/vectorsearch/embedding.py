import numpy as np


class EmbeddingBase:
    """Base class for embedding."""

    meta_type = None

    def __init__(self, model, chunk_size=500, prefix_query=None, prefix_passage=None):
        self.model = model
        self.chunk_size = chunk_size
        self.prefix_query = prefix_query
        self.prefix_passage = prefix_passage

    def embed(self, text: str, query=False) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingBase):
    """Sentence Transformer Embedding."""

    meta_type = "SentenceTransformerEmbedding"

    def embed(self, text: str, query=False) -> np.ndarray:
        # Add appropriate prefix based on query type
        if query and self.prefix_query:
            text = self.prefix_query + text
        elif not query and self.prefix_passage:
            text = self.prefix_passage + text

        texts = [
            text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
        ]
        embeddings = self.model.encode(texts)
        return embeddings


class FastEmbedEmbedding(EmbeddingBase):
    """FastEmbed-based embedding using ONNX runtime for speed."""

    meta_type = "FastEmbedEmbedding"

    def __init__(self, model_name, chunk_size=500, prefix_query=None, prefix_passage=None):
        """
        Initialize FastEmbed embedding.

        Args:
            model_name: Model identifier for FastEmbed (e.g., 'intfloat/e5-base-v2')
            chunk_size: Maximum text length for chunking
            prefix_query: Prefix to add to query text
            prefix_passage: Prefix to add to passage/document text
        """
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "FastEmbed library not installed. "
                "Install with: pip install fastembed"
            )

        self.model = TextEmbedding(model_name=model_name)
        self.chunk_size = chunk_size
        self.prefix_query = prefix_query
        self.prefix_passage = prefix_passage

    def embed(self, text: str, query=False) -> np.ndarray:
        """
        Embed text using FastEmbed.

        Args:
            text: Text to embed
            query: If True, add query prefix; if False, add passage prefix

        Returns:
            numpy array of embeddings
        """
        # Add appropriate prefix based on query type
        if query and self.prefix_query:
            text = self.prefix_query + text
        elif not query and self.prefix_passage:
            text = self.prefix_passage + text

        # Chunk text
        texts = [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]

        # FastEmbed returns a generator, convert to list
        embeddings = list(self.model.embed(texts))

        # Convert to numpy array
        return np.array(embeddings)
