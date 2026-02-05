# -*- coding: utf-8 -*-
"""Similarity algorithms for vector search."""

import numpy as np

# Try to import torch for optional GPU acceleration
try:
    import torch
    from torch.nn.functional import cosine_similarity as torch_cosine_similarity
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def numpy_cosine_similarity(vectors, query):
    """Calculate cosine similarity using numpy.

    Args:
        vectors: numpy array of shape (n, d) where n is number of vectors
        query: numpy array of shape (d,) or (1, d)

    Returns:
        numpy array of similarities of shape (n,)
    """
    # Ensure query is 1D
    query = query.flatten()

    # Normalize vectors and query
    vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query)

    # Avoid division by zero
    vectors_norm = np.where(vectors_norm == 0, 1, vectors_norm)
    query_norm = query_norm if query_norm != 0 else 1

    # Calculate cosine similarity
    normalized_vectors = vectors / vectors_norm
    normalized_query = query / query_norm

    similarities = np.dot(normalized_vectors, normalized_query)
    return similarities


class SimilarityAlgorithmBase:
    """Base class for similarity algorithms"""

    def __init__(self, size=10):
        self.size = size

    def __call__(self, vectors, query):
        """Return a similarity value for the given query"""
        return self.query(vectors, query)

    def query(self, vectors, query):
        """Return a similarity value for the given query"""
        raise NotImplementedError


class CosineSimilarityAlgorithm(SimilarityAlgorithmBase):
    """Cosine similarity algorithm using numpy (default) or torch (if available)."""

    def __init__(self, size=10, use_gpu=False):
        """Initialize cosine similarity algorithm.

        Args:
            size: Number of top results to return
            use_gpu: If True and torch is available, use GPU acceleration
        """
        super().__init__(size)
        self.use_gpu = use_gpu and HAS_TORCH

    def query(self, vectors, query):
        """Return top similar vectors for the given query.

        Args:
            vectors: numpy array of shape (n, d)
            query: numpy array of shape (d,) or (1, d)

        Returns:
            Tuple of (indices, similarities) for top matches
        """
        size = min(self.size, vectors.shape[0])

        if self.use_gpu and HAS_TORCH:
            return self._query_torch(vectors, query, size)
        else:
            return self._query_numpy(vectors, query, size)

    def _query_numpy(self, vectors, query, size):
        """Query using numpy (CPU)."""
        similarities = numpy_cosine_similarity(vectors, query)

        # Get top-k indices
        if size >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Use argpartition for efficiency when k << n
            top_indices = np.argpartition(similarities, -size)[-size:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        top_values = similarities[top_indices]
        return top_indices, top_values

    def _query_torch(self, vectors, query, size):
        """Query using torch (GPU if available)."""
        t_vectors = torch.tensor(vectors, dtype=torch.float32)
        t_query = torch.tensor(query, dtype=torch.float32)

        # Use torch's cosine similarity
        similarities = torch_cosine_similarity(t_vectors, t_query)

        top_values, top_indices = torch.topk(similarities, size)
        return top_indices.numpy(), top_values.numpy()
