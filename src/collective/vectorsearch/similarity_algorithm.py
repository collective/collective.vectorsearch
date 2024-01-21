import torch
from torch.nn.functional import cosine_similarity


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
    """Cosine similarity algorithm"""

    def query(self, vectors, query):
        """Return a similarity value for the given query"""
        if vectors.shape[0] < self.size:
            size = vectors.shape[0]
        else:
            size = self.size
        t_vectors = torch.tensor(vectors, dtype=torch.float32)
        t_query = torch.tensor(query, dtype=torch.float32)
        similarities = cosine_similarity(t_vectors, t_query)
        top10_values, top10_indices = torch.topk(similarities, size)
        return top10_indices.numpy(), top10_values.numpy()
