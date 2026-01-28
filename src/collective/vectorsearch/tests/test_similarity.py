# -*- coding: utf-8 -*-
"""Similarity algorithm tests for this package."""
import unittest
import numpy as np


class TestSimilarityAlgorithmBase(unittest.TestCase):
    """Test the SimilarityAlgorithmBase class."""

    def test_default_size(self):
        """Test that default size is 10."""
        from collective.vectorsearch.similarity_algorithm import (
            SimilarityAlgorithmBase
        )

        algorithm = SimilarityAlgorithmBase()
        self.assertEqual(algorithm.size, 10)

    def test_custom_size(self):
        """Test setting custom size."""
        from collective.vectorsearch.similarity_algorithm import (
            SimilarityAlgorithmBase
        )

        algorithm = SimilarityAlgorithmBase(size=20)
        self.assertEqual(algorithm.size, 20)

    def test_call_method(self):
        """Test that __call__ delegates to query method."""
        from collective.vectorsearch.similarity_algorithm import (
            SimilarityAlgorithmBase
        )

        algorithm = SimilarityAlgorithmBase()
        vectors = np.array([[1, 2, 3]])
        query = np.array([[1, 2, 3]])

        # Should raise NotImplementedError from query method
        with self.assertRaises(NotImplementedError):
            algorithm(vectors, query)

    def test_query_not_implemented(self):
        """Test that query method raises NotImplementedError."""
        from collective.vectorsearch.similarity_algorithm import (
            SimilarityAlgorithmBase
        )

        algorithm = SimilarityAlgorithmBase()
        vectors = np.array([[1, 2, 3]])
        query = np.array([[1, 2, 3]])

        with self.assertRaises(NotImplementedError):
            algorithm.query(vectors, query)


class TestCosineSimilarityAlgorithm(unittest.TestCase):
    """Test the CosineSimilarityAlgorithm class."""

    def test_identical_vectors(self):
        """Test similarity between identical vectors (should be 1.0)."""
        from collective.vectorsearch.similarity_algorithm import (
            CosineSimilarityAlgorithm
        )

        algorithm = CosineSimilarityAlgorithm(size=1)
        vectors = np.array([[1.0, 2.0, 3.0]])
        query = np.array([[1.0, 2.0, 3.0]])

        indices, scores = algorithm.query(vectors, query)

        # Should return index 0 with score close to 1.0
        self.assertEqual(len(indices), 1)
        self.assertEqual(indices[0], 0)
        self.assertAlmostEqual(scores[0], 1.0, places=5)

    def test_orthogonal_vectors(self):
        """Test similarity between orthogonal vectors (should be 0.0)."""
        from collective.vectorsearch.similarity_algorithm import (
            CosineSimilarityAlgorithm
        )

        algorithm = CosineSimilarityAlgorithm(size=1)
        vectors = np.array([[1.0, 0.0, 0.0]])
        query = np.array([[0.0, 1.0, 0.0]])

        indices, scores = algorithm.query(vectors, query)

        # Should return index 0 with score close to 0.0
        self.assertEqual(len(indices), 1)
        self.assertEqual(indices[0], 0)
        self.assertAlmostEqual(scores[0], 0.0, places=5)

    def test_top_k_selection(self):
        """Test that top-k vectors are selected correctly."""
        from collective.vectorsearch.similarity_algorithm import (
            CosineSimilarityAlgorithm
        )

        algorithm = CosineSimilarityAlgorithm(size=3)

        # Create vectors with known similarities to query
        vectors = np.array([
            [1.0, 0.0, 0.0],  # Similar to query
            [0.0, 1.0, 0.0],  # Orthogonal to query
            [1.0, 1.0, 0.0],  # Somewhat similar
            [2.0, 0.0, 0.0],  # Very similar to query
        ])
        query = np.array([[1.0, 0.0, 0.0]])

        indices, scores = algorithm.query(vectors, query)

        # Should return 3 results
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(scores), 3)

        # First result should be index 3 (most similar)
        # or index 0 (also very similar, both have score 1.0)
        self.assertIn(indices[0], [0, 3])

        # Scores should be in descending order
        self.assertTrue(scores[0] >= scores[1] >= scores[2])

    def test_fewer_vectors_than_size(self):
        """Test behavior when there are fewer vectors than requested size."""
        from collective.vectorsearch.similarity_algorithm import (
            CosineSimilarityAlgorithm
        )

        algorithm = CosineSimilarityAlgorithm(size=10)

        # Only 3 vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        query = np.array([[1.0, 0.0, 0.0]])

        indices, scores = algorithm.query(vectors, query)

        # Should return only 3 results
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(scores), 3)

    def test_return_types(self):
        """Test that indices and scores are numpy arrays."""
        from collective.vectorsearch.similarity_algorithm import (
            CosineSimilarityAlgorithm
        )

        algorithm = CosineSimilarityAlgorithm(size=2)
        vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        query = np.array([[1.0, 2.0, 3.0]])

        indices, scores = algorithm.query(vectors, query)

        self.assertIsInstance(indices, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)

    def test_negative_similarity(self):
        """Test that opposite direction vectors have negative similarity."""
        from collective.vectorsearch.similarity_algorithm import (
            CosineSimilarityAlgorithm
        )

        algorithm = CosineSimilarityAlgorithm(size=1)
        vectors = np.array([[-1.0, -1.0, -1.0]])
        query = np.array([[1.0, 1.0, 1.0]])

        indices, scores = algorithm.query(vectors, query)

        # Opposite vectors should have similarity close to -1.0
        self.assertEqual(len(indices), 1)
        self.assertLess(scores[0], 0.0)
        self.assertAlmostEqual(scores[0], -1.0, places=5)
