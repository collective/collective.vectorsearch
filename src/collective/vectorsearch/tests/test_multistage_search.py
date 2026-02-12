# -*- coding: utf-8 -*-
"""Tests for multi-stage vector search algorithms."""

import unittest
from unittest.mock import Mock, patch

import numpy as np


class TestBatchMinHammingDistance(unittest.TestCase):
    """Test batch_min_hamming_distance function."""

    def test_identical_hashes_distance_zero(self):
        from collective.vectorsearch.indexers import batch_min_hamming_distance

        query = (123456789, 987654321)
        doc_hashes = [(123456789, 987654321)]
        self.assertEqual(batch_min_hamming_distance(query, doc_hashes), 0)

    def test_empty_doc_hashes_returns_129(self):
        from collective.vectorsearch.indexers import batch_min_hamming_distance

        query = (123, 456)
        self.assertEqual(batch_min_hamming_distance(query, []), 129)
        self.assertEqual(batch_min_hamming_distance(query, None), 129)

    def test_all_bits_different(self):
        from collective.vectorsearch.indexers import batch_min_hamming_distance

        query = (0, 0)
        doc_hashes = [((1 << 64) - 1, (1 << 64) - 1)]
        self.assertEqual(batch_min_hamming_distance(query, doc_hashes), 128)

    def test_single_bit_difference(self):
        from collective.vectorsearch.indexers import batch_min_hamming_distance

        query = (0, 0)
        # Only bit 0 of high is set
        doc_hashes = [(1, 0)]
        self.assertEqual(batch_min_hamming_distance(query, doc_hashes), 1)

    def test_min_across_multiple_chunks(self):
        from collective.vectorsearch.indexers import batch_min_hamming_distance

        query = (100, 200)
        # Chunk 1: far away (many bits different)
        # Chunk 2: identical (distance 0)
        doc_hashes = [((1 << 64) - 1, (1 << 64) - 1), (100, 200)]
        self.assertEqual(batch_min_hamming_distance(query, doc_hashes), 0)

    def test_known_distance(self):
        from collective.vectorsearch.indexers import batch_min_hamming_distance

        # XOR of 0b1111 and 0b0000 = 0b1111 = 4 bits
        query = (0b1111, 0)
        doc_hashes = [(0, 0)]
        self.assertEqual(batch_min_hamming_distance(query, doc_hashes), 4)

    def test_uses_bit_count(self):
        """Verify .bit_count() is used (Python 3.10+)."""
        # This test ensures the method works correctly with large integers
        from collective.vectorsearch.indexers import batch_min_hamming_distance

        # Create a hash with exactly 32 bits set in high, 32 in low
        high_val = (1 << 32) - 1  # Lower 32 bits set
        low_val = (1 << 32) - 1
        query = (0, 0)
        doc_hashes = [(high_val, low_val)]
        self.assertEqual(batch_min_hamming_distance(query, doc_hashes), 64)


class TestBinaryHashRoundtrip(unittest.TestCase):
    """Test binary hash conversion roundtrip."""

    def test_roundtrip(self):
        from collective.vectorsearch.indexers import (
            binary_hash_to_integers,
            integers_to_binary_hash,
        )

        # Create a random binary hash
        rng = np.random.RandomState(42)
        original = rng.randint(0, 2, size=128).astype(np.uint8)

        high, low = binary_hash_to_integers(original)
        reconstructed = integers_to_binary_hash(high, low)

        np.testing.assert_array_equal(original, reconstructed)

    def test_all_zeros(self):
        from collective.vectorsearch.indexers import binary_hash_to_integers

        zeros = np.zeros(128, dtype=np.uint8)
        high, low = binary_hash_to_integers(zeros)
        self.assertEqual(high, 0)
        self.assertEqual(low, 0)

    def test_all_ones(self):
        from collective.vectorsearch.indexers import binary_hash_to_integers

        ones = np.ones(128, dtype=np.uint8)
        high, low = binary_hash_to_integers(ones)
        self.assertEqual(high, (1 << 64) - 1)
        self.assertEqual(low, (1 << 64) - 1)


class TestQueryDispatcher(unittest.TestCase):
    """Test query_index dispatcher logic."""

    def _make_index(self):
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")
        return index

    @patch("collective.vectorsearch.vector_index.api.portal.get_registry_record")
    @patch("collective.vectorsearch.vector_index.queryUtility")
    def test_exhaustive_cosine_default(self, mock_utility, mock_registry):
        """Test that exhaustive_cosine is used by default."""
        index = self._make_index()

        mock_registry.side_effect = Exception("no registry")

        # Mock embedding
        mock_embedding = Mock()
        mock_embedding.embed.return_value = np.array([[0.1, 0.2, 0.3]])
        index._v_embedding = mock_embedding

        # Mock similarity algorithm
        mock_algo = Mock()
        mock_algo.return_value = (np.array([0]), np.array([0.95]))
        index._v_similarity_algorithm = mock_algo

        # Mock _get_all_doc_vectors
        with patch.object(
            index,
            "_get_all_doc_vectors",
            return_value=(np.array([1], dtype=int), np.array([[0.1, 0.2, 0.3]])),
        ):
            record = Mock()
            record.keys = ["test query"]
            result = index.query_index(record)

        self.assertIsNotNone(result)
        self.assertIn(1, result)

    @patch("collective.vectorsearch.vector_index.api.portal.get_registry_record")
    @patch("collective.vectorsearch.vector_index.queryUtility")
    def test_2stage_fallback_without_itq(self, mock_utility, mock_registry):
        """Test 2-stage falls back to exhaustive when ITQ data is missing."""
        index = self._make_index()

        # Settings: request 2-stage
        def mock_reg(key, default=None):
            settings_map = {
                "collective.vectorsearch.approximation_algorithm": "itq_lsh_2stage",
                "collective.vectorsearch.embedding_model": "all-minilm-l6",
                "collective.vectorsearch.embedding_chunk_size": 500,
                "collective.vectorsearch.storage_backend": "btrees",
                "collective.vectorsearch.external_db_uri": "",
                "collective.vectorsearch.pivot_threshold": 200,
                "collective.vectorsearch.itq_candidates": 100,
            }
            return settings_map.get(key, default)

        mock_registry.side_effect = mock_reg

        mock_embedding = Mock()
        mock_embedding.embed.return_value = np.array([[0.1, 0.2, 0.3]])
        index._v_embedding = mock_embedding
        index._v_similarity_algorithm = Mock(
            return_value=(np.array([0]), np.array([0.95]))
        )
        # ITQ boundary NOT set → should fall back
        index._v_itq_boundary = None

        with patch.object(
            index,
            "_get_all_doc_vectors",
            return_value=(np.array([1], dtype=int), np.array([[0.1, 0.2, 0.3]])),
        ):
            record = Mock()
            record.keys = ["test query"]
            result = index.query_index(record)

        # Should still return results (via exhaustive fallback)
        self.assertIsNotNone(result)

    @patch("collective.vectorsearch.vector_index.api.portal.get_registry_record")
    @patch("collective.vectorsearch.vector_index.queryUtility")
    def test_3stage_fallback_to_2stage(self, mock_utility, mock_registry):
        """Test 3-stage falls back to 2-stage when pivot data is missing."""
        index = self._make_index()

        def mock_reg(key, default=None):
            settings_map = {
                "collective.vectorsearch.approximation_algorithm": "itq_lsh_3stage",
                "collective.vectorsearch.embedding_model": "all-minilm-l6",
                "collective.vectorsearch.embedding_chunk_size": 500,
                "collective.vectorsearch.storage_backend": "btrees",
                "collective.vectorsearch.external_db_uri": "",
                "collective.vectorsearch.pivot_threshold": 200,
                "collective.vectorsearch.itq_candidates": 100,
            }
            return settings_map.get(key, default)

        mock_registry.side_effect = mock_reg

        mock_embedding = Mock()
        mock_embedding.embed.return_value = np.array([[0.1, 0.2, 0.3]])
        index._v_embedding = mock_embedding
        index._v_similarity_algorithm = Mock(
            return_value=(np.array([0]), np.array([0.95]))
        )

        # ITQ available, but pivot NOT → should try 2-stage
        mock_itq = Mock()
        mock_itq.compute_hash.return_value = np.zeros(128, dtype=np.uint8)
        index._v_itq_boundary = mock_itq
        index._v_pivot_data = None  # No pivot data

        # Set up a docid with metadata
        index._docid_to_path[1] = "/plone/doc1"

        mock_catalog = Mock()
        mock_metadata = {"itq_hashes": ((0, 0),), "llm_vector": [[0.1, 0.2, 0.3]]}
        mock_catalog.getMetadataForRID.return_value = mock_metadata

        with patch(
            "collective.vectorsearch.vector_index.api.portal.get_tool",
            return_value=mock_catalog,
        ):
            record = Mock()
            record.keys = ["test query"]
            result = index.query_index(record)

        self.assertIsNotNone(result)


class TestCosineOnCandidates(unittest.TestCase):
    """Test _cosine_on_candidates helper."""

    def test_returns_empty_bucket_for_no_vectors(self):
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")
        index._v_similarity_algorithm = Mock()

        mock_catalog = Mock()
        mock_catalog.getMetadataForRID.return_value = None

        result = index._cosine_on_candidates(
            np.array([[0.1, 0.2]]), [1, 2, 3], mock_catalog
        )
        self.assertEqual(len(result), 0)

    def test_scores_candidates(self):
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        mock_algo = Mock()
        mock_algo.return_value = (np.array([0, 1]), np.array([0.95, 0.80]))
        index._v_similarity_algorithm = mock_algo
        # Prevent _ensure_initialized from running
        index._v_embedding = Mock()

        mock_catalog = Mock()
        mock_catalog.getMetadataForRID.side_effect = lambda docid: {
            "llm_vector": [[0.1, 0.2, 0.3]]
        }

        query_vectors = np.array([[0.1, 0.2, 0.3]])
        result = index._cosine_on_candidates(query_vectors, [10, 20], mock_catalog)

        self.assertEqual(len(result), 2)
        # Scores should be multiplied by 100_000_000
        self.assertEqual(result[10], int(0.95 * 100_000_000))
        self.assertEqual(result[20], int(0.80 * 100_000_000))


class TestPivotFilter(unittest.TestCase):
    """Test _pivot_filter method."""

    def test_returns_empty_when_no_catalog(self):
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        with patch(
            "collective.vectorsearch.vector_index.api.portal.get_tool",
            side_effect=Exception("no catalog"),
        ):
            result = index._pivot_filter([500] * 8, 200)
            self.assertEqual(len(result), 0)

    def test_returns_empty_when_no_pivot_indexes(self):
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        mock_catalog = Mock()
        mock_catalog.Indexes = {}

        with patch(
            "collective.vectorsearch.vector_index.api.portal.get_tool",
            return_value=mock_catalog,
        ):
            result = index._pivot_filter([500] * 8, 200)
            self.assertEqual(len(result), 0)

    def test_intersects_pivot_results(self):
        from BTrees.IIBTree import IISet

        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # Create mock indexes that return different docid sets
        def make_mock_index(docids):
            mock_idx = Mock()
            mock_idx.query_options = ("query", "range", "not", "operator")
            mock_idx.operators = ("and", "or")
            mock_idx.useOperator = "or"
            mock_idx.query_index.return_value = IISet(docids)
            return mock_idx

        mock_catalog = Mock()
        indexes = {}
        # pivot1: docs 1,2,3
        # pivot2: docs 2,3,4
        # pivot3-8: all docs
        indexes["pivot1"] = make_mock_index([1, 2, 3])
        indexes["pivot2"] = make_mock_index([2, 3, 4])
        for i in range(3, 9):
            indexes[f"pivot{i}"] = make_mock_index([1, 2, 3, 4])

        mock_catalog.Indexes = indexes

        with patch(
            "collective.vectorsearch.vector_index.api.portal.get_tool",
            return_value=mock_catalog,
        ):
            result = index._pivot_filter([500] * 8, 200)

        # Intersection of {1,2,3} and {2,3,4} = {2,3}
        self.assertEqual(set(result), {2, 3})

    def test_short_circuits_on_empty_pivot(self):
        from BTrees.IIBTree import IISet

        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        def make_mock_index(docids):
            mock_idx = Mock()
            mock_idx.query_options = ("query", "range", "not", "operator")
            mock_idx.operators = ("and", "or")
            mock_idx.useOperator = "or"
            mock_idx.query_index.return_value = IISet(docids)
            return mock_idx

        mock_catalog = Mock()
        indexes = {}
        indexes["pivot1"] = make_mock_index([])  # Empty result
        for i in range(2, 9):
            indexes[f"pivot{i}"] = make_mock_index([1, 2, 3])

        mock_catalog.Indexes = indexes

        with patch(
            "collective.vectorsearch.vector_index.api.portal.get_tool",
            return_value=mock_catalog,
        ):
            result = index._pivot_filter([500] * 8, 200)

        # Should short circuit and return empty
        self.assertEqual(len(result), 0)


class TestQuery2Stage(unittest.TestCase):
    """Test _query_itq_lsh_2stage method."""

    def test_ranks_by_hamming_distance(self):
        """Test that documents are ranked by Hamming distance, top-K selected."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # Setup ITQ boundary
        mock_itq = Mock()
        mock_itq.compute_hash.return_value = np.zeros(128, dtype=np.uint8)
        index._v_itq_boundary = mock_itq

        # Setup similarity algorithm and prevent _ensure_initialized
        mock_algo = Mock()
        mock_algo.return_value = (np.array([0, 1]), np.array([0.95, 0.80]))
        index._v_similarity_algorithm = mock_algo
        index._v_embedding = Mock()

        # Add documents
        index._docid_to_path[1] = "/plone/doc1"
        index._docid_to_path[2] = "/plone/doc2"
        index._docid_to_path[3] = "/plone/doc3"

        # Doc1: hash = (0,0) → distance 0 from query (0,0) → most similar
        # Doc2: hash = (0xFF, 0) → distance 8 → medium
        # Doc3: hash = (0xFF..FF, 0xFF..FF) → distance 128 → least similar
        mock_catalog = Mock()

        def mock_metadata(docid):
            if docid == 1:
                return {
                    "itq_hashes": ((0, 0),),
                    "llm_vector": [[0.1, 0.2, 0.3]],
                }
            elif docid == 2:
                return {
                    "itq_hashes": ((0xFF, 0),),
                    "llm_vector": [[0.2, 0.3, 0.4]],
                }
            elif docid == 3:
                return {
                    "itq_hashes": (((1 << 64) - 1, (1 << 64) - 1),),
                    "llm_vector": [[0.4, 0.5, 0.6]],
                }
            return None

        mock_catalog.getMetadataForRID.side_effect = mock_metadata

        with patch(
            "collective.vectorsearch.vector_index.api.portal.get_tool",
            return_value=mock_catalog,
        ):
            # Request top 2 candidates → doc3 (distance 128) excluded
            settings = {"itq_candidates": 2}
            query_vectors = np.array([[0.1, 0.2, 0.3]])
            passage_vectors = np.array([[0.1, 0.2, 0.3]])
            result = index._query_itq_lsh_2stage(
                query_vectors, passage_vectors, settings
            )

        # Doc1 and doc2 should be selected (closest by Hamming distance)
        self.assertIsNotNone(result)
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_all_docs_included_when_candidates_exceeds_total(self):
        """Test that all docs pass when itq_candidates >= total documents."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        mock_itq = Mock()
        mock_itq.compute_hash.return_value = np.zeros(128, dtype=np.uint8)
        index._v_itq_boundary = mock_itq

        mock_algo = Mock()
        mock_algo.return_value = (np.array([0]), np.array([0.95]))
        index._v_similarity_algorithm = mock_algo
        index._v_embedding = Mock()

        index._docid_to_path[1] = "/plone/doc1"

        mock_catalog = Mock()
        mock_catalog.getMetadataForRID.return_value = {
            "itq_hashes": (((1 << 64) - 1, (1 << 64) - 1),),
            "llm_vector": [[0.1, 0.2, 0.3]],
        }

        with patch(
            "collective.vectorsearch.vector_index.api.portal.get_tool",
            return_value=mock_catalog,
        ):
            # itq_candidates=100 but only 1 doc → all included
            settings = {"itq_candidates": 100}
            query_vectors = np.array([[0.1, 0.2, 0.3]])
            passage_vectors = np.array([[0.1, 0.2, 0.3]])
            result = index._query_itq_lsh_2stage(
                query_vectors, passage_vectors, settings
            )

        self.assertIsNotNone(result)
        self.assertIn(1, result)


if __name__ == "__main__":
    unittest.main()
