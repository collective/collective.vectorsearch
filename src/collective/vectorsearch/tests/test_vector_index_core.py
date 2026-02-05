# -*- coding: utf-8 -*-
"""Core VectorIndex functionality tests."""

import unittest
from unittest.mock import Mock, patch

import numpy as np


def create_mock_provider():
    """Create a mock model provider for testing."""
    mock_embedding = Mock()
    mock_embedding.chunk_size = 500

    mock_provider = Mock()
    mock_provider.get_embedding_instance.return_value = mock_embedding
    mock_provider.query_prefix = None
    mock_provider.passage_prefix = None
    return mock_provider


class TestVectorIndexCore(unittest.TestCase):
    """Test VectorIndex core functionality without Plone dependencies."""

    def test_index_initialization_with_defaults(self):
        """Test that VectorIndex initializes with default settings."""
        from collective.vectorsearch.vector_index import VectorIndex

        mock_provider = create_mock_provider()

        with patch(
            "collective.vectorsearch.vector_index.queryUtility",
            return_value=mock_provider,
        ):
            index = VectorIndex("test_index")

            self.assertEqual(index.id, "test_index")
            self.assertEqual(index.indexed_attrs, ["test_index"])
            self.assertIsNotNone(index.embedding)
            self.assertIsNotNone(index.similarity_algorithm)

    def test_index_initialization_with_indexed_attrs(self):
        """Test initialization with indexed_attrs in extra parameter."""
        from collective.vectorsearch.vector_index import VectorIndex

        extra = {"indexed_attrs": "title,description"}
        index = VectorIndex("test_index", extra=extra)

        self.assertEqual(index.indexed_attrs, ["title", "description"])

    def test_index_initialization_with_comma_separated_attrs(self):
        """Test parsing of comma-separated indexed_attrs."""
        from collective.vectorsearch.vector_index import VectorIndex

        # Test with spaces around commas
        extra = {"indexed_attrs": "title , description , text "}
        index = VectorIndex("test_index", extra=extra)

        self.assertEqual(index.indexed_attrs, ["title", "description", "text"])

    def test_get_index_source_names(self):
        """Test getIndexSourceNames returns indexed_attrs."""
        from collective.vectorsearch.vector_index import VectorIndex

        extra = {"indexed_attrs": "title,description"}
        index = VectorIndex("test_index", extra=extra)

        names = index.getIndexSourceNames()
        self.assertEqual(names, ["title", "description"])

    def test_get_index_source_names_default(self):
        """Test getIndexSourceNames returns index id when no indexed_attrs."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        names = index.getIndexSourceNames()
        self.assertEqual(names, ["test_index"])

    def test_get_index_query_names(self):
        """Test getIndexQueryNames returns index id."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("my_vector_index")

        names = index.getIndexQueryNames()
        self.assertEqual(names, ("my_vector_index",))

    def test_get_index_type(self):
        """Test getIndexType returns VectorIndex."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        index_type = index.getIndexType()
        self.assertEqual(index_type, "VectorIndex")

    def test_unique_values_returns_empty_tuple(self):
        """Test uniqueValues returns empty tuple."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        result = index.uniqueValues()
        self.assertEqual(result, ())

    def test_num_objects_initial(self):
        """Test numObjects returns 0 for new index."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        count = index.numObjects()
        self.assertEqual(count, 0)

    def test_index_size_initial(self):
        """Test indexSize returns 0 for new index."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        size = index.indexSize()
        self.assertEqual(size, 0)

    def test_clear_resets_index(self):
        """Test clear method resets the index."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # Add some dummy data
        index._docvectors[1] = np.array([[1.0, 2.0, 3.0]])
        index.length.change(1)
        index.document_count.change(1)

        # Clear the index
        index.clear()

        # Verify it's reset
        self.assertEqual(index.numObjects(), 0)
        self.assertEqual(index.indexSize(), 0)
        self.assertEqual(len(index._docvectors), 0)

    def test_get_settings_with_registry_error(self):
        """Test _get_settings falls back to defaults on registry error."""
        from collective.vectorsearch.vector_index import VectorIndex

        with patch(
            "collective.vectorsearch.vector_index.api.portal.get_registry_record"
        ) as mock_registry:
            # Simulate registry error
            mock_registry.side_effect = Exception("Registry not available")

            index = VectorIndex("test_index")

            settings = index._get_settings()

            # Should return default values
            self.assertEqual(settings["embedding_model"], "all-minilm-l6")
            self.assertEqual(settings["embedding_chunk_size"], 500)
            self.assertEqual(settings["approximation_algorithm"], "exhaustive_cosine")


class TestVectorIndexITQPivot(unittest.TestCase):
    """Test ITQ hash and pivot distance functionality."""

    def test_itq_hashes_btree_initialized(self):
        """Test that _itq_hashes BTree is initialized."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        self.assertTrue(hasattr(index, "_itq_hashes"))
        self.assertEqual(len(index._itq_hashes), 0)

    def test_pivot_distances_btree_initialized(self):
        """Test that _pivot_distances BTree is initialized."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        self.assertTrue(hasattr(index, "_pivot_distances"))
        self.assertEqual(len(index._pivot_distances), 0)

    def test_get_itq_hash_returns_none_for_missing(self):
        """Test getITQHash returns None for non-existent document."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        result = index.getITQHash(999)
        self.assertIsNone(result)

    def test_get_pivot_distances_returns_none_for_missing(self):
        """Test getPivotDistances returns None for non-existent document."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        result = index.getPivotDistances(999)
        self.assertIsNone(result)

    def test_get_pivot_distance_returns_none_for_missing(self):
        """Test getPivotDistance returns None for non-existent document."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        result = index.getPivotDistance(999, 0)
        self.assertIsNone(result)

    def test_get_pivot_distance_returns_none_for_invalid_index(self):
        """Test getPivotDistance returns None for invalid pivot index."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # Manually add pivot distances (new format: list of tuples, one per chunk)
        index._pivot_distances[1] = ((100, 200, 300, 400, 500, 600, 700, 800),)

        # Invalid index should return None
        result = index.getPivotDistance(1, 10)
        self.assertIsNone(result)

        result = index.getPivotDistance(1, -1)
        self.assertIsNone(result)

    def test_get_pivot_distance_returns_correct_value(self):
        """Test getPivotDistance returns correct value."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # Manually add pivot distances (new format: list of tuples, one per chunk)
        index._pivot_distances[1] = ((100, 200, 300, 400, 500, 600, 700, 800),)

        # Check each pivot (returns first chunk's value)
        for i in range(8):
            result = index.getPivotDistance(1, i)
            self.assertEqual(result, (i + 1) * 100)

    def test_get_pivot_distances_for_index(self):
        """Test getPivotDistancesForIndex returns all chunk distances for a pivot."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # Add pivot distances for document with 3 chunks
        index._pivot_distances[1] = (
            (100, 200, 300, 400, 500, 600, 700, 800),  # chunk 0
            (110, 210, 310, 410, 510, 610, 710, 810),  # chunk 1
            (120, 220, 320, 420, 520, 620, 720, 820),  # chunk 2
        )

        # Check pivot 0 (first pivot) returns all chunk distances
        result = index.getPivotDistancesForIndex(1, 0)
        self.assertEqual(result, (100, 110, 120))

        # Check pivot 3 (fourth pivot)
        result = index.getPivotDistancesForIndex(1, 3)
        self.assertEqual(result, (400, 410, 420))

    def test_get_itq_hashes_multiple_chunks(self):
        """Test getITQHashes returns all chunk hashes."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # Add ITQ hashes for document with 2 chunks
        index._itq_hashes[1] = ((123, 456), (789, 101112))

        result = index.getITQHashes(1)
        self.assertEqual(result, ((123, 456), (789, 101112)))

        # Legacy getITQHash returns first chunk only
        result = index.getITQHash(1)
        self.assertEqual(result, (123, 456))

    def test_clear_also_clears_itq_pivot_data(self):
        """Test clear method also clears ITQ hashes and pivot distances."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # Add some dummy data (new format: tuple of tuples)
        index._docvectors[1] = np.array([[1.0, 2.0, 3.0]])
        index._itq_hashes[1] = ((123, 456),)
        index._pivot_distances[1] = ((100, 200, 300, 400, 500, 600, 700, 800),)
        index.length.change(1)
        index.document_count.change(1)

        # Clear the index
        index.clear()

        # Verify all data is cleared
        self.assertEqual(len(index._docvectors), 0)
        self.assertEqual(len(index._itq_hashes), 0)
        self.assertEqual(len(index._pivot_distances), 0)

    def test_binary_hash_to_integers_conversion(self):
        """Test _binary_hash_to_integers method."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # Create a known binary hash
        binary_hash = np.zeros(128, dtype=np.uint8)
        binary_hash[0] = 1  # First bit of high
        binary_hash[64] = 1  # First bit of low

        high, low = index._binary_hash_to_integers(binary_hash)

        # First bit set means value should be 2^63
        self.assertEqual(high, 1 << 63)
        self.assertEqual(low, 1 << 63)

    def test_binary_hash_to_integers_all_ones(self):
        """Test _binary_hash_to_integers with all ones."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # All ones
        binary_hash = np.ones(128, dtype=np.uint8)

        high, low = index._binary_hash_to_integers(binary_hash)

        # All bits set
        expected = (1 << 64) - 1  # 2^64 - 1
        self.assertEqual(high, expected)
        self.assertEqual(low, expected)

    def test_binary_hash_to_integers_invalid_length(self):
        """Test _binary_hash_to_integers with invalid length."""
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index")

        # Wrong length
        binary_hash = np.zeros(64, dtype=np.uint8)

        high, low = index._binary_hash_to_integers(binary_hash)

        self.assertIsNone(high)
        self.assertIsNone(low)

    def test_is_model_consistent_empty_index(self):
        """Test isModelConsistent returns True for empty index."""
        from collective.vectorsearch.vector_index import VectorIndex

        with patch(
            "collective.vectorsearch.vector_index.api.portal.get_registry_record"
        ) as mock_registry:
            mock_registry.return_value = "all-minilm-l6"

            index = VectorIndex("test_index")

            # Empty index should be consistent
            self.assertTrue(index.isModelConsistent())

    def test_get_itq_pivot_stats(self):
        """Test getITQPivotStats returns correct statistics."""
        from collective.vectorsearch.vector_index import VectorIndex

        mock_provider = Mock()
        mock_provider.get_embedding_instance.return_value = Mock()
        mock_provider.get_itq_boundary.return_value = Mock()
        mock_provider.get_pivot_data.return_value = Mock()
        mock_provider.query_prefix = None
        mock_provider.passage_prefix = None

        with patch(
            "collective.vectorsearch.vector_index.queryUtility",
            return_value=mock_provider,
        ):
            index = VectorIndex("test_index")

            # Add some data
            index._docvectors[1] = np.array([[1.0, 2.0, 3.0]])
            index._itq_hashes[1] = (123, 456)
            index._pivot_distances[1] = (100, 200, 300, 400, 500, 600, 700, 800)
            index.length.change(1)
            index.document_count.change(1)

            stats = index.getITQPivotStats()

            self.assertEqual(stats["documents"], 1)
            self.assertEqual(stats["vectors"], 1)
            self.assertEqual(stats["itq_hashes"], 1)
            self.assertEqual(stats["pivot_distances"], 1)
            self.assertTrue(stats["itq_data_available"])
            self.assertTrue(stats["pivot_data_available"])


class TestIndexerUtilities(unittest.TestCase):
    """Test indexer utility functions."""

    def test_binary_hash_to_integers(self):
        """Test binary_hash_to_integers conversion."""
        from collective.vectorsearch.indexers import binary_hash_to_integers

        # Create a known binary hash
        binary_hash = np.zeros(128, dtype=np.uint8)
        binary_hash[0] = 1  # First bit of high

        high, low = binary_hash_to_integers(binary_hash)

        self.assertEqual(high, 1 << 63)
        self.assertEqual(low, 0)

    def test_integers_to_binary_hash(self):
        """Test integers_to_binary_hash conversion."""
        from collective.vectorsearch.indexers import integers_to_binary_hash

        high = 1 << 63
        low = 1 << 63

        binary_hash = integers_to_binary_hash(high, low)

        self.assertEqual(binary_hash[0], 1)
        self.assertEqual(binary_hash[64], 1)
        self.assertEqual(np.sum(binary_hash), 2)

    def test_compute_hamming_distance(self):
        """Test compute_hamming_distance function."""
        from collective.vectorsearch.indexers import compute_hamming_distance

        # Same hashes should have distance 0
        distance = compute_hamming_distance(123, 456, 123, 456)
        self.assertEqual(distance, 0)

        # All bits different
        high1 = 0
        low1 = 0
        high2 = (1 << 64) - 1
        low2 = (1 << 64) - 1

        distance = compute_hamming_distance(high1, low1, high2, low2)
        self.assertEqual(distance, 128)

    def test_distance_to_index_value(self):
        """Test distance_to_index_value conversion."""
        from collective.vectorsearch.indexers import distance_to_index_value

        # 0.5 should become 500 with default scale
        result = distance_to_index_value(0.5)
        self.assertEqual(result, 500)

        # 1.234 should become 1234
        result = distance_to_index_value(1.234)
        self.assertEqual(result, 1234)

    def test_index_value_to_distance(self):
        """Test index_value_to_distance conversion."""
        from collective.vectorsearch.indexers import index_value_to_distance

        # 500 should become 0.5
        result = index_value_to_distance(500)
        self.assertEqual(result, 0.5)

    def test_get_pivot_range_for_threshold(self):
        """Test get_pivot_range_for_threshold function."""
        from collective.vectorsearch.indexers import get_pivot_range_for_threshold

        min_val, max_val = get_pivot_range_for_threshold(500, 200)

        self.assertEqual(min_val, 300)
        self.assertEqual(max_val, 700)

        # Edge case: min should not go below 0
        min_val, max_val = get_pivot_range_for_threshold(100, 200)

        self.assertEqual(min_val, 0)
        self.assertEqual(max_val, 300)
