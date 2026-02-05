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
