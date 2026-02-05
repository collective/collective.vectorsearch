# -*- coding: utf-8 -*-
"""Registry settings tests for this package."""

import unittest

from plone import api

from collective.vectorsearch.testing import (
    COLLECTIVE_VECTORSEARCH_INTEGRATION_TESTING,
)


class TestRegistrySettings(unittest.TestCase):
    """Test that registry settings are properly configured."""

    layer = COLLECTIVE_VECTORSEARCH_INTEGRATION_TESTING

    def setUp(self):
        """Custom shared utility setup for tests."""
        self.portal = self.layer["portal"]

    def test_registry_records_exist(self):
        """Test that all registry records are created on install."""
        # Test embedding_model
        value = api.portal.get_registry_record(
            "collective.vectorsearch.embedding_model"
        )
        self.assertIsNotNone(value)

        # Test embedding_chunk_size
        value = api.portal.get_registry_record(
            "collective.vectorsearch.embedding_chunk_size"
        )
        self.assertIsNotNone(value)

        # Test storage_backend
        value = api.portal.get_registry_record(
            "collective.vectorsearch.storage_backend"
        )
        self.assertIsNotNone(value)

        # Test approximation_algorithm
        value = api.portal.get_registry_record(
            "collective.vectorsearch.approximation_algorithm"
        )
        self.assertIsNotNone(value)

    def test_default_model(self):
        """Test default embedding model."""
        value = api.portal.get_registry_record(
            "collective.vectorsearch.embedding_model"
        )
        self.assertEqual(value, "all-minilm-l6")

    def test_default_chunk_size(self):
        """Test default chunk size."""
        value = api.portal.get_registry_record(
            "collective.vectorsearch.embedding_chunk_size"
        )
        self.assertEqual(value, 500)

    def test_default_storage_backend(self):
        """Test default storage backend."""
        value = api.portal.get_registry_record(
            "collective.vectorsearch.storage_backend"
        )
        self.assertEqual(value, "btrees")

    def test_default_approximation_algorithm(self):
        """Test default approximation algorithm."""
        value = api.portal.get_registry_record(
            "collective.vectorsearch.approximation_algorithm"
        )
        self.assertEqual(value, "exhaustive_cosine")

    def test_vector_index_reads_from_registry(self):
        """Test that VectorIndex uses registry settings."""
        from collective.vectorsearch.vector_index import VectorIndex

        # Create an index instance
        index = VectorIndex("test_index")

        # Verify it uses the registry settings
        # The embedding should have the default chunk size
        self.assertEqual(index.embedding.chunk_size, 500)

        # The similarity algorithm should be CosineSimilarityAlgorithm
        from collective.vectorsearch.similarity_algorithm import (
            CosineSimilarityAlgorithm,
        )

        self.assertIsInstance(index.similarity_algorithm, CosineSimilarityAlgorithm)

    def test_settings_can_be_changed(self):
        """Test that registry settings can be modified."""
        # Change a setting
        api.portal.set_registry_record(
            "collective.vectorsearch.embedding_chunk_size", 1000
        )

        # Verify the change
        value = api.portal.get_registry_record(
            "collective.vectorsearch.embedding_chunk_size"
        )
        self.assertEqual(value, 1000)

        # Create a new index and verify it uses the new setting
        from collective.vectorsearch.vector_index import VectorIndex

        index = VectorIndex("test_index2")
        self.assertEqual(index.embedding.chunk_size, 1000)

        # Reset to default for other tests
        api.portal.set_registry_record(
            "collective.vectorsearch.embedding_chunk_size", 500
        )
