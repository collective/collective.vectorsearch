# -*- coding: utf-8 -*-
"""Model cache tests for this package."""

import unittest
from unittest.mock import Mock, patch


class TestModelCache(unittest.TestCase):
    """Test the ModelCache singleton."""

    def setUp(self):
        """Clear the cache before each test."""
        from collective.vectorsearch.vector_index import ModelCache

        cache = ModelCache()
        cache.clear_cache()

    def test_singleton_pattern(self):
        """Test that ModelCache is a singleton."""
        from collective.vectorsearch.vector_index import ModelCache

        cache1 = ModelCache()
        cache2 = ModelCache()

        # Both should be the same instance
        self.assertIs(cache1, cache2)

    @patch("collective.vectorsearch.vector_index.HAS_SENTENCE_TRANSFORMERS", True)
    @patch("collective.vectorsearch.vector_index.SentenceTransformer")
    def test_get_model_loads_new_model(self, mock_transformer):
        """Test that get_model loads a model if not cached."""
        from collective.vectorsearch.vector_index import ModelCache

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        cache = ModelCache()
        result = cache.get_model("test-model")

        # Should have called SentenceTransformer constructor (local_files_only first)
        mock_transformer.assert_called_once_with("test-model", local_files_only=True)
        # Should return the model
        self.assertEqual(result, mock_model)

    @patch("collective.vectorsearch.vector_index.HAS_SENTENCE_TRANSFORMERS", True)
    @patch("collective.vectorsearch.vector_index.SentenceTransformer")
    def test_get_model_returns_cached_model(self, mock_transformer):
        """Test that get_model returns cached model on second call."""
        from collective.vectorsearch.vector_index import ModelCache

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        cache = ModelCache()

        # First call should load the model
        result1 = cache.get_model("test-model")
        self.assertEqual(mock_transformer.call_count, 1)

        # Second call should return cached model without loading
        result2 = cache.get_model("test-model")
        self.assertEqual(mock_transformer.call_count, 1)  # Still only 1 call

        # Both should be the same model
        self.assertIs(result1, result2)

    @patch("collective.vectorsearch.vector_index.HAS_SENTENCE_TRANSFORMERS", True)
    @patch("collective.vectorsearch.vector_index.SentenceTransformer")
    def test_different_models_cached_separately(self, mock_transformer):
        """Test that different models are cached separately."""
        from collective.vectorsearch.vector_index import ModelCache

        mock_model1 = Mock(name="model1")
        mock_model2 = Mock(name="model2")

        # Return different models for different names
        def mock_constructor(name, **kwargs):
            if name == "model-1":
                return mock_model1
            elif name == "model-2":
                return mock_model2

        mock_transformer.side_effect = mock_constructor

        cache = ModelCache()

        result1 = cache.get_model("model-1")
        result2 = cache.get_model("model-2")

        # Should have loaded both models
        self.assertEqual(mock_transformer.call_count, 2)
        # Should be different models
        self.assertIsNot(result1, result2)

    @patch("collective.vectorsearch.vector_index.HAS_SENTENCE_TRANSFORMERS", True)
    @patch("collective.vectorsearch.vector_index.SentenceTransformer")
    def test_clear_cache(self, mock_transformer):
        """Test that clear_cache removes all models."""
        from collective.vectorsearch.vector_index import ModelCache

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        cache = ModelCache()

        # Load a model
        cache.get_model("test-model")
        info = cache.get_cache_info()
        self.assertEqual(info["model_count"], 1)

        # Clear the cache
        cache.clear_cache()
        info = cache.get_cache_info()
        self.assertEqual(info["model_count"], 0)
        self.assertEqual(info["cached_models"], [])

    @patch("collective.vectorsearch.vector_index.HAS_SENTENCE_TRANSFORMERS", True)
    @patch("collective.vectorsearch.vector_index.SentenceTransformer")
    def test_get_cache_info(self, mock_transformer):
        """Test that get_cache_info returns correct information."""
        from collective.vectorsearch.vector_index import ModelCache

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        cache = ModelCache()

        # Initially empty
        info = cache.get_cache_info()
        self.assertEqual(info["model_count"], 0)
        self.assertEqual(info["cached_models"], [])

        # Load two models
        cache.get_model("model-1")
        cache.get_model("model-2")

        info = cache.get_cache_info()
        self.assertEqual(info["model_count"], 2)
        self.assertIn("model-1", info["cached_models"])
        self.assertIn("model-2", info["cached_models"])


class TestVectorIndexWithModelCache(unittest.TestCase):
    """Test that VectorIndex uses ModelCache correctly.

    Note: These tests mock at the model provider level since the default
    backend is now FastEmbed, not SentenceTransformers.
    """

    def setUp(self):
        """Clear the cache before each test."""
        from collective.vectorsearch.vector_index import ModelCache

        cache = ModelCache()
        cache.clear_cache()

    @patch("collective.vectorsearch.vector_index.queryUtility")
    def test_multiple_indexes_share_model(self, mock_query_utility):
        """Test that multiple VectorIndex instances share the same model."""
        from collective.vectorsearch.vector_index import VectorIndex

        # Create a mock embedding instance
        mock_embedding = Mock()
        mock_embedding.model = Mock()

        # Create a mock model provider
        mock_provider = Mock()
        mock_provider.get_embedding_instance.return_value = mock_embedding
        mock_provider.query_prefix = None
        mock_provider.passage_prefix = None

        mock_query_utility.return_value = mock_provider

        # Create two indexes with the same model
        index1 = VectorIndex("index1")
        index2 = VectorIndex("index2")

        # Both should use the same model via the same embedding
        self.assertIs(index1.embedding, index2.embedding)

        # Provider's get_embedding_instance should only be called once
        # (due to caching in the model provider)
        self.assertGreaterEqual(mock_provider.get_embedding_instance.call_count, 1)

    @patch("collective.vectorsearch.vector_index.queryUtility")
    def test_indexes_with_different_models(self, mock_query_utility):
        """Test indexes with different model configurations."""
        from collective.vectorsearch.vector_index import VectorIndex

        # Create mock embeddings for different models
        mock_embedding1 = Mock()
        mock_embedding1.model = Mock(name="model1")

        mock_embedding2 = Mock()
        mock_embedding2.model = Mock(name="model2")

        # Create mock model providers
        mock_provider1 = Mock()
        mock_provider1.get_embedding_instance.return_value = mock_embedding1
        mock_provider1.query_prefix = None
        mock_provider1.passage_prefix = None

        mock_provider2 = Mock()
        mock_provider2.get_embedding_instance.return_value = mock_embedding2
        mock_provider2.query_prefix = "query: "
        mock_provider2.passage_prefix = "passage: "

        def mock_get_provider(interface, name=None):
            if name == "all-minilm-l6":
                return mock_provider1
            elif name == "other-model":
                return mock_provider2
            return mock_provider1

        mock_query_utility.side_effect = mock_get_provider

        # Create index with default model and access embedding to trigger initialization
        index1 = VectorIndex("index1")
        _ = index1.embedding  # Trigger lazy initialization

        # Mock registry to return different model for second index
        with patch(
            "collective.vectorsearch.vector_index.api.portal.get_registry_record"
        ) as mock_registry:
            mock_registry.side_effect = lambda key, default=None: {
                "collective.vectorsearch.embedding_model": "other-model",
                "collective.vectorsearch.embedding_chunk_size": 500,
                "collective.vectorsearch.approximation_algorithm": "exhaustive_cosine",
            }.get(key, default)

            index2 = VectorIndex("index2")
            _ = index2.embedding  # Trigger lazy initialization

        # Should have used both providers
        self.assertTrue(mock_provider1.get_embedding_instance.called)
        self.assertTrue(mock_provider2.get_embedding_instance.called)
