# -*- coding: utf-8 -*-
"""Model cache tests for this package."""
import unittest
from unittest.mock import Mock, patch, MagicMock


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

    @patch('collective.vectorsearch.vector_index.SentenceTransformer')
    def test_get_model_loads_new_model(self, mock_transformer):
        """Test that get_model loads a model if not cached."""
        from collective.vectorsearch.vector_index import ModelCache

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        cache = ModelCache()
        result = cache.get_model('test-model')

        # Should have called SentenceTransformer constructor
        mock_transformer.assert_called_once_with('test-model')
        # Should return the model
        self.assertEqual(result, mock_model)

    @patch('collective.vectorsearch.vector_index.SentenceTransformer')
    def test_get_model_returns_cached_model(self, mock_transformer):
        """Test that get_model returns cached model on second call."""
        from collective.vectorsearch.vector_index import ModelCache

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        cache = ModelCache()

        # First call should load the model
        result1 = cache.get_model('test-model')
        self.assertEqual(mock_transformer.call_count, 1)

        # Second call should return cached model without loading
        result2 = cache.get_model('test-model')
        self.assertEqual(mock_transformer.call_count, 1)  # Still only 1 call

        # Both should be the same model
        self.assertIs(result1, result2)

    @patch('collective.vectorsearch.vector_index.SentenceTransformer')
    def test_different_models_cached_separately(self, mock_transformer):
        """Test that different models are cached separately."""
        from collective.vectorsearch.vector_index import ModelCache

        mock_model1 = Mock(name='model1')
        mock_model2 = Mock(name='model2')

        # Return different models for different names
        def mock_constructor(name):
            if name == 'model-1':
                return mock_model1
            elif name == 'model-2':
                return mock_model2

        mock_transformer.side_effect = mock_constructor

        cache = ModelCache()

        result1 = cache.get_model('model-1')
        result2 = cache.get_model('model-2')

        # Should have loaded both models
        self.assertEqual(mock_transformer.call_count, 2)
        # Should be different models
        self.assertIsNot(result1, result2)

    def test_clear_cache(self):
        """Test that clear_cache removes all models."""
        from collective.vectorsearch.vector_index import ModelCache

        with patch('collective.vectorsearch.vector_index.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_transformer.return_value = mock_model

            cache = ModelCache()

            # Load a model
            cache.get_model('test-model')
            info = cache.get_cache_info()
            self.assertEqual(info['model_count'], 1)

            # Clear the cache
            cache.clear_cache()
            info = cache.get_cache_info()
            self.assertEqual(info['model_count'], 0)
            self.assertEqual(info['cached_models'], [])

    def test_get_cache_info(self):
        """Test that get_cache_info returns correct information."""
        from collective.vectorsearch.vector_index import ModelCache

        with patch('collective.vectorsearch.vector_index.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_transformer.return_value = mock_model

            cache = ModelCache()

            # Initially empty
            info = cache.get_cache_info()
            self.assertEqual(info['model_count'], 0)
            self.assertEqual(info['cached_models'], [])

            # Load two models
            cache.get_model('model-1')
            cache.get_model('model-2')

            info = cache.get_cache_info()
            self.assertEqual(info['model_count'], 2)
            self.assertIn('model-1', info['cached_models'])
            self.assertIn('model-2', info['cached_models'])


class TestVectorIndexWithModelCache(unittest.TestCase):
    """Test that VectorIndex uses ModelCache correctly."""

    def setUp(self):
        """Clear the cache before each test."""
        from collective.vectorsearch.vector_index import ModelCache
        cache = ModelCache()
        cache.clear_cache()

    @patch('collective.vectorsearch.vector_index.SentenceTransformer')
    def test_multiple_indexes_share_model(self, mock_transformer):
        """Test that multiple VectorIndex instances share the same model."""
        from collective.vectorsearch.vector_index import VectorIndex, ModelCache

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        # Create two indexes with the same model
        index1 = VectorIndex('index1')
        index2 = VectorIndex('index2')

        # SentenceTransformer should only be called once
        self.assertEqual(mock_transformer.call_count, 1)

        # Both should use the same model
        self.assertIs(index1.embedding.model, index2.embedding.model)

        # Cache should show one model
        cache = ModelCache()
        info = cache.get_cache_info()
        self.assertEqual(info['model_count'], 1)

    @patch('collective.vectorsearch.vector_index.SentenceTransformer')
    def test_indexes_with_different_models(self, mock_transformer):
        """Test indexes with different model configurations."""
        from collective.vectorsearch.vector_index import VectorIndex, ModelCache
        from plone import api

        # Mock different models
        mock_model1 = Mock(name='model1')
        mock_model2 = Mock(name='model2')

        def mock_constructor(name):
            if 'gte-small' in name:
                return mock_model1
            elif 'other-model' in name:
                return mock_model2

        mock_transformer.side_effect = mock_constructor

        # Create index with default model
        index1 = VectorIndex('index1')

        # Mock registry to return different model for second index
        with patch('collective.vectorsearch.vector_index.api.portal.get_registry_record') as mock_registry:
            mock_registry.side_effect = lambda key, default=None: {
                'collective.vectorsearch.embedding_model_name': 'other-model',
                'collective.vectorsearch.embedding_prefix_query': 'query: ',
                'collective.vectorsearch.embedding_chunk_size': 500,
                'collective.vectorsearch.similarity_algorithm': 'cosine',
            }.get(key, default)

            index2 = VectorIndex('index2')

        # Should have loaded both models
        self.assertGreaterEqual(mock_transformer.call_count, 2)

        # Cache should show multiple models
        cache = ModelCache()
        info = cache.get_cache_info()
        self.assertGreaterEqual(info['model_count'], 1)
