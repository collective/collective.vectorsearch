# -*- coding: utf-8 -*-
"""Embedding tests for this package."""
import unittest
import numpy as np
from unittest.mock import Mock


class TestEmbeddingBase(unittest.TestCase):
    """Test the EmbeddingBase class."""

    def test_base_class_has_meta_type(self):
        """Test that base class has meta_type attribute."""
        from collective.vectorsearch.embedding import EmbeddingBase
        self.assertIsNone(EmbeddingBase.meta_type)

    def test_base_class_initialization(self):
        """Test base class initialization."""
        from collective.vectorsearch.embedding import EmbeddingBase

        mock_model = Mock()
        embedding = EmbeddingBase(
            model=mock_model,
            chunk_size=100,
            prefix_query="test: "
        )

        self.assertEqual(embedding.model, mock_model)
        self.assertEqual(embedding.chunk_size, 100)
        self.assertEqual(embedding.prefix_query, "test: ")

    def test_base_class_embed_not_implemented(self):
        """Test that base class embed method raises NotImplementedError."""
        from collective.vectorsearch.embedding import EmbeddingBase

        mock_model = Mock()
        embedding = EmbeddingBase(model=mock_model)

        with self.assertRaises(NotImplementedError):
            embedding.embed("test text")


class TestSentenceTransformerEmbedding(unittest.TestCase):
    """Test the SentenceTransformerEmbedding class."""

    def setUp(self):
        """Create a mock model for testing."""
        self.mock_model = Mock()
        # Mock encode method to return a simple numpy array
        self.mock_model.encode = Mock(
            return_value=np.array([[0.1, 0.2, 0.3]])
        )

    def test_meta_type(self):
        """Test that meta_type is set correctly."""
        from collective.vectorsearch.embedding import SentenceTransformerEmbedding
        self.assertEqual(
            SentenceTransformerEmbedding.meta_type,
            "SentenceTransformerEmbedding"
        )

    def test_embed_simple_text(self):
        """Test embedding a simple text."""
        from collective.vectorsearch.embedding import SentenceTransformerEmbedding

        embedding = SentenceTransformerEmbedding(
            model=self.mock_model,
            chunk_size=500
        )

        result = embedding.embed("test text")

        # Check that encode was called
        self.mock_model.encode.assert_called_once()
        # Check that result is a numpy array
        self.assertIsInstance(result, np.ndarray)

    def test_embed_with_query_prefix(self):
        """Test embedding with query prefix."""
        from collective.vectorsearch.embedding import SentenceTransformerEmbedding

        embedding = SentenceTransformerEmbedding(
            model=self.mock_model,
            chunk_size=500,
            prefix_query="query: "
        )

        embedding.embed("test text", query=True)

        # Check that encode was called with prefixed text
        call_args = self.mock_model.encode.call_args
        texts = call_args[0][0]
        self.assertTrue(texts[0].startswith("query: "))

    def test_embed_without_query_prefix(self):
        """Test embedding without query prefix."""
        from collective.vectorsearch.embedding import SentenceTransformerEmbedding

        embedding = SentenceTransformerEmbedding(
            model=self.mock_model,
            chunk_size=500,
            prefix_query="query: "
        )

        embedding.embed("test text", query=False)

        # Check that encode was called without prefix
        call_args = self.mock_model.encode.call_args
        texts = call_args[0][0]
        self.assertFalse(texts[0].startswith("query: "))
        self.assertEqual(texts[0], "test text")

    def test_text_chunking(self):
        """Test that long text is chunked correctly."""
        from collective.vectorsearch.embedding import SentenceTransformerEmbedding

        embedding = SentenceTransformerEmbedding(
            model=self.mock_model,
            chunk_size=10  # Small chunk size for testing
        )

        # Create a text longer than chunk size
        long_text = "a" * 25  # 25 characters

        embedding.embed(long_text)

        # Check that encode was called
        call_args = self.mock_model.encode.call_args
        texts = call_args[0][0]

        # Should be split into 3 chunks: 10 + 10 + 5
        self.assertEqual(len(texts), 3)
        self.assertEqual(len(texts[0]), 10)
        self.assertEqual(len(texts[1]), 10)
        self.assertEqual(len(texts[2]), 5)

    def test_default_chunk_size(self):
        """Test default chunk size is 500."""
        from collective.vectorsearch.embedding import SentenceTransformerEmbedding

        embedding = SentenceTransformerEmbedding(model=self.mock_model)
        self.assertEqual(embedding.chunk_size, 500)
