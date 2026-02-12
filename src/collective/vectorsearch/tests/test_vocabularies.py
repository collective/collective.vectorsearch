# -*- coding: utf-8 -*-
"""Vocabulary and interface invariant tests for this package."""

import unittest

from zope.interface import Invalid


class TestStorageBackendsVocabulary(unittest.TestCase):
    """Test StorageBackendsVocabulary labels."""

    def test_btrees_is_available(self):
        """Test that BTrees has no 'not yet available' label."""
        from collective.vectorsearch.vocabularies import StorageBackendsVocabulary

        vocab = StorageBackendsVocabulary()(None)
        term = vocab.getTermByToken("btrees")
        self.assertEqual(term.title, "BTrees (Internal)")
        self.assertNotIn("not yet available", term.title)

    def test_faiss_not_yet_available(self):
        """Test that FAISS is labeled as not yet available."""
        from collective.vectorsearch.vocabularies import StorageBackendsVocabulary

        vocab = StorageBackendsVocabulary()(None)
        term = vocab.getTermByToken("faiss")
        self.assertIn("not yet available", term.title)

    def test_duckdb_not_yet_available(self):
        """Test that DuckDB is labeled as not yet available."""
        from collective.vectorsearch.vocabularies import StorageBackendsVocabulary

        vocab = StorageBackendsVocabulary()(None)
        term = vocab.getTermByToken("duckdb")
        self.assertIn("not yet available", term.title)

    def test_annoy_not_yet_available(self):
        """Test that Annoy is labeled as not yet available."""
        from collective.vectorsearch.vocabularies import StorageBackendsVocabulary

        vocab = StorageBackendsVocabulary()(None)
        term = vocab.getTermByToken("annoy")
        self.assertIn("not yet available", term.title)


class TestApproximationAlgorithmsVocabulary(unittest.TestCase):
    """Test ApproximationAlgorithmsVocabulary labels."""

    def test_exhaustive_cosine_is_available(self):
        """Test that Exhaustive Cosine has no 'not yet available' label."""
        from collective.vectorsearch.vocabularies import (
            ApproximationAlgorithmsVocabulary,
        )

        vocab = ApproximationAlgorithmsVocabulary()(None)
        term = vocab.getTermByToken("exhaustive_cosine")
        self.assertNotIn("not yet available", term.title)

    def test_itq_lsh_2stage_is_available(self):
        """Test that ITQ LSH 2-stage has no 'not yet available' label."""
        from collective.vectorsearch.vocabularies import (
            ApproximationAlgorithmsVocabulary,
        )

        vocab = ApproximationAlgorithmsVocabulary()(None)
        term = vocab.getTermByToken("itq_lsh_2stage")
        self.assertNotIn("not yet available", term.title)

    def test_itq_lsh_3stage_is_available(self):
        """Test that ITQ LSH 3-stage has no 'not yet available' label."""
        from collective.vectorsearch.vocabularies import (
            ApproximationAlgorithmsVocabulary,
        )

        vocab = ApproximationAlgorithmsVocabulary()(None)
        term = vocab.getTermByToken("itq_lsh_3stage")
        self.assertNotIn("not yet available", term.title)

    def test_hnsw_not_yet_available(self):
        """Test that HNSW is labeled as not yet available."""
        from collective.vectorsearch.vocabularies import (
            ApproximationAlgorithmsVocabulary,
        )

        vocab = ApproximationAlgorithmsVocabulary()(None)
        term = vocab.getTermByToken("hnsw")
        self.assertIn("not yet available", term.title)


class TestSettingsInvariants(unittest.TestCase):
    """Test IVectorSearchSettings invariant validation."""

    def _make_settings(self, **kwargs):
        """Create a mock settings object for invariant testing."""

        class MockSettings:
            storage_backend = kwargs.get("storage_backend", "btrees")
            external_db_uri = kwargs.get("external_db_uri", "")
            approximation_algorithm = kwargs.get(
                "approximation_algorithm", "exhaustive_cosine"
            )

        return MockSettings()

    def test_btrees_backend_is_valid(self):
        """Test that btrees backend passes validation."""
        from collective.vectorsearch.interfaces import IVectorSearchSettings

        settings = self._make_settings(storage_backend="btrees")
        # Should not raise
        IVectorSearchSettings.validateInvariants(settings)

    def test_faiss_backend_is_rejected(self):
        """Test that faiss backend is rejected by invariant."""
        from collective.vectorsearch.interfaces import IVectorSearchSettings

        settings = self._make_settings(
            storage_backend="faiss", external_db_uri="http://example.com/faiss"
        )
        with self.assertRaises(Invalid):
            IVectorSearchSettings.validateInvariants(settings)

    def test_duckdb_backend_is_rejected(self):
        """Test that duckdb backend is rejected by invariant."""
        from collective.vectorsearch.interfaces import IVectorSearchSettings

        settings = self._make_settings(
            storage_backend="duckdb",
            external_db_uri="duckdb:///path/to/db.duckdb",
        )
        with self.assertRaises(Invalid):
            IVectorSearchSettings.validateInvariants(settings)

    def test_exhaustive_cosine_is_valid(self):
        """Test that exhaustive_cosine algorithm passes validation."""
        from collective.vectorsearch.interfaces import IVectorSearchSettings

        settings = self._make_settings(approximation_algorithm="exhaustive_cosine")
        # Should not raise
        IVectorSearchSettings.validateInvariants(settings)

    def test_itq_lsh_2stage_is_valid(self):
        """Test that itq_lsh_2stage algorithm passes validation."""
        from collective.vectorsearch.interfaces import IVectorSearchSettings

        settings = self._make_settings(approximation_algorithm="itq_lsh_2stage")
        # Should not raise
        IVectorSearchSettings.validateInvariants(settings)

    def test_itq_lsh_3stage_is_valid(self):
        """Test that itq_lsh_3stage algorithm passes validation."""
        from collective.vectorsearch.interfaces import IVectorSearchSettings

        settings = self._make_settings(approximation_algorithm="itq_lsh_3stage")
        # Should not raise
        IVectorSearchSettings.validateInvariants(settings)

    def test_hnsw_algorithm_is_rejected(self):
        """Test that hnsw algorithm is rejected by invariant."""
        from collective.vectorsearch.interfaces import IVectorSearchSettings

        settings = self._make_settings(approximation_algorithm="hnsw")
        with self.assertRaises(Invalid):
            IVectorSearchSettings.validateInvariants(settings)
