# -*- coding: utf-8 -*-
"""Tests for data_loader module."""

import unittest

import numpy as np


class TestITQData(unittest.TestCase):
    """Test ITQData class."""

    def test_itq_data_instantiation(self):
        """Test ITQData can be instantiated with numpy arrays."""
        from collective.vectorsearch.data_loader import ITQData

        mean_vector = np.zeros(384)
        pca_matrix = np.random.randn(384, 128).astype(np.float32)
        rotation_matrix = np.eye(128).astype(np.float32)

        itq = ITQData(mean_vector, pca_matrix, rotation_matrix)

        self.assertEqual(itq.mean_vector.shape, (384,))
        self.assertEqual(itq.pca_matrix.shape, (384, 128))
        self.assertEqual(itq.rotation_matrix.shape, (128, 128))
        self.assertEqual(itq.metadata, {})

    def test_itq_data_with_metadata(self):
        """Test ITQData with metadata."""
        from collective.vectorsearch.data_loader import ITQData

        mean_vector = np.zeros(768)
        pca_matrix = np.random.randn(768, 128).astype(np.float32)
        rotation_matrix = np.eye(128).astype(np.float32)
        metadata = {"model": "e5-base", "version": "1.0"}

        itq = ITQData(mean_vector, pca_matrix, rotation_matrix, metadata)

        self.assertEqual(itq.metadata["model"], "e5-base")
        self.assertEqual(itq.metadata["version"], "1.0")

    def test_compute_hash_single_vector(self):
        """Test compute_hash with a single vector."""
        from collective.vectorsearch.data_loader import ITQData

        # Create simple ITQ data
        mean_vector = np.zeros(384)
        # PCA matrix that projects to first 128 dims
        pca_matrix = np.eye(384, 128).astype(np.float32)
        rotation_matrix = np.eye(128).astype(np.float32)

        itq = ITQData(mean_vector, pca_matrix, rotation_matrix)

        # Test vector with known values
        vector = np.ones(384)
        hash_result = itq.compute_hash(vector)

        self.assertEqual(hash_result.shape, (128,))
        self.assertEqual(hash_result.dtype, np.uint8)
        # All values should be 1 (since ones @ eye = ones > 0)
        np.testing.assert_array_equal(hash_result, np.ones(128, dtype=np.uint8))

    def test_compute_hash_batch_vectors(self):
        """Test compute_hash with batch of vectors."""
        from collective.vectorsearch.data_loader import ITQData

        mean_vector = np.zeros(384)
        pca_matrix = np.eye(384, 128).astype(np.float32)
        rotation_matrix = np.eye(128).astype(np.float32)

        itq = ITQData(mean_vector, pca_matrix, rotation_matrix)

        # Batch of 5 vectors
        vectors = np.random.randn(5, 384).astype(np.float32)
        hash_result = itq.compute_hash(vectors)

        self.assertEqual(hash_result.shape, (5, 128))
        self.assertEqual(hash_result.dtype, np.uint8)
        # All values should be 0 or 1
        self.assertTrue(np.all((hash_result == 0) | (hash_result == 1)))

    def test_compute_hash_centering(self):
        """Test that centering is applied correctly."""
        from collective.vectorsearch.data_loader import ITQData

        # Mean vector of ones
        mean_vector = np.ones(384)
        pca_matrix = np.eye(384, 128).astype(np.float32)
        rotation_matrix = np.eye(128).astype(np.float32)

        itq = ITQData(mean_vector, pca_matrix, rotation_matrix)

        # Vector of ones should become zeros after centering
        vector = np.ones(384)
        hash_result = itq.compute_hash(vector)

        # 0 > 0 is False, so all should be 0
        np.testing.assert_array_equal(hash_result, np.zeros(128, dtype=np.uint8))


class TestPivotData(unittest.TestCase):
    """Test PivotData class."""

    def test_pivot_data_instantiation(self):
        """Test PivotData can be instantiated."""
        from collective.vectorsearch.data_loader import PivotData

        pivots = np.random.randn(8, 384).astype(np.float32)
        pivot_data = PivotData(pivots)

        self.assertEqual(pivot_data.pivots.shape, (8, 384))
        self.assertEqual(pivot_data.num_pivots, 8)
        self.assertEqual(pivot_data.vector_dims, 384)

    def test_compute_distances(self):
        """Test compute_distances method."""
        from collective.vectorsearch.data_loader import PivotData

        # Create orthonormal pivots for predictable results
        pivots = np.eye(8, 384).astype(np.float32)
        pivot_data = PivotData(pivots)

        # Query vector
        query = np.zeros(384)
        query[0] = 1.0  # Same direction as first pivot

        distances = pivot_data.compute_distances(query)

        self.assertEqual(distances.shape, (8,))
        # Distance to first pivot should be 0 (same direction)
        self.assertAlmostEqual(distances[0], 0.0, places=5)
        # Distance to other pivots should be 1 (orthogonal)
        for i in range(1, 8):
            self.assertAlmostEqual(distances[i], 1.0, places=5)

    def test_filter_candidates(self):
        """Test filter_candidates method."""
        from collective.vectorsearch.data_loader import PivotData

        pivots = np.eye(8, 384).astype(np.float32)
        pivot_data = PivotData(pivots)

        # Query pivot distances
        query_distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        # Document pivot distances (3 documents)
        doc_distances = np.array(
            [
                [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],  # max diff = 0.05
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # max diff = 0.4
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # max diff = 0.0
            ]
        )

        # With threshold 0.1, only docs 0 and 2 should pass
        mask = pivot_data.filter_candidates(doc_distances, query_distances, threshold=0.1)

        self.assertEqual(mask.shape, (3,))
        self.assertTrue(mask[0])  # diff = 0.05 < 0.1
        self.assertFalse(mask[1])  # diff = 0.4 >= 0.1
        self.assertTrue(mask[2])  # diff = 0.0 < 0.1


class TestValidation(unittest.TestCase):
    """Test validation functions."""

    def test_validate_itq_data_valid(self):
        """Test validation of valid ITQ data."""
        from collective.vectorsearch.data_loader import ITQData, validate_itq_data

        itq = ITQData(
            mean_vector=np.zeros(384),
            pca_matrix=np.zeros((384, 128)),
            rotation_matrix=np.zeros((128, 128)),
        )

        self.assertTrue(validate_itq_data(itq, 384, 128))

    def test_validate_itq_data_wrong_mean_shape(self):
        """Test validation fails with wrong mean_vector shape."""
        from collective.vectorsearch.data_loader import ITQData, validate_itq_data

        itq = ITQData(
            mean_vector=np.zeros(768),  # Wrong!
            pca_matrix=np.zeros((384, 128)),
            rotation_matrix=np.zeros((128, 128)),
        )

        self.assertFalse(validate_itq_data(itq, 384, 128))

    def test_validate_itq_data_wrong_pca_shape(self):
        """Test validation fails with wrong pca_matrix shape."""
        from collective.vectorsearch.data_loader import ITQData, validate_itq_data

        itq = ITQData(
            mean_vector=np.zeros(384),
            pca_matrix=np.zeros((384, 64)),  # Wrong hash length!
            rotation_matrix=np.zeros((128, 128)),
        )

        self.assertFalse(validate_itq_data(itq, 384, 128))

    def test_validate_itq_data_wrong_rotation_shape(self):
        """Test validation fails with wrong rotation_matrix shape."""
        from collective.vectorsearch.data_loader import ITQData, validate_itq_data

        itq = ITQData(
            mean_vector=np.zeros(384),
            pca_matrix=np.zeros((384, 128)),
            rotation_matrix=np.zeros((64, 64)),  # Wrong!
        )

        self.assertFalse(validate_itq_data(itq, 384, 128))

    def test_validate_pivot_data_valid(self):
        """Test validation of valid pivot data."""
        from collective.vectorsearch.data_loader import PivotData, validate_pivot_data

        pivot = PivotData(pivots=np.zeros((8, 384)))

        self.assertTrue(validate_pivot_data(pivot, 384, 8))

    def test_validate_pivot_data_wrong_shape(self):
        """Test validation fails with wrong pivot shape."""
        from collective.vectorsearch.data_loader import PivotData, validate_pivot_data

        pivot = PivotData(pivots=np.zeros((4, 384)))  # Wrong pivot count!

        self.assertFalse(validate_pivot_data(pivot, 384, 8))


class TestLoadFunctions(unittest.TestCase):
    """Test data loading functions."""

    def test_load_itq_data_not_found(self):
        """Test load_itq_data returns None for missing data."""
        from collective.vectorsearch.data_loader import load_itq_data

        result = load_itq_data("nonexistent_model")
        self.assertIsNone(result)

    def test_load_pivot_data_not_found(self):
        """Test load_pivot_data returns None for missing data."""
        from collective.vectorsearch.data_loader import load_pivot_data

        result = load_pivot_data("nonexistent_model")
        self.assertIsNone(result)


class TestProviderDataLoading(unittest.TestCase):
    """Test data loading through model providers."""

    def test_provider_get_itq_boundary_loads_data(self):
        """Test that get_itq_boundary loads data correctly."""
        from collective.vectorsearch.data_loader import ITQData
        from collective.vectorsearch.model_providers import AllMiniLMProvider

        provider = AllMiniLMProvider()
        result = provider.get_itq_boundary()

        # Data files are now included, should load successfully
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ITQData)
        self.assertEqual(result.mean_vector.shape, (384,))
        self.assertEqual(result.pca_matrix.shape, (384, 128))
        self.assertEqual(result.rotation_matrix.shape, (128, 128))

    def test_provider_get_pivot_data_loads_data(self):
        """Test that get_pivot_data loads data correctly."""
        from collective.vectorsearch.data_loader import PivotData
        from collective.vectorsearch.model_providers import AllMiniLMProvider

        provider = AllMiniLMProvider()
        result = provider.get_pivot_data()

        # Data files are now included, should load successfully
        self.assertIsNotNone(result)
        self.assertIsInstance(result, PivotData)
        self.assertEqual(result.pivots.shape, (8, 384))

    def test_provider_get_itq_boundary_returns_none_for_nonexistent(self):
        """Test that get_itq_boundary returns None for non-existent data."""
        from collective.vectorsearch.model_providers import BaseEmbeddingModelProvider

        # Create a mock provider with non-existent data_file_id
        class MockProvider(BaseEmbeddingModelProvider):
            id = "nonexistent-model"
            title = "Nonexistent Model"
            description = "Test model"
            model_name = "test/model"
            vector_dimensions = 512

        provider = MockProvider()
        result = provider.get_itq_boundary()
        self.assertIsNone(result)

    def test_provider_get_pivot_data_returns_none_for_nonexistent(self):
        """Test that get_pivot_data returns None for non-existent data."""
        from collective.vectorsearch.model_providers import BaseEmbeddingModelProvider

        # Create a mock provider with non-existent data_file_id
        class MockProvider(BaseEmbeddingModelProvider):
            id = "nonexistent-model"
            title = "Nonexistent Model"
            description = "Test model"
            model_name = "test/model"
            vector_dimensions = 512

        provider = MockProvider()
        result = provider.get_pivot_data()
        self.assertIsNone(result)

    def test_provider_data_file_id(self):
        """Test _get_data_file_id method."""
        from collective.vectorsearch.model_providers import (
            AllMiniLMProvider,
            E5BaseMultilingualGPUProvider,
            E5BaseMultilingualProvider,
        )

        # AllMiniLM should convert hyphens to underscores
        minilm = AllMiniLMProvider()
        self.assertEqual(minilm._get_data_file_id(), "all_minilm_l6")

        # E5 CPU should convert hyphens to underscores
        e5_cpu = E5BaseMultilingualProvider()
        self.assertEqual(e5_cpu._get_data_file_id(), "e5_base_multilingual")

        # E5 GPU should use explicit data_file_id
        e5_gpu = E5BaseMultilingualGPUProvider()
        self.assertEqual(e5_gpu._get_data_file_id(), "e5_base_multilingual")

    def test_e5_provider_loads_data(self):
        """Test E5 provider loads ITQ/pivot data correctly."""
        from collective.vectorsearch.data_loader import ITQData, PivotData
        from collective.vectorsearch.model_providers import E5BaseMultilingualProvider

        provider = E5BaseMultilingualProvider()
        itq = provider.get_itq_boundary()
        pivot = provider.get_pivot_data()

        self.assertIsNotNone(itq)
        self.assertIsInstance(itq, ITQData)
        self.assertEqual(itq.mean_vector.shape, (768,))
        self.assertEqual(itq.pca_matrix.shape, (768, 128))
        self.assertEqual(itq.rotation_matrix.shape, (128, 128))

        self.assertIsNotNone(pivot)
        self.assertIsInstance(pivot, PivotData)
        self.assertEqual(pivot.pivots.shape, (8, 768))

    def test_e5_gpu_shares_data_with_cpu(self):
        """Test E5 GPU variant shares ITQ/pivot data with CPU variant."""
        import numpy as np

        from collective.vectorsearch.model_providers import (
            E5BaseMultilingualGPUProvider,
            E5BaseMultilingualProvider,
        )

        cpu_provider = E5BaseMultilingualProvider()
        gpu_provider = E5BaseMultilingualGPUProvider()

        cpu_itq = cpu_provider.get_itq_boundary()
        gpu_itq = gpu_provider.get_itq_boundary()

        cpu_pivot = cpu_provider.get_pivot_data()
        gpu_pivot = gpu_provider.get_pivot_data()

        # Both should load data
        self.assertIsNotNone(cpu_itq)
        self.assertIsNotNone(gpu_itq)
        self.assertIsNotNone(cpu_pivot)
        self.assertIsNotNone(gpu_pivot)

        # Data should be identical (same files)
        np.testing.assert_array_equal(cpu_itq.mean_vector, gpu_itq.mean_vector)
        np.testing.assert_array_equal(cpu_itq.pca_matrix, gpu_itq.pca_matrix)
        np.testing.assert_array_equal(cpu_itq.rotation_matrix, gpu_itq.rotation_matrix)
        np.testing.assert_array_equal(cpu_pivot.pivots, gpu_pivot.pivots)


def test_suite():
    """Create test suite."""
    return unittest.TestSuite(
        [
            unittest.TestLoader().loadTestsFromTestCase(TestITQData),
            unittest.TestLoader().loadTestsFromTestCase(TestPivotData),
            unittest.TestLoader().loadTestsFromTestCase(TestValidation),
            unittest.TestLoader().loadTestsFromTestCase(TestLoadFunctions),
            unittest.TestLoader().loadTestsFromTestCase(TestProviderDataLoading),
        ]
    )


if __name__ == "__main__":
    unittest.main()
