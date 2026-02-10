"""Unit tests for annotations module."""

import unittest
from unittest.mock import MagicMock, patch


class TestAnnotationKeys(unittest.TestCase):
    """Test annotation key constants."""

    def test_annotation_keys_defined(self):
        """Test that all annotation keys are defined."""
        from collective.vectorsearch.annotations import (
            ANNOTATION_KEY_VECTORS,
            ANNOTATION_KEY_ITQ_HASHES,
            ANNOTATION_KEY_PIVOT_DISTANCES,
            ANNOTATION_KEY_MODEL_ID,
        )

        self.assertEqual(ANNOTATION_KEY_VECTORS, "collective.vectorsearch.vectors")
        self.assertEqual(ANNOTATION_KEY_ITQ_HASHES, "collective.vectorsearch.itq_hashes")
        self.assertEqual(ANNOTATION_KEY_PIVOT_DISTANCES, "collective.vectorsearch.pivot_distances")
        self.assertEqual(ANNOTATION_KEY_MODEL_ID, "collective.vectorsearch.model_id")


class TestAnnotationHelpers(unittest.TestCase):
    """Test annotation helper functions."""

    def test_get_vector_data_returns_none_for_non_annotatable(self):
        """Test get_vector_data returns None for non-annotatable objects."""
        from collective.vectorsearch.annotations import get_vector_data

        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_annotations:
            mock_annotations.side_effect = TypeError("Not annotatable")
            result = get_vector_data(obj)
            self.assertIsNone(result)

    def test_get_vector_data_returns_none_when_no_vectors(self):
        """Test get_vector_data returns None when no vectors stored."""
        from collective.vectorsearch.annotations import get_vector_data

        mock_annotations = {}
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations
            result = get_vector_data(obj)
            self.assertIsNone(result)

    def test_get_vector_data_returns_dict_when_vectors_exist(self):
        """Test get_vector_data returns dict when vectors exist."""
        from collective.vectorsearch.annotations import (
            get_vector_data,
            ANNOTATION_KEY_VECTORS,
            ANNOTATION_KEY_ITQ_HASHES,
            ANNOTATION_KEY_PIVOT_DISTANCES,
            ANNOTATION_KEY_MODEL_ID,
        )

        mock_annotations = {
            ANNOTATION_KEY_VECTORS: [[1.0, 2.0], [3.0, 4.0]],
            ANNOTATION_KEY_ITQ_HASHES: [(123, 456)],
            ANNOTATION_KEY_PIVOT_DISTANCES: [(100, 200, 300, 400, 500, 600, 700, 800)],
            ANNOTATION_KEY_MODEL_ID: "test-model",
        }
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations
            result = get_vector_data(obj)

            self.assertIsNotNone(result)
            self.assertEqual(result["vectors"], [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(result["itq_hashes"], [(123, 456)])
            self.assertEqual(result["model_id"], "test-model")

    def test_get_vectors_returns_vectors(self):
        """Test get_vectors returns vectors list."""
        from collective.vectorsearch.annotations import (
            get_vectors,
            ANNOTATION_KEY_VECTORS,
        )

        mock_annotations = {
            ANNOTATION_KEY_VECTORS: [[1.0, 2.0, 3.0]],
        }
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations
            result = get_vectors(obj)
            self.assertEqual(result, [[1.0, 2.0, 3.0]])

    def test_get_itq_hashes_returns_hashes(self):
        """Test get_itq_hashes returns ITQ hashes."""
        from collective.vectorsearch.annotations import (
            get_itq_hashes,
            ANNOTATION_KEY_ITQ_HASHES,
        )

        mock_annotations = {
            ANNOTATION_KEY_ITQ_HASHES: [(123, 456), (789, 101112)],
        }
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations
            result = get_itq_hashes(obj)
            self.assertEqual(result, [(123, 456), (789, 101112)])

    def test_get_pivot_distances_returns_distances(self):
        """Test get_pivot_distances returns pivot distances."""
        from collective.vectorsearch.annotations import (
            get_pivot_distances,
            ANNOTATION_KEY_PIVOT_DISTANCES,
        )

        mock_annotations = {
            ANNOTATION_KEY_PIVOT_DISTANCES: [(100, 200, 300, 400, 500, 600, 700, 800)],
        }
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations
            result = get_pivot_distances(obj)
            self.assertEqual(result, [(100, 200, 300, 400, 500, 600, 700, 800)])

    def test_get_model_id_returns_model(self):
        """Test get_model_id returns model ID."""
        from collective.vectorsearch.annotations import (
            get_model_id,
            ANNOTATION_KEY_MODEL_ID,
        )

        mock_annotations = {
            ANNOTATION_KEY_MODEL_ID: "all-minilm-l6",
        }
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations
            result = get_model_id(obj)
            self.assertEqual(result, "all-minilm-l6")

    def test_set_vector_data_stores_in_annotations(self):
        """Test set_vector_data stores data in annotations."""
        from collective.vectorsearch.annotations import (
            set_vector_data,
            ANNOTATION_KEY_VECTORS,
            ANNOTATION_KEY_ITQ_HASHES,
            ANNOTATION_KEY_PIVOT_DISTANCES,
            ANNOTATION_KEY_MODEL_ID,
        )

        mock_annotations = {}
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations

            set_vector_data(
                obj,
                vectors=[[1.0, 2.0]],
                itq_hashes=[(123, 456)],
                pivot_distances=[(100, 200, 300, 400, 500, 600, 700, 800)],
                model_id="test-model",
            )

            self.assertEqual(mock_annotations[ANNOTATION_KEY_VECTORS], [[1.0, 2.0]])
            self.assertEqual(mock_annotations[ANNOTATION_KEY_ITQ_HASHES], [(123, 456)])
            self.assertEqual(mock_annotations[ANNOTATION_KEY_MODEL_ID], "test-model")

    def test_set_vector_data_converts_numpy_arrays(self):
        """Test set_vector_data converts numpy arrays to lists."""
        import numpy as np
        from collective.vectorsearch.annotations import (
            set_vector_data,
            ANNOTATION_KEY_VECTORS,
        )

        mock_annotations = {}
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations

            vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
            set_vector_data(
                obj,
                vectors=vectors,
                itq_hashes=None,
                pivot_distances=None,
                model_id=None,
            )

            # Should be converted to list
            self.assertEqual(mock_annotations[ANNOTATION_KEY_VECTORS], [[1.0, 2.0], [3.0, 4.0]])
            self.assertIsInstance(mock_annotations[ANNOTATION_KEY_VECTORS], list)

    def test_clear_vector_data_removes_all_keys(self):
        """Test clear_vector_data removes all annotation keys."""
        from collective.vectorsearch.annotations import (
            clear_vector_data,
            ANNOTATION_KEY_VECTORS,
            ANNOTATION_KEY_ITQ_HASHES,
            ANNOTATION_KEY_PIVOT_DISTANCES,
            ANNOTATION_KEY_MODEL_ID,
        )

        mock_annotations = {
            ANNOTATION_KEY_VECTORS: [[1.0, 2.0]],
            ANNOTATION_KEY_ITQ_HASHES: [(123, 456)],
            ANNOTATION_KEY_PIVOT_DISTANCES: [(100, 200, 300, 400, 500, 600, 700, 800)],
            ANNOTATION_KEY_MODEL_ID: "test-model",
        }
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations

            clear_vector_data(obj)

            self.assertNotIn(ANNOTATION_KEY_VECTORS, mock_annotations)
            self.assertNotIn(ANNOTATION_KEY_ITQ_HASHES, mock_annotations)
            self.assertNotIn(ANNOTATION_KEY_PIVOT_DISTANCES, mock_annotations)
            self.assertNotIn(ANNOTATION_KEY_MODEL_ID, mock_annotations)

    def test_has_vector_data_returns_true_when_vectors_exist(self):
        """Test has_vector_data returns True when vectors exist."""
        from collective.vectorsearch.annotations import (
            has_vector_data,
            ANNOTATION_KEY_VECTORS,
        )

        mock_annotations = {
            ANNOTATION_KEY_VECTORS: [[1.0, 2.0]],
        }
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations
            result = has_vector_data(obj)
            self.assertTrue(result)

    def test_has_vector_data_returns_false_when_no_vectors(self):
        """Test has_vector_data returns False when no vectors exist."""
        from collective.vectorsearch.annotations import has_vector_data

        mock_annotations = {}
        obj = MagicMock()
        with patch("collective.vectorsearch.annotations.IAnnotations") as mock_ia:
            mock_ia.return_value = mock_annotations
            result = has_vector_data(obj)
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
