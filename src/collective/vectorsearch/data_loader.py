# -*- coding: utf-8 -*-
"""Data loading utilities for ITQ and pivot data.

This module provides classes and functions to load pre-computed ITQ
(Iterative Quantization) boundary data and pivot data for approximate
nearest neighbor search.

Based on: https://github.com/cmscom/lsh-cascade-poc
"""

import logging
from importlib.resources import as_file, files
from typing import Optional

import numpy as np

logger = logging.getLogger("collective.vectorsearch")


class ITQData:
    """Container for ITQ transformation data.

    ITQ (Iterative Quantization) transforms high-dimensional vectors
    into compact binary hashes for fast similarity search.

    Attributes:
        mean_vector: Centering vector (vector_dims,)
        pca_matrix: PCA projection matrix (vector_dims, hash_length)
        rotation_matrix: ITQ rotation matrix (hash_length, hash_length)
        metadata: Optional metadata dictionary
    """

    def __init__(
        self,
        mean_vector: np.ndarray,
        pca_matrix: np.ndarray,
        rotation_matrix: np.ndarray,
        metadata: Optional[dict] = None,
    ):
        self.mean_vector = mean_vector
        self.pca_matrix = pca_matrix
        self.rotation_matrix = rotation_matrix
        self.metadata = metadata or {}

    def compute_hash(self, embedding: np.ndarray) -> np.ndarray:
        """Compute binary hash from embedding vector(s).

        Supports both single vector (D,) and batch (N, D) inputs.

        Args:
            embedding: Input vector(s) with shape (D,) or (N, D)

        Returns:
            Binary hash array with shape (hash_length,) or (N, hash_length)
            Values are uint8 (0 or 1)
        """
        single_input = embedding.ndim == 1
        if single_input:
            embedding = embedding.reshape(1, -1)

        # 1. Centering
        centered = embedding - self.mean_vector

        # 2. PCA projection
        projected = centered @ self.pca_matrix

        # 3. ITQ rotation
        rotated = projected @ self.rotation_matrix

        # 4. Sign quantization
        binary_hash = (rotated > 0).astype(np.uint8)

        if single_input:
            return binary_hash[0]
        return binary_hash


class PivotData:
    """Container for pivot data used in triangle inequality filtering.

    Pivots are reference vectors used to prune search candidates
    using the triangle inequality.

    Attributes:
        pivots: Pivot vectors with shape (num_pivots, vector_dims)
    """

    def __init__(self, pivots: np.ndarray):
        self.pivots = pivots

    @property
    def num_pivots(self) -> int:
        """Number of pivot vectors."""
        return self.pivots.shape[0]

    @property
    def vector_dims(self) -> int:
        """Dimensionality of pivot vectors."""
        return self.pivots.shape[1]

    def compute_distances(self, embedding: np.ndarray) -> np.ndarray:
        """Compute cosine distances from embedding to each pivot.

        Args:
            embedding: Query vector with shape (D,) or (1, D)

        Returns:
            Distances array with shape (num_pivots,)
            Values are cosine distances (1 - cosine_similarity)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # cosine_similarity returns (1, num_pivots), we want (num_pivots,)
        similarity = cosine_similarity(embedding, self.pivots)[0]
        return 1 - similarity  # distance = 1 - similarity

    def filter_candidates(
        self,
        doc_pivot_distances: np.ndarray,
        query_pivot_distances: np.ndarray,
        threshold: float = 0.20,
    ) -> np.ndarray:
        """Filter candidates using triangle inequality.

        Documents whose maximum distance difference to any pivot exceeds
        the threshold are filtered out.

        Args:
            doc_pivot_distances: Document-pivot distances (N, num_pivots)
            query_pivot_distances: Query-pivot distances (num_pivots,)
            threshold: Maximum allowed distance difference

        Returns:
            Boolean mask (N,) where True indicates a candidate
        """
        diff = np.abs(doc_pivot_distances - query_pivot_distances)
        return np.max(diff, axis=1) < threshold


def load_itq_data(model_id: str) -> Optional[ITQData]:
    """Load ITQ boundary data for a model.

    Looks for data files in the package's data/itq/{model_id}/ directory.
    Expected files:
        - mean_vector.npy
        - pca_matrix.npy
        - rotation_matrix.npy
        - metadata.npy (optional)

    Args:
        model_id: Model identifier (e.g., 'all_minilm_l6', 'e5_base_multilingual')
                  Note: Use underscores, not hyphens

    Returns:
        ITQData instance or None if data not available
    """
    try:
        data_pkg = files("collective.vectorsearch.data.itq").joinpath(model_id)

        # Load required files
        with as_file(data_pkg.joinpath("mean_vector.npy")) as path:
            mean_vector = np.load(path)

        with as_file(data_pkg.joinpath("pca_matrix.npy")) as path:
            pca_matrix = np.load(path)

        with as_file(data_pkg.joinpath("rotation_matrix.npy")) as path:
            rotation_matrix = np.load(path)

        # Load optional metadata
        metadata = None
        try:
            with as_file(data_pkg.joinpath("metadata.npy")) as path:
                metadata = np.load(path, allow_pickle=True).item()
        except (FileNotFoundError, TypeError):
            pass

        logger.debug(f"Loaded ITQ data for model: {model_id}")
        return ITQData(mean_vector, pca_matrix, rotation_matrix, metadata)

    except FileNotFoundError:
        logger.debug(f"ITQ data not found for model: {model_id}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load ITQ data for {model_id}: {e}")
        return None


def load_pivot_data(model_id: str) -> Optional[PivotData]:
    """Load pivot data for a model.

    Looks for data file at data/pivot/{model_id}.npy

    Args:
        model_id: Model identifier (e.g., 'all_minilm_l6', 'e5_base_multilingual')
                  Note: Use underscores, not hyphens

    Returns:
        PivotData instance or None if data not available
    """
    try:
        data_pkg = files("collective.vectorsearch.data.pivot")
        filename = f"{model_id}.npy"

        with as_file(data_pkg.joinpath(filename)) as path:
            pivots = np.load(path)

        logger.debug(f"Loaded pivot data for model: {model_id}")
        return PivotData(pivots)

    except FileNotFoundError:
        logger.debug(f"Pivot data not found for model: {model_id}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load pivot data for {model_id}: {e}")
        return None


def validate_itq_data(
    itq_data: ITQData, vector_dims: int, hash_length: int = 128
) -> bool:
    """Validate ITQ data dimensions match model configuration.

    Args:
        itq_data: ITQData instance to validate
        vector_dims: Expected vector dimensionality
        hash_length: Expected hash length (default: 128)

    Returns:
        True if valid, False otherwise
    """
    if itq_data.mean_vector.shape != (vector_dims,):
        logger.error(
            f"ITQ mean_vector shape mismatch: expected ({vector_dims},), "
            f"got {itq_data.mean_vector.shape}"
        )
        return False

    if itq_data.pca_matrix.shape != (vector_dims, hash_length):
        logger.error(
            f"ITQ pca_matrix shape mismatch: expected ({vector_dims}, {hash_length}), "
            f"got {itq_data.pca_matrix.shape}"
        )
        return False

    if itq_data.rotation_matrix.shape != (hash_length, hash_length):
        logger.error(
            f"ITQ rotation_matrix shape mismatch: "
            f"expected ({hash_length}, {hash_length}), "
            f"got {itq_data.rotation_matrix.shape}"
        )
        return False

    return True


def validate_pivot_data(
    pivot_data: PivotData, vector_dims: int, num_pivots: int = 8
) -> bool:
    """Validate pivot data dimensions match model configuration.

    Args:
        pivot_data: PivotData instance to validate
        vector_dims: Expected vector dimensionality
        num_pivots: Expected number of pivots (default: 8)

    Returns:
        True if valid, False otherwise
    """
    if pivot_data.pivots.shape != (num_pivots, vector_dims):
        logger.error(
            f"Pivot shape mismatch: expected ({num_pivots}, {vector_dims}), "
            f"got {pivot_data.pivots.shape}"
        )
        return False

    return True
