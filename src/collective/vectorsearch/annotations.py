"""Annotation storage for vector search data.

This module provides utilities for storing and retrieving vector search data
(embeddings, ITQ hashes, pivot distances) in content object annotations.

The data flow is:
1. Event subscriber computes embeddings when content is created/modified
2. Data is stored in content annotations
3. Catalog indexers read from annotations
4. VectorIndex reads from annotations during search

Annotation Keys:
- collective.vectorsearch.vectors: List of embedding vectors (as Python lists)
- collective.vectorsearch.itq_hashes: List of (high, low) tuples for ITQ hashes
- collective.vectorsearch.pivot_distances: List of 8-tuples for pivot distances
- collective.vectorsearch.model_id: ID of the embedding model used
"""

import logging

import numpy as np
from zope.annotation.interfaces import IAnnotations

logger = logging.getLogger("collective.vectorsearch")

# Annotation keys
ANNOTATION_KEY_VECTORS = "collective.vectorsearch.vectors"
ANNOTATION_KEY_ITQ_HASHES = "collective.vectorsearch.itq_hashes"
ANNOTATION_KEY_PIVOT_DISTANCES = "collective.vectorsearch.pivot_distances"
ANNOTATION_KEY_MODEL_ID = "collective.vectorsearch.model_id"


def get_vector_data(obj):
    """Get all vector search data from object annotations.

    Args:
        obj: Content object

    Returns:
        dict with keys 'vectors', 'itq_hashes', 'pivot_distances', 'model_id'
        or None if no data exists
    """
    try:
        annotations = IAnnotations(obj, None)
        if annotations is None:
            return None

        vectors = annotations.get(ANNOTATION_KEY_VECTORS)
        if vectors is None:
            return None

        return {
            "vectors": vectors,
            "itq_hashes": annotations.get(ANNOTATION_KEY_ITQ_HASHES),
            "pivot_distances": annotations.get(ANNOTATION_KEY_PIVOT_DISTANCES),
            "model_id": annotations.get(ANNOTATION_KEY_MODEL_ID),
        }
    except TypeError:
        # Object doesn't support annotations
        logger.debug(f"Object {obj} does not support annotations")
        return None


def get_vectors(obj):
    """Get embedding vectors from object annotations.

    Args:
        obj: Content object

    Returns:
        List of vectors (each vector is a list of floats), or None
    """
    try:
        annotations = IAnnotations(obj, None)
        if annotations is None:
            return None
        return annotations.get(ANNOTATION_KEY_VECTORS)
    except TypeError:
        return None


def get_itq_hashes(obj):
    """Get ITQ hashes from object annotations.

    Args:
        obj: Content object

    Returns:
        List of (high, low) tuples, or None
    """
    try:
        annotations = IAnnotations(obj, None)
        if annotations is None:
            return None
        return annotations.get(ANNOTATION_KEY_ITQ_HASHES)
    except TypeError:
        return None


def get_pivot_distances(obj):
    """Get pivot distances from object annotations.

    Args:
        obj: Content object

    Returns:
        List of 8-tuples (distances to each pivot), or None
    """
    try:
        annotations = IAnnotations(obj, None)
        if annotations is None:
            return None
        return annotations.get(ANNOTATION_KEY_PIVOT_DISTANCES)
    except TypeError:
        return None


def get_model_id(obj):
    """Get the model ID used for embedding from object annotations.

    Args:
        obj: Content object

    Returns:
        Model ID string, or None
    """
    try:
        annotations = IAnnotations(obj, None)
        if annotations is None:
            return None
        return annotations.get(ANNOTATION_KEY_MODEL_ID)
    except TypeError:
        return None


def set_vector_data(obj, vectors, itq_hashes, pivot_distances, model_id):
    """Store all vector search data in object annotations.

    Args:
        obj: Content object
        vectors: List of embedding vectors (numpy arrays will be converted to lists)
        itq_hashes: List of (high, low) tuples for ITQ hashes
        pivot_distances: List of 8-tuples for pivot distances
        model_id: ID of the embedding model used
    """
    try:
        annotations = IAnnotations(obj)

        # Convert numpy arrays to lists if necessary
        if vectors is not None:
            if isinstance(vectors, np.ndarray):
                vectors = vectors.tolist()
            elif isinstance(vectors, list) and len(vectors) > 0:
                if isinstance(vectors[0], np.ndarray):
                    vectors = [v.tolist() for v in vectors]

        # Store in annotations
        annotations[ANNOTATION_KEY_VECTORS] = vectors
        annotations[ANNOTATION_KEY_ITQ_HASHES] = itq_hashes
        annotations[ANNOTATION_KEY_PIVOT_DISTANCES] = pivot_distances
        annotations[ANNOTATION_KEY_MODEL_ID] = model_id

        logger.debug(
            f"Stored vector data for {obj}: "
            f"{len(vectors) if vectors else 0} vectors, "
            f"{len(itq_hashes) if itq_hashes else 0} ITQ hashes, "
            f"{len(pivot_distances) if pivot_distances else 0} pivot distances"
        )
    except TypeError as e:
        logger.warning(f"Could not store vector data for {obj}: {e}")
        raise


def clear_vector_data(obj):
    """Remove all vector search data from object annotations.

    Args:
        obj: Content object
    """
    try:
        annotations = IAnnotations(obj, None)
        if annotations is None:
            return

        for key in [
            ANNOTATION_KEY_VECTORS,
            ANNOTATION_KEY_ITQ_HASHES,
            ANNOTATION_KEY_PIVOT_DISTANCES,
            ANNOTATION_KEY_MODEL_ID,
        ]:
            if key in annotations:
                del annotations[key]

        logger.debug(f"Cleared vector data for {obj}")
    except TypeError:
        # Object doesn't support annotations
        pass


def has_vector_data(obj):
    """Check if object has vector search data in annotations.

    Args:
        obj: Content object

    Returns:
        True if object has vector data, False otherwise
    """
    try:
        annotations = IAnnotations(obj, None)
        if annotations is None:
            return False
        return ANNOTATION_KEY_VECTORS in annotations
    except TypeError:
        return False
