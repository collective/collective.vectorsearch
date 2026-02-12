"""Indexers for ITQ hash and pivot distances.

These indexers retrieve pre-computed values from content annotations to be stored
in PortalCatalog for approximate nearest neighbor search using ITQ-LSH
and pivot filtering.

Architecture:
- Event subscribers compute ITQ hash and pivot distances when content changes
- Data is stored in content annotations
- These indexers retrieve the pre-computed values for PortalCatalog storage
- Each document may have multiple vectors (chunks), so each indexer returns
  a list of values

ITQ Hash:
- 128-bit binary hash stored as list of (high_64bit, low_64bit) tuples
- One tuple per chunk in the document
- Used for Hamming distance calculation in stage 2 filtering

Pivot Distances:
- 8 pivot distance values stored as lists of integers (pivot1 to pivot8)
- Distances are multiplied by 1000 and stored as integers
- KeywordIndex stores all values for all chunks, enabling range queries
- Used for triangle inequality filtering in stage 1
"""

import logging

import numpy as np
from plone.indexer import indexer
from Products.CMFCore.interfaces import IContentish

from collective.vectorsearch.annotations import (
    get_itq_hashes as get_itq_hashes_from_annotations,
)
from collective.vectorsearch.annotations import (
    get_pivot_distances as get_pivot_distances_from_annotations,
)
from collective.vectorsearch.annotations import (
    get_vectors as get_vectors_from_annotations,
)

logger = logging.getLogger("collective.vectorsearch")


# Vector Indexer (metadata column)


@indexer(IContentish)
def llm_vector(obj):
    """Indexer for raw embedding vectors of all chunks.

    Reads pre-computed vectors from content annotations.

    Returns:
        list: List of vectors (each a list of floats), one per chunk
              Returns None if no vectors available
    """
    try:
        vectors = get_vectors_from_annotations(obj)
        if vectors:
            return vectors
    except Exception as e:
        logger.debug(f"Could not get llm_vector from annotations: {e}")
    return None


# ITQ Hash Indexer (metadata column)


@indexer(IContentish)
def itq_hashes(obj):
    """Indexer for ITQ hashes of all chunks.

    Reads pre-computed ITQ hashes from content annotations.

    Returns:
        tuple: Tuple of (high_64bit, low_64bit) tuples, one per chunk
               Returns None if no hashes available
    """
    try:
        hashes = get_itq_hashes_from_annotations(obj)
        if hashes:
            # Convert to tuple for catalog storage
            return tuple(hashes)
    except Exception as e:
        logger.debug(f"Could not get itq_hashes from annotations: {e}")
    return None


# Pivot Distance Indexers (KeywordIndex - returns list of values)


def _get_pivot_distances_for_index(obj, pivot_index):
    """Get all pivot distances for a specific pivot index.

    Reads pre-computed pivot distances from content annotations.

    Args:
        obj: Content object
        pivot_index: 0-based pivot index (0-7)

    Returns:
        tuple: Tuple of integer distances for all chunks
               Returns None if no distances available
    """
    try:
        all_distances = get_pivot_distances_from_annotations(obj)
        if all_distances:
            # Extract the specific pivot index from each chunk's distances
            result = []
            for chunk_distances in all_distances:
                if chunk_distances and pivot_index < len(chunk_distances):
                    result.append(chunk_distances[pivot_index])
            if result:
                return tuple(result)
    except Exception as e:
        logger.debug(f"Could not get pivot distances from annotations: {e}")
    return None


@indexer(IContentish)
def pivot1(obj):
    """Indexer for distances to pivot 1 (all chunks)."""
    return _get_pivot_distances_for_index(obj, 0)


@indexer(IContentish)
def pivot2(obj):
    """Indexer for distances to pivot 2 (all chunks)."""
    return _get_pivot_distances_for_index(obj, 1)


@indexer(IContentish)
def pivot3(obj):
    """Indexer for distances to pivot 3 (all chunks)."""
    return _get_pivot_distances_for_index(obj, 2)


@indexer(IContentish)
def pivot4(obj):
    """Indexer for distances to pivot 4 (all chunks)."""
    return _get_pivot_distances_for_index(obj, 3)


@indexer(IContentish)
def pivot5(obj):
    """Indexer for distances to pivot 5 (all chunks)."""
    return _get_pivot_distances_for_index(obj, 4)


@indexer(IContentish)
def pivot6(obj):
    """Indexer for distances to pivot 6 (all chunks)."""
    return _get_pivot_distances_for_index(obj, 5)


@indexer(IContentish)
def pivot7(obj):
    """Indexer for distances to pivot 7 (all chunks)."""
    return _get_pivot_distances_for_index(obj, 6)


@indexer(IContentish)
def pivot8(obj):
    """Indexer for distances to pivot 8 (all chunks)."""
    return _get_pivot_distances_for_index(obj, 7)


# Utility functions for search


def binary_hash_to_integers(binary_hash):
    """Convert 128-bit binary hash to two 64-bit integers.

    Args:
        binary_hash: numpy array of 128 uint8 values (0 or 1)

    Returns:
        tuple: (high_64bits, low_64bits) as Python integers
    """
    if binary_hash is None or len(binary_hash) != 128:
        return None, None

    high_bits = binary_hash[:64]
    low_bits = binary_hash[64:]

    high_int = 0
    low_int = 0

    for i, bit in enumerate(high_bits):
        if bit:
            high_int |= 1 << (63 - i)

    for i, bit in enumerate(low_bits):
        if bit:
            low_int |= 1 << (63 - i)

    return high_int, low_int


def integers_to_binary_hash(high_int, low_int):
    """Convert two 64-bit integers back to 128-bit binary hash.

    Args:
        high_int: Upper 64 bits as integer
        low_int: Lower 64 bits as integer

    Returns:
        numpy array of 128 uint8 values (0 or 1)
    """
    if high_int is None or low_int is None:
        return None

    binary_hash = np.zeros(128, dtype=np.uint8)

    for i in range(64):
        if high_int & (1 << (63 - i)):
            binary_hash[i] = 1
        if low_int & (1 << (63 - i)):
            binary_hash[64 + i] = 1

    return binary_hash


def compute_hamming_distance(hash1_high, hash1_low, hash2_high, hash2_low):
    """Compute Hamming distance between two 128-bit hashes.

    Args:
        hash1_high, hash1_low: First hash as two 64-bit integers
        hash2_high, hash2_low: Second hash as two 64-bit integers

    Returns:
        Integer Hamming distance (0-128)
    """
    xor_high = hash1_high ^ hash2_high
    xor_low = hash1_low ^ hash2_low

    # Count bits using Python's bin() or bit_count() (Python 3.10+)
    return bin(xor_high).count("1") + bin(xor_low).count("1")


def compute_min_hamming_distance(query_hash, doc_hashes):
    """Compute minimum Hamming distance between query hash and document hashes.

    Args:
        query_hash: (high, low) tuple for query
        doc_hashes: List of (high, low) tuples for document chunks

    Returns:
        Integer minimum Hamming distance, or None if no hashes
    """
    if not doc_hashes:
        return None

    query_high, query_low = query_hash
    min_distance = float("inf")

    for doc_high, doc_low in doc_hashes:
        distance = compute_hamming_distance(query_high, query_low, doc_high, doc_low)
        min_distance = min(min_distance, distance)

    return int(min_distance) if min_distance != float("inf") else None


def batch_min_hamming_distance(query_hash_ints, doc_hashes):
    """Compute minimum Hamming distance between query hash and document chunk hashes.

    Uses Python 3.10+ int.bit_count() for fast popcount.

    Args:
        query_hash_ints: (high_64bit, low_64bit) tuple for query
        doc_hashes: sequence of (high_64bit, low_64bit) tuples for document chunks

    Returns:
        int: minimum Hamming distance (0-128), or 129 if doc_hashes is empty
    """
    if not doc_hashes:
        return 129

    q_high, q_low = query_hash_ints
    min_dist = 129

    for doc_high, doc_low in doc_hashes:
        dist = (q_high ^ doc_high).bit_count() + (q_low ^ doc_low).bit_count()
        if dist < min_dist:
            min_dist = dist
            if dist == 0:
                break

    return min_dist


def distance_to_index_value(distance, scale=1000):
    """Convert float distance to integer for index storage.

    Args:
        distance: Float distance value (typically 0.0 to 2.0 for cosine distance)
        scale: Multiplication factor (default 1000 for 3 decimal precision)

    Returns:
        Integer value suitable for index
    """
    if distance is None:
        return None
    return int(round(distance * scale))


def index_value_to_distance(index_value, scale=1000):
    """Convert integer index value back to float distance.

    Args:
        index_value: Integer from index
        scale: Multiplication factor used during storage

    Returns:
        Float distance value
    """
    if index_value is None:
        return None
    return index_value / scale


def get_pivot_range_for_threshold(query_pivot_value, threshold_int):
    """Get range of pivot index values for a given threshold.

    Args:
        query_pivot_value: Integer pivot distance for query
        threshold_int: Integer threshold (e.g., 200 for 0.2 distance)

    Returns:
        tuple: (min_value, max_value) for range query
    """
    min_val = max(0, query_pivot_value - threshold_int)
    max_val = query_pivot_value + threshold_int
    return min_val, max_val
