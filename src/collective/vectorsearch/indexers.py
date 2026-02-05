"""Indexers for ITQ hash and pivot distances.

These indexers retrieve pre-computed values from VectorIndex to be stored
in PortalCatalog for approximate nearest neighbor search using ITQ-LSH
and pivot filtering.

Architecture:
- VectorIndex computes ITQ hash and pivot distances during index_doc()
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

from plone.indexer import indexer
from Products.CMFCore.interfaces import IContentish

logger = logging.getLogger("collective.vectorsearch")


def _get_vector_index_and_rid(obj):
    """Get VectorIndex and RID for an object.

    Returns:
        tuple: (vector_index, rid) or (None, None) if not available
    """
    try:
        from plone import api

        catalog = api.portal.get_tool("portal_catalog")
        if "llm_vector" not in catalog.Indexes:
            return None, None

        vector_index = catalog.Indexes["llm_vector"]

        # Get the document ID (RID) for this object
        path = "/".join(obj.getPhysicalPath())
        rid = catalog.getrid(path)

        return vector_index, rid
    except Exception as e:
        logger.debug(f"Could not get vector index: {e}")
        return None, None


# ITQ Hash Indexer (metadata column)


@indexer(IContentish)
def itq_hashes(obj):
    """Indexer for ITQ hashes of all chunks.

    Returns:
        list: List of (high_64bit, low_64bit) tuples, one per chunk
              Returns None if no hashes available
    """
    vector_index, rid = _get_vector_index_and_rid(obj)
    path = "/".join(obj.getPhysicalPath()) if hasattr(obj, "getPhysicalPath") else "unknown"
    logger.info(f"itq_hashes indexer called for {path}, rid={rid}")

    if vector_index is None or rid is None:
        logger.info(f"itq_hashes: vector_index={vector_index is not None}, rid={rid} - returning None")
        return None

    hashes = vector_index.getITQHashes(rid)
    logger.info(f"itq_hashes: got {len(hashes) if hashes else 0} hashes for rid={rid}")
    if not hashes:
        return None

    return hashes


# Pivot Distance Indexers (KeywordIndex - returns list of values)


def _get_pivot_distances_for_index(obj, pivot_index):
    """Get all pivot distances for a specific pivot index.

    Args:
        obj: Content object
        pivot_index: 0-based pivot index (0-7)

    Returns:
        list or tuple: List of integer distances for all chunks
              Returns None if no distances available
    """
    vector_index, rid = _get_vector_index_and_rid(obj)
    if vector_index is None or rid is None:
        return None

    distances = vector_index.getPivotDistancesForIndex(rid, pivot_index)
    if pivot_index == 0:  # Only log for pivot1 to reduce noise
        path = "/".join(obj.getPhysicalPath()) if hasattr(obj, "getPhysicalPath") else "unknown"
        logger.info(f"pivot1 indexer: rid={rid}, distances={distances}")
    if not distances:
        return None

    return distances


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
    import numpy as np

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
