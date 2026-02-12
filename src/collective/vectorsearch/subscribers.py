"""Event subscribers for vector search data computation.

This module provides event subscribers that compute and store vector search data
(embeddings, ITQ hashes, pivot distances) in content object annotations when
content is created or modified.

The data flow is:
1. Content is created or modified
2. Event subscriber is triggered
3. Embeddings are computed using the configured model
4. ITQ hashes and pivot distances are computed
5. All data is stored in content annotations
6. Catalog indexers can then read from annotations
"""

import logging

from plone import api
from zope.annotation.interfaces import IAnnotations
from zope.component import queryUtility

from collective.vectorsearch.annotations import (
    clear_vector_data,
    set_vector_data,
)
from collective.vectorsearch.indexers import binary_hash_to_integers
from collective.vectorsearch.interfaces import IEmbeddingModelProvider

logger = logging.getLogger("collective.vectorsearch")


def _get_settings():
    """Get vector search settings from registry.

    Returns:
        dict with settings, or defaults if registry not available
    """
    try:
        registry = api.portal.get_registry_record
        return {
            "embedding_model": registry(
                "collective.vectorsearch.embedding_model", default="all-minilm-l6"
            ),
            "embedding_chunk_size": registry(
                "collective.vectorsearch.embedding_chunk_size", default=500
            ),
        }
    except Exception as e:
        logger.debug(f"Could not read registry: {e}")
        return {
            "embedding_model": "all-minilm-l6",
            "embedding_chunk_size": 500,
        }


def _get_searchable_text(obj):
    """Extract searchable text from content object.

    Args:
        obj: Content object

    Returns:
        str: Searchable text, or None if not available
    """
    try:
        # Try to use plone.app.contenttypes indexer
        from plone.app.contenttypes.indexers import SearchableText

        text = SearchableText(obj)
        if text:
            return text
    except (ImportError, Exception) as e:
        logger.debug(f"Could not get SearchableText from indexer: {e}")

    # Fallback: try direct attribute access
    text_parts = []

    # Try common text attributes
    for attr in ["title", "description", "text"]:
        value = getattr(obj, attr, None)
        if value:
            # Handle RichTextValue
            if hasattr(value, "output"):
                text_parts.append(value.output)
            elif hasattr(value, "raw"):
                text_parts.append(value.raw)
            elif isinstance(value, str):
                text_parts.append(value)

    if text_parts:
        return " ".join(text_parts)

    return None


def compute_and_store_vectors(obj):
    """Compute embeddings, ITQ hashes, pivot distances and store in annotations.

    Args:
        obj: Content object

    Returns:
        int: Number of chunks computed, or 0 if failed
    """
    # Get settings
    settings = _get_settings()
    model_id = settings.get("embedding_model", "all-minilm-l6")
    chunk_size = settings.get("embedding_chunk_size", 500)

    # Get model provider
    model_provider = queryUtility(IEmbeddingModelProvider, name=model_id)
    if model_provider is None:
        logger.warning(f"Model provider '{model_id}' not found")
        model_provider = queryUtility(IEmbeddingModelProvider, name="all-minilm-l6")
        if model_provider is None:
            logger.error("No embedding model provider available")
            return 0

    # Get text from content
    text = _get_searchable_text(obj)
    if not text:
        logger.debug(f"No searchable text for {obj}")
        clear_vector_data(obj)
        return 0

    # Get prefixes from model provider
    prefix_query = getattr(model_provider, "query_prefix", None)
    prefix_passage = getattr(model_provider, "passage_prefix", None)

    # Get embedding instance from provider
    try:
        embedding_instance = model_provider.get_embedding_instance(
            chunk_size=chunk_size,
            prefix_query=prefix_query,
            prefix_passage=prefix_passage,
        )
    except Exception as e:
        logger.error(f"Failed to get embedding instance: {e}")
        return 0

    # Compute embeddings
    try:
        vectors = embedding_instance.embed(text)
    except Exception as e:
        logger.error(f"Failed to embed text for {obj}: {e}")
        return 0

    num_chunks = len(vectors)
    logger.debug(f"Computed {num_chunks} embedding chunks for {obj}")

    # Compute ITQ hashes
    itq_hashes_list = []
    if hasattr(model_provider, "get_itq_boundary"):
        try:
            itq_data = model_provider.get_itq_boundary()
            if itq_data is not None:
                for vector in vectors:
                    binary_hash = itq_data.compute_hash(vector)
                    high, low = binary_hash_to_integers(binary_hash)
                    itq_hashes_list.append((high, low))
        except Exception as e:
            logger.warning(f"Failed to compute ITQ hashes for {obj}: {e}")

    # Compute pivot distances
    pivot_distances_list = []
    if hasattr(model_provider, "get_pivot_data"):
        try:
            pivot_data = model_provider.get_pivot_data()
            if pivot_data is not None:
                for vector in vectors:
                    distances = pivot_data.compute_distances(vector)
                    # Convert to integers (distance * 1000)
                    int_distances = tuple(int(round(d * 1000)) for d in distances)
                    pivot_distances_list.append(int_distances)
        except Exception as e:
            logger.warning(f"Failed to compute pivot distances for {obj}: {e}")

    # Store in annotations
    try:
        # Convert numpy arrays to lists for ZODB storage
        vectors_list = vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)

        set_vector_data(
            obj,
            vectors=vectors_list,
            itq_hashes=itq_hashes_list if itq_hashes_list else None,
            pivot_distances=pivot_distances_list if pivot_distances_list else None,
            model_id=model_id,
        )
        logger.debug(
            f"Stored vector data for {obj}: {num_chunks} vectors, "
            f"{len(itq_hashes_list)} ITQ hashes, "
            f"{len(pivot_distances_list)} pivot distances"
        )
    except Exception as e:
        logger.error(f"Failed to store vector data for {obj}: {e}")
        return 0

    return num_chunks


def content_added(obj, event):
    """Event handler for content added.

    Computes and stores embeddings when new content is created.
    """
    try:
        # Skip if object is not annotatable
        IAnnotations(obj)
    except TypeError:
        logger.debug(f"Object {obj} is not annotatable, skipping")
        return

    path = (
        "/".join(obj.getPhysicalPath()) if hasattr(obj, "getPhysicalPath") else str(obj)
    )
    logger.debug(f"Content added: {path}")

    compute_and_store_vectors(obj)


def content_modified(obj, event):
    """Event handler for content modified.

    Recomputes and stores embeddings when content is modified.
    """
    try:
        # Skip if object is not annotatable
        IAnnotations(obj)
    except TypeError:
        logger.debug(f"Object {obj} is not annotatable, skipping")
        return

    path = (
        "/".join(obj.getPhysicalPath()) if hasattr(obj, "getPhysicalPath") else str(obj)
    )
    logger.debug(f"Content modified: {path}")

    compute_and_store_vectors(obj)
