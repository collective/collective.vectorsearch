import time
from logging import getLogger

from AccessControl.class_init import InitializeClass
from AccessControl.Permissions import search_zcatalog
from AccessControl.SecurityInfo import ClassSecurityInfo
from Acquisition import Implicit
from App.special_dtml import DTMLFile
from BTrees.IIBTree import IIBucket
from BTrees.IOBTree import IOBTree
from BTrees.Length import Length
from OFS.SimpleItem import SimpleItem
from Persistence import Persistent
from Products.PluginIndexes.interfaces import IQueryIndex
from Products.ZCatalog.ZCatalog import manage_zcatalog_indexes
from zope.interface import implementer

try:
    from plone.app.contenttypes.indexers import SearchableText
except ImportError:
    SearchableText = None

import numpy as np
from plone import api

from collective.vectorsearch.interfaces import IEmbeddingModelProvider, IVectorIndex
from collective.vectorsearch.similarity_algorithm import CosineSimilarityAlgorithm

# Optional: sentence_transformers for GPU support
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False
from zope.component import queryUtility

logger = getLogger("collective.vectorsearch")


class ModelCache:
    """Singleton cache for SentenceTransformer models.

    This prevents loading the same model multiple times into memory,
    which can be very expensive (100MB+ per model).

    Note: Requires the 'gpu' extras to be installed (sentence_transformers).
    """

    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    def get_model(self, model_name):
        """Get a cached model or load it if not already cached.

        Args:
            model_name: Name of the SentenceTransformer model

        Returns:
            SentenceTransformer model instance

        Raises:
            ImportError: If sentence_transformers is not installed
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence_transformers is not installed. "
                "Install with: pip install collective.vectorsearch[gpu]"
            )

        if model_name not in self._models:
            logger.info(f"Loading model '{model_name}' into cache")
            self._models[model_name] = SentenceTransformer(model_name)
        else:
            logger.debug(f"Using cached model '{model_name}'")
        return self._models[model_name]

    def clear_cache(self):
        """Clear all cached models. Useful for testing or memory management."""
        logger.info("Clearing model cache")
        self._models.clear()

    def get_cache_info(self):
        """Get information about cached models.

        Returns:
            dict: Model names and their memory status
        """
        return {
            "cached_models": list(self._models.keys()),
            "model_count": len(self._models),
        }

    @staticmethod
    def is_available():
        """Check if SentenceTransformer support is available."""
        return HAS_SENTENCE_TRANSFORMERS


@implementer(IVectorIndex, IQueryIndex)
class VectorIndex(Persistent, Implicit, SimpleItem):
    """ """

    meta_type = "VectorIndex"
    operators = ("and", "or")
    useOperator = "or"
    query_options = ("query",)

    manage_options = ({"label": "Settings", "action": "manage_main"},)

    manage = manage_main = DTMLFile("dtml/manageVectorIndex", globals())
    manage_main._setName("manage_main")

    security = ClassSecurityInfo()
    # security.declareObjectProtected(manage_zcatalog_indexes)

    def __init__(self, id, extra=None, *args, **kwargs):
        self.id = id
        self._docvectors = IOBTree()
        self.length = Length()
        self.document_count = Length()
        # Track which model was used to create vectors (None = no vectors yet)
        self.indexed_with_model = None

        # Storage for ITQ hashes (128-bit as two 64-bit integers)
        # Key: docid, Value: (high_64bit, low_64bit)
        self._itq_hashes = IOBTree()

        # Storage for pivot distances (8 integer values per document)
        # Key: docid, Value: tuple of 8 integers (distance * 1000)
        self._pivot_distances = IOBTree()

        # Handle indexed_attrs from extra parameter
        if extra is not None and isinstance(extra, dict):
            indexed_attrs = extra.get("indexed_attrs", "")
            if indexed_attrs:
                if isinstance(indexed_attrs, str):
                    self.indexed_attrs = [
                        attr.strip() for attr in indexed_attrs.split(",")
                    ]
                else:
                    self.indexed_attrs = indexed_attrs
            else:
                self.indexed_attrs = [self.id]
        else:
            self.indexed_attrs = [self.id]

        # Lazy initialization flags - model/embedding loaded on first use
        # Use _v_ prefix for volatile attributes (not persisted to ZODB)
        # ONNX sessions and model objects cannot be pickled
        self._v_embedding = None
        self._v_model_provider = None
        self._v_similarity_algorithm = None
        self._v_itq_boundary = None
        self._v_pivot_data = None

    def _ensure_initialized(self):
        """Lazy initialization of embedding model and similarity algorithm.

        This is called on first use (index_doc or query_index) to avoid
        loading heavy ML models during Quickinstall.

        Note: Volatile attributes (_v_*) are not persisted to ZODB, so we
        must use getattr() with defaults when checking them.
        """
        if getattr(self, "_v_embedding", None) is not None:
            return

        settings = self._get_settings()
        model_id = settings.get("embedding_model", "all-minilm-l6")
        chunk_size = settings.get("embedding_chunk_size", 500)
        approx_algo = settings.get("approximation_algorithm", "exhaustive_cosine")

        # Get model provider utility
        model_provider = queryUtility(IEmbeddingModelProvider, name=model_id)

        if model_provider is None:
            logger.warning(
                f"Model provider '{model_id}' not found, using all-minilm-l6"
            )
            model_provider = queryUtility(IEmbeddingModelProvider, name="all-minilm-l6")

        self._v_model_provider = model_provider

        # Get prefixes from model provider
        prefix_query = getattr(model_provider, "query_prefix", None)
        prefix_passage = getattr(model_provider, "passage_prefix", None)

        # Get embedding instance from provider
        self._v_embedding = model_provider.get_embedding_instance(
            chunk_size=chunk_size,
            prefix_query=prefix_query,
            prefix_passage=prefix_passage,
        )

        # Load ITQ data if needed (Phase 2: not implemented yet)
        if approx_algo in ("itq_lsh_2stage", "itq_lsh_3stage"):
            self._v_itq_boundary = model_provider.get_itq_boundary()
            self._v_pivot_data = model_provider.get_pivot_data()

        # Initialize similarity algorithm
        self._v_similarity_algorithm = CosineSimilarityAlgorithm()

    @property
    def embedding(self):
        """Lazy-loaded embedding instance."""
        self._ensure_initialized()
        return self._v_embedding

    @property
    def model_provider(self):
        """Lazy-loaded model provider."""
        self._ensure_initialized()
        return self._v_model_provider

    @property
    def similarity_algorithm(self):
        """Lazy-loaded similarity algorithm."""
        self._ensure_initialized()
        return self._v_similarity_algorithm

    def _get_settings(self):
        """Retrieve all configuration from registry."""
        try:
            registry = api.portal.get_registry_record

            settings = {
                "embedding_model": registry(
                    "collective.vectorsearch.embedding_model", default="all-minilm-l6"
                ),
                "embedding_chunk_size": registry(
                    "collective.vectorsearch.embedding_chunk_size", default=500
                ),
                "storage_backend": registry(
                    "collective.vectorsearch.storage_backend", default="btrees"
                ),
                "external_db_uri": registry(
                    "collective.vectorsearch.external_db_uri", default=""
                ),
                "approximation_algorithm": registry(
                    "collective.vectorsearch.approximation_algorithm",
                    default="exhaustive_cosine",
                ),
                "pivot_threshold": registry(
                    "collective.vectorsearch.pivot_threshold", default=20
                ),
                "hamming_distance_threshold": registry(
                    "collective.vectorsearch.hamming_distance_threshold", default=3
                ),
            }

            # Log current implementation status
            self._log_implementation_status(settings)

            return settings

        except Exception as e:
            logger.warning(f"Could not read registry: {e}")
            return self._get_default_settings()

    def _log_implementation_status(self, settings):
        """Log warnings for features not yet implemented."""

        # Storage backend implementation status
        implemented_backends = ["btrees"]
        if settings["storage_backend"] not in implemented_backends:
            logger.warning(
                f"Storage backend '{settings['storage_backend']}' is not yet implemented. "
                f"Current implementation: {', '.join(implemented_backends)}"
            )

        # Approximation algorithm implementation status
        implemented_algorithms = ["exhaustive_cosine"]
        if settings["approximation_algorithm"] not in implemented_algorithms:
            logger.warning(
                f"Approximation algorithm '{settings['approximation_algorithm']}' is not yet implemented. "
                f"Current implementation: {', '.join(implemented_algorithms)}"
            )

    def _get_default_settings(self):
        """Fallback default settings."""
        return {
            "embedding_model": "all-minilm-l6",
            "embedding_chunk_size": 500,
            "storage_backend": "btrees",
            "external_db_uri": "",
            "approximation_algorithm": "exhaustive_cosine",
            "pivot_threshold": 20,
            "hamming_distance_threshold": 3,
        }

    def _change_length(self, name, value):
        length_obj = getattr(self, name, None)
        if length_obj is not None:
            length_obj.change(value)
        else:
            setattr(self, name, Length(value))

    security.declareProtected(manage_zcatalog_indexes, "index_object")

    def index_object(self, documentId, obj, threshold=None):
        count = 0
        if SearchableText is not None:
            try:
                text = SearchableText(obj)
                row = self.index_doc(documentId, text)
                count += row
            except Exception as e:
                logger.warning(
                    "Failed to index SearchableText for document %s: %s", documentId, e
                )
        fields = self.getIndexSourceNames()
        for field in fields:
            try:
                value = getattr(obj, field, None)
                if value is not None:
                    row = self.index_doc(documentId, value)
                    count += row
            except Exception as e:
                logger.warning(
                    "Failed to index field '%s' for document %s: %s",
                    field,
                    documentId,
                    e,
                )
        return count  # Number of vector rows

    def index_doc(self, docid, text):
        # Skip empty or invalid text
        if not text or not isinstance(text, str):
            logger.debug("Skipping empty or invalid text for document %s", docid)
            return 0

        # Try to embed FIRST, before modifying any data
        # This ensures we don't corrupt counts if embedding fails
        try:
            vectors = self.embedding.embed(text)
        except Exception as e:
            logger.error("Failed to embed text for document %s: %s", docid, e)
            # Don't touch existing data if embedding fails
            return 0

        # Only update counts AFTER successful embedding
        old_vectors = self._docvectors.get(docid, None)
        if old_vectors is not None:
            self._change_length("document_count", -1)
            old_row, old_col = old_vectors.shape
            self._change_length("length", -old_row)

        row, col = vectors.shape
        self._change_length("document_count", 1)
        self._change_length("length", row)
        self._docvectors[docid] = vectors

        # Track which model was used and detect model changes
        settings = self._get_settings()
        current_model = settings.get("embedding_model", "all-minilm-l6")
        previous_model = getattr(self, "indexed_with_model", None)

        if previous_model is None:
            # First document being indexed
            self.indexed_with_model = current_model
        elif previous_model != current_model:
            # Model has changed - clear ITQ/pivot data and update tracking
            logger.warning(
                f"Embedding model changed from '{previous_model}' to '{current_model}'. "
                "ITQ/pivot data has been cleared. Consider running a full reindex."
            )
            self._clear_itq_pivot_data()
            self.indexed_with_model = current_model

        # Compute and store ITQ hash and pivot distances for all chunks
        self._compute_and_store_itq_pivot_all(docid, vectors)

        return row

    def _clear_itq_pivot_data(self):
        """Clear all ITQ hashes and pivot distances.

        Called when the embedding model changes, as the previous
        ITQ/pivot data is no longer valid.
        """
        if hasattr(self, "_itq_hashes"):
            self._itq_hashes = IOBTree()
        if hasattr(self, "_pivot_distances"):
            self._pivot_distances = IOBTree()
        logger.info("Cleared ITQ hashes and pivot distances due to model change")

    def _compute_and_store_itq_pivot_all(self, docid, vectors):
        """Compute and store ITQ hashes and pivot distances for all chunks.

        Args:
            docid: Document ID
            vectors: numpy array of embedding vectors (one per chunk)
        """
        # Ensure ITQ/pivot data BTrees exist (for existing indexes without them)
        if not hasattr(self, "_itq_hashes"):
            self._itq_hashes = IOBTree()
        if not hasattr(self, "_pivot_distances"):
            self._pivot_distances = IOBTree()

        # Get ITQ data from model provider
        # Use hasattr to support older model providers without these methods
        if self.model_provider is None:
            logger.warning(f"model_provider is None for doc {docid}, skipping ITQ/pivot")
            return

        provider_class = self.model_provider.__class__.__name__
        num_chunks = len(vectors)
        logger.info(f"Computing ITQ/pivot for doc {docid} ({num_chunks} chunks) using {provider_class}")

        # Compute ITQ hashes for all chunks
        itq_hashes_list = []
        if hasattr(self.model_provider, "get_itq_boundary"):
            itq_data = self.model_provider.get_itq_boundary()
            if itq_data is not None:
                for i, vector in enumerate(vectors):
                    try:
                        binary_hash = itq_data.compute_hash(vector)
                        high, low = self._binary_hash_to_integers(binary_hash)
                        itq_hashes_list.append((high, low))
                    except Exception as e:
                        logger.warning(f"Failed to compute ITQ hash for doc {docid} chunk {i}: {e}")
                if itq_hashes_list:
                    self._itq_hashes[docid] = tuple(itq_hashes_list)
                    logger.info(f"Stored {len(itq_hashes_list)} ITQ hashes for doc {docid}")
            else:
                logger.warning(f"ITQ data is None for {provider_class}")
        else:
            logger.warning(f"{provider_class} has no get_itq_boundary method")

        # Compute pivot distances for all chunks
        pivot_distances_list = []
        if hasattr(self.model_provider, "get_pivot_data"):
            pivot_data = self.model_provider.get_pivot_data()
            if pivot_data is not None:
                for i, vector in enumerate(vectors):
                    try:
                        distances = pivot_data.compute_distances(vector)
                        # Convert to integers (distance * 1000)
                        int_distances = tuple(int(round(d * 1000)) for d in distances)
                        pivot_distances_list.append(int_distances)
                    except Exception as e:
                        logger.warning(f"Failed to compute pivot distances for doc {docid} chunk {i}: {e}")
                if pivot_distances_list:
                    self._pivot_distances[docid] = tuple(pivot_distances_list)
                    logger.info(f"Stored pivot distances for {len(pivot_distances_list)} chunks of doc {docid}")
            else:
                logger.warning(f"Pivot data is None for {provider_class}")
        else:
            logger.warning(f"{provider_class} has no get_pivot_data method")

    def _binary_hash_to_integers(self, binary_hash):
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

    security.declareProtected(manage_zcatalog_indexes, "unindex_object")

    def unindex_object(self, docid):
        old_vectors = self._docvectors.get(docid, None)
        if old_vectors is not None:
            self._change_length("document_count", -1)
            old_row, old_col = old_vectors.shape
            self._change_length("length", -old_row)
            try:
                del self._docvectors[docid]
            except KeyError:
                logger.warning(
                    "Document %s not found in index during unindexing", docid
                )

        # Also remove ITQ hash and pivot distances
        if hasattr(self, "_itq_hashes") and docid in self._itq_hashes:
            try:
                del self._itq_hashes[docid]
            except KeyError:
                pass

        if hasattr(self, "_pivot_distances") and docid in self._pivot_distances:
            try:
                del self._pivot_distances[docid]
            except KeyError:
                pass

    def _apply_index(self, request):
        """Apply the index to a search request.

        Currently not implemented for vector search.
        Use query_index() for vector similarity search instead.
        """
        start_time = time.perf_counter()
        logger.debug("_apply_index called with request: %s", request)
        elapsed = time.perf_counter() - start_time
        logger.debug("_apply_index completed in %.4f seconds", elapsed)

    @security.protected(search_zcatalog)
    def query(self, query, nbest=10):
        """Query the index (legacy interface).

        Returns empty list. Use query_index() for vector similarity search.
        """
        start_time = time.perf_counter()
        logger.debug("query called with query=%s, nbest=%s", query, nbest)
        elapsed = time.perf_counter() - start_time
        logger.debug("query completed in %.4f seconds", elapsed)
        return []

    security.declareProtected(search_zcatalog, "query_index")

    def query_index(self, record, resultset=None):
        query_str = " ".join(record.keys)
        if not query_str:
            return None
        query = self.embedding.embed(query_str, query=True)
        docids, vectors = self._get_all_doc_vectors()
        indices, scores = self.similarity_algorithm(vectors, query)
        bucket = IIBucket()
        for docid, score in zip(docids[indices], scores):
            int_docid = int(docid)
            if int_docid in bucket:
                pass
                # bucket[int_docid] += int(score * 100_000_000)
            else:
                # Convert float score to integer for Zope catalog compatibility.
                # Multiply by 100,000,000 to preserve 8 decimal places of precision.
                # Zope's IIBucket requires integer values for scoring.
                bucket[int_docid] = int(score * 100_000_000)
        return bucket

    def _get_all_doc_vectors(self):
        items = list(self._docvectors.items())
        if not items:
            # Return empty arrays if no documents are indexed
            return np.array([], dtype=int), np.array([]).reshape(0, 0)
        vectors = np.concatenate([v for k, v in items])
        docids = np.concatenate([[k] * v.shape[0] for k, v in items])
        return docids, vectors

    security.declareProtected(search_zcatalog, "getEntryForObject")

    def getEntryForObject(self, documentId, default=None):
        """Get the index entry for a specific document.

        Returns the vector embedding for the document if it exists.
        """
        start_time = time.perf_counter()
        result = self._docvectors.get(documentId, default)
        elapsed = time.perf_counter() - start_time
        logger.debug(
            "getEntryForObject: documentId=%s, found=%s, time=%.4f seconds",
            documentId,
            result is not None,
            elapsed,
        )
        return result

    security.declareProtected(search_zcatalog, "uniqueValues")

    def uniqueValues(self, name=None, withLengths=0):
        """Return unique values for the index.

        Vector indexes don't have traditional unique values like keyword indexes.
        Returns an empty tuple for compatibility with the catalog interface.
        """
        logger.debug("uniqueValues called: name=%s, withLengths=%s", name, withLengths)
        # Vector embeddings don't have discrete unique values
        # Return empty tuple for catalog compatibility
        return ()

    security.declareProtected(search_zcatalog, "numObjects")

    def numObjects(self):
        return self.document_count()

    security.declareProtected(search_zcatalog, "indexSize")

    def indexSize(self):
        return self.length()

    security.declareProtected(manage_zcatalog_indexes, "clear")

    def clear(self):
        self._docvectors = IOBTree()
        self._itq_hashes = IOBTree()
        self._pivot_distances = IOBTree()
        self.length = Length()
        self.document_count = Length()
        self.indexed_with_model = None

    security.declareProtected(search_zcatalog, "getITQHashes")

    def getITQHashes(self, docid):
        """Get the ITQ hashes for all chunks of a document.

        Args:
            docid: Document ID (RID)

        Returns:
            tuple: List of (high_64bit, low_64bit) tuples, one per chunk
                   Returns None if not found
        """
        if not hasattr(self, "_itq_hashes"):
            return None
        return self._itq_hashes.get(docid, None)

    # Legacy method for backward compatibility
    security.declareProtected(search_zcatalog, "getITQHash")

    def getITQHash(self, docid):
        """Get the first ITQ hash for a document (legacy).

        Deprecated: Use getITQHashes() for multi-chunk support.

        Args:
            docid: Document ID (RID)

        Returns:
            tuple: (high_64bit, low_64bit) of first chunk, or None if not found
        """
        hashes = self.getITQHashes(docid)
        if hashes and len(hashes) > 0:
            return hashes[0]
        return None

    security.declareProtected(search_zcatalog, "getPivotDistancesAll")

    def getPivotDistancesAll(self, docid):
        """Get the pivot distances for all chunks of a document.

        Args:
            docid: Document ID (RID)

        Returns:
            tuple: List of (d1, d2, ..., d8) tuples, one per chunk
                   Returns None if not found
        """
        if not hasattr(self, "_pivot_distances"):
            return None
        return self._pivot_distances.get(docid, None)

    security.declareProtected(search_zcatalog, "getPivotDistancesForIndex")

    def getPivotDistancesForIndex(self, docid, pivot_index):
        """Get distances to a specific pivot for all chunks.

        Args:
            docid: Document ID (RID)
            pivot_index: 0-based index of the pivot (0-7)

        Returns:
            tuple: List of integer distances for all chunks, or None if not found
        """
        all_distances = self.getPivotDistancesAll(docid)
        if all_distances is None or pivot_index < 0 or pivot_index >= 8:
            return None

        # Extract the specific pivot distance from each chunk
        result = []
        for chunk_distances in all_distances:
            if pivot_index < len(chunk_distances):
                result.append(chunk_distances[pivot_index])
        return tuple(result) if result else None

    # Legacy methods for backward compatibility
    security.declareProtected(search_zcatalog, "getPivotDistances")

    def getPivotDistances(self, docid):
        """Get the pivot distances for first chunk (legacy).

        Deprecated: Use getPivotDistancesAll() for multi-chunk support.

        Args:
            docid: Document ID (RID)

        Returns:
            tuple: 8 integer distance values for first chunk, or None if not found
        """
        all_distances = self.getPivotDistancesAll(docid)
        if all_distances and len(all_distances) > 0:
            return all_distances[0]
        return None

    security.declareProtected(search_zcatalog, "getPivotDistance")

    def getPivotDistance(self, docid, pivot_index):
        """Get a specific pivot distance for first chunk (legacy).

        Deprecated: Use getPivotDistancesForIndex() for multi-chunk support.

        Args:
            docid: Document ID (RID)
            pivot_index: 0-based index of the pivot (0-7)

        Returns:
            int: Distance value for first chunk, or None if not found
        """
        distances = self.getPivotDistances(docid)
        if distances is None or pivot_index < 0 or pivot_index >= len(distances):
            return None
        return distances[pivot_index]

    security.declareProtected(search_zcatalog, "getIndexedModel")

    def getIndexedModel(self):
        """Return the model ID used to create the indexed vectors.

        Returns:
            str or None: Model ID if vectors exist, None if index is empty
        """
        return getattr(self, "indexed_with_model", None)

    security.declareProtected(search_zcatalog, "isModelConsistent")

    def isModelConsistent(self):
        """Check if the current model matches the model used for indexing.

        Returns:
            bool: True if models match or index is empty, False if mismatch
        """
        indexed_model = self.getIndexedModel()
        if indexed_model is None:
            return True  # No data indexed yet

        settings = self._get_settings()
        current_model = settings.get("embedding_model", "all-minilm-l6")
        return indexed_model == current_model

    security.declareProtected(search_zcatalog, "getITQPivotStats")

    def getITQPivotStats(self):
        """Get statistics about ITQ hash and pivot distance storage.

        Returns:
            dict: Statistics including counts and consistency info
        """
        # Count documents and chunks for ITQ hashes
        itq_docs = 0
        itq_chunks = 0
        if hasattr(self, "_itq_hashes"):
            itq_docs = len(self._itq_hashes)
            for hashes in self._itq_hashes.values():
                if hashes:
                    itq_chunks += len(hashes)

        # Count documents and chunks for pivot distances
        pivot_docs = 0
        pivot_chunks = 0
        if hasattr(self, "_pivot_distances"):
            pivot_docs = len(self._pivot_distances)
            for distances in self._pivot_distances.values():
                if distances:
                    pivot_chunks += len(distances)

        stats = {
            "documents": self.document_count() if hasattr(self, "document_count") else 0,
            "vectors": self.length() if hasattr(self, "length") else 0,
            "itq_hashes": itq_docs,
            "itq_hashes_chunks": itq_chunks,
            "pivot_distances": pivot_docs,
            "pivot_distances_chunks": pivot_chunks,
            "indexed_model": self.getIndexedModel(),
            "model_consistent": self.isModelConsistent(),
        }

        # Check if ITQ/pivot data is available for current model
        # Wrap in try/except to handle model loading failures gracefully
        # (stats display should work even if the model can't be loaded)
        try:
            provider = self.model_provider
            if provider is not None:
                if hasattr(provider, "get_itq_boundary"):
                    stats["itq_data_available"] = (
                        provider.get_itq_boundary() is not None
                    )
                else:
                    stats["itq_data_available"] = False

                if hasattr(provider, "get_pivot_data"):
                    stats["pivot_data_available"] = (
                        provider.get_pivot_data() is not None
                    )
                else:
                    stats["pivot_data_available"] = False
            else:
                stats["itq_data_available"] = False
                stats["pivot_data_available"] = False
        except Exception as e:
            logger.debug(f"Could not check ITQ/pivot data availability: {e}")
            # If model can't be loaded, check if data exists based on stored data
            stats["itq_data_available"] = itq_docs > 0
            stats["pivot_data_available"] = pivot_docs > 0

        return stats

    security.declareProtected(manage_zcatalog_indexes, "clearAndRecomputeITQPivot")

    def clearAndRecomputeITQPivot(self):
        """Clear and recompute all ITQ hashes and pivot distances.

        This is useful when:
        - The embedding model has changed
        - ITQ/pivot data files have been updated
        - Data is inconsistent

        Returns:
            dict: Statistics about the recomputation
        """
        logger.info(f"clearAndRecomputeITQPivot called. _docvectors has {len(self._docvectors)} items")

        # Clear existing data
        self._clear_itq_pivot_data()

        # Recompute for all documents
        recomputed = 0
        errors = 0

        for docid, vectors in self._docvectors.items():
            logger.info(f"Processing docid {docid}, vectors shape: {vectors.shape if vectors is not None else 'None'}")
            if vectors is not None and len(vectors) > 0:
                try:
                    self._compute_and_store_itq_pivot_all(docid, vectors)
                    recomputed += 1
                except Exception as e:
                    logger.warning(f"Failed to recompute ITQ/pivot for doc {docid}: {e}")
                    errors += 1

        logger.info(f"Recomputed ITQ/pivot for {recomputed} documents ({errors} errors)")
        logger.info(f"After recompute: _itq_hashes has {len(self._itq_hashes)} items, _pivot_distances has {len(self._pivot_distances)} items")

        return {
            "recomputed": recomputed,
            "errors": errors,
            "total": len(self._docvectors),
        }

    security.declareProtected(search_zcatalog, "getIndexSourceNames")

    def getIndexSourceNames(self):
        """Return the list of indexed attribute names."""
        return getattr(self, "indexed_attrs", [self.id])

    security.declareProtected(search_zcatalog, "getIndexQueryNames")

    def getIndexQueryNames(self):
        return (self.id,)

    security.declareProtected(search_zcatalog, "getIndexType")

    def getIndexType(self):
        """Return the type of this index."""
        start_time = time.perf_counter()
        result = "VectorIndex"
        elapsed = time.perf_counter() - start_time
        logger.debug("getIndexType called, time=%.4f seconds", elapsed)
        return result


InitializeClass(VectorIndex)


# Note: manage_addVectorIndex is now handled by AddVectorIndexView in browser/add_vector_index.py
