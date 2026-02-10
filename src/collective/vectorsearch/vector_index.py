import time
from logging import getLogger

from AccessControl.class_init import InitializeClass
from AccessControl.Permissions import search_zcatalog
from AccessControl.SecurityInfo import ClassSecurityInfo
from Acquisition import Implicit
from App.special_dtml import DTMLFile
from BTrees.IIBTree import IIBucket
from BTrees.Length import Length
from BTrees.OOBTree import OOBTree
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

from collective.vectorsearch.annotations import (
    clear_vector_data,
    has_vector_data,
)
from collective.vectorsearch.annotations import (
    get_itq_hashes as get_itq_hashes_from_annotations,
)
from collective.vectorsearch.annotations import (
    get_model_id as get_model_id_from_annotations,
)
from collective.vectorsearch.annotations import (
    get_pivot_distances as get_pivot_distances_from_annotations,
)
from collective.vectorsearch.annotations import (
    get_vectors as get_vectors_from_annotations,
)
from collective.vectorsearch.interfaces import IEmbeddingModelProvider, IVectorIndex
from collective.vectorsearch.similarity_algorithm import CosineSimilarityAlgorithm
from collective.vectorsearch.subscribers import compute_and_store_vectors

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
            # Try local-only first to avoid HTTP requests to HuggingFace Hub.
            # Falls back to network access if local loading fails for any
            # reason (not downloaded, corrupted cache, JSON parse errors, etc.)
            try:
                self._models[model_name] = SentenceTransformer(
                    model_name, local_files_only=True
                )
            except Exception:
                logger.info(
                    f"Model '{model_name}' local load failed, trying with network access..."
                )
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

        # Document tracking (docid -> path mapping)
        self._docid_to_path = OOBTree()
        self.length = Length()
        self.document_count = Length()
        # Track which model was used to create vectors (None = no vectors yet)
        self.indexed_with_model = None

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
                    "collective.vectorsearch.pivot_threshold", default=200
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
            "pivot_threshold": 200,
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
        """Index a content object.

        This method:
        1. Computes embeddings and stores them in annotations (if not already present,
           or if the embedding model has changed since last computation)
        2. Calls index_doc() to register the document in VectorIndex

        By computing and storing in annotations HERE (before index_doc and before
        other indexers like pivot1-8 run), we ensure that annotation data is
        available for all indexers within the same catalog_object() call.
        """
        # Step 1: Ensure annotations are populated with vector data
        # This must happen BEFORE index_doc() and before pivot/itq indexers
        try:
            needs_compute = not has_vector_data(obj)

            if not needs_compute:
                # Check if the model has changed since last computation
                annotation_model = get_model_id_from_annotations(obj)
                settings = self._get_settings()
                current_model = settings.get("embedding_model", "all-minilm-l6")
                if annotation_model and annotation_model != current_model:
                    needs_compute = True

            if needs_compute:
                compute_and_store_vectors(obj)
        except Exception as e:
            logger.warning(
                "Failed to compute vectors for document %s: %s", documentId, e
            )

        # Step 2: Index the document (reads from annotations)
        count = self.index_doc(documentId, obj)
        return count  # Number of vector rows

    def index_doc(self, docid, obj_or_text):
        """Index a document using pre-computed vectors from annotations.

        Reads vectors from content annotations (pre-computed by index_object
        or event subscriber). Updates internal counts and path mapping.

        Args:
            docid: Document ID (RID from catalog)
            obj_or_text: Content object or text string.
                         If a content object, reads annotations directly.
                         If a text string, looks up the object by docid.

        Returns:
            int: Number of vector chunks indexed, or 0 if no vectors found
        """
        # Determine the content object
        if hasattr(obj_or_text, "getPhysicalPath"):
            obj = obj_or_text
        else:
            obj = self._get_object_for_docid(docid)

        if obj is None:
            logger.debug(f"Could not find object for docid {docid}")
            return 0

        # Read vectors from annotations
        vectors_list = get_vectors_from_annotations(obj)
        if not vectors_list:
            logger.debug(f"No vectors in annotations for docid {docid}")
            return 0

        # Convert lists to numpy array
        vectors = np.array(vectors_list)
        if vectors.size == 0:
            return 0

        # Get object path for tracking
        path = "/".join(obj.getPhysicalPath())

        # Update counts - handle existing document
        if hasattr(self, "_docid_to_path") and docid in self._docid_to_path:
            # Get old chunk count from metadata or use new count as estimate
            old_row = len(vectors_list)
            try:
                catalog = api.portal.get_tool("portal_catalog")
                metadata = catalog.getMetadataForRID(docid)
                old_vectors = metadata.get("llm_vector") if metadata else None
                if old_vectors:
                    old_row = len(old_vectors)
            except Exception:
                pass
            self._change_length("document_count", -1)
            self._change_length("length", -old_row)

        # Update counts with new data
        row = len(vectors_list)
        self._change_length("document_count", 1)
        self._change_length("length", row)

        # Ensure _docid_to_path exists
        if not hasattr(self, "_docid_to_path"):
            self._docid_to_path = OOBTree()

        # Store path mapping
        self._docid_to_path[docid] = path

        # Track which model was used
        model_id = get_model_id_from_annotations(obj)
        if model_id:
            self.indexed_with_model = model_id

        return row

    def _get_object_for_docid(self, docid):
        """Get content object from document ID (RID).

        Args:
            docid: Document ID (RID from catalog)

        Returns:
            Content object, or None if not found
        """
        try:
            catalog = api.portal.get_tool("portal_catalog")
            path = catalog.getpath(docid)
            portal = api.portal.get()
            obj = portal.unrestrictedTraverse(path, None)
            return obj
        except Exception as e:
            logger.debug(f"Could not get object for docid {docid}: {e}")
            return None

    def _clear_itq_pivot_data(self):
        """Clear all ITQ hashes and pivot distances from annotations.

        Called when the embedding model changes, as the previous
        ITQ/pivot data is no longer valid.
        """
        if hasattr(self, "_docid_to_path"):
            portal = api.portal.get()
            for _docid, path in self._docid_to_path.items():
                try:
                    obj = portal.unrestrictedTraverse(path, None)
                    if obj:
                        clear_vector_data(obj)
                except Exception:
                    continue
        logger.info("Cleared ITQ hashes and pivot distances due to model change")

    security.declareProtected(manage_zcatalog_indexes, "unindex_object")

    def unindex_object(self, docid):
        old_chunk_count = 0

        # Get chunk count from catalog metadata
        if hasattr(self, "_docid_to_path") and docid in self._docid_to_path:
            try:
                catalog = api.portal.get_tool("portal_catalog")
                metadata = catalog.getMetadataForRID(docid)
                if metadata:
                    vectors_list = metadata.get("llm_vector")
                    if vectors_list:
                        old_chunk_count = len(vectors_list)
            except Exception:
                pass

            # Remove from path mapping
            try:
                del self._docid_to_path[docid]
            except KeyError:
                pass

        # Update counts
        if old_chunk_count > 0:
            self._change_length("document_count", -1)
            self._change_length("length", -old_chunk_count)

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
        """Get all document vectors for search.

        Reads vectors from catalog metadata (no object traversal).

        Returns:
            tuple: (docids, vectors) numpy arrays
        """
        all_docids = []
        all_vectors = []

        if hasattr(self, "_docid_to_path") and len(self._docid_to_path) > 0:
            try:
                catalog = api.portal.get_tool("portal_catalog")
            except Exception:
                catalog = None

            for docid in self._docid_to_path:
                try:
                    vectors_list = None
                    if catalog:
                        metadata = catalog.getMetadataForRID(docid)
                        if metadata:
                            vectors_list = metadata.get("llm_vector")
                    if vectors_list:
                        vectors = np.array(vectors_list)
                        all_docids.extend([docid] * len(vectors))
                        all_vectors.append(vectors)
                except Exception as e:
                    logger.debug(f"Could not get vectors for docid {docid}: {e}")
                    continue

        if not all_vectors:
            return np.array([], dtype=int), np.array([]).reshape(0, 0)

        vectors = np.concatenate(all_vectors)
        docids = np.array(all_docids, dtype=int)
        return docids, vectors

    security.declareProtected(search_zcatalog, "getEntryForObject")

    def getEntryForObject(self, documentId, default=None):
        """Get the index entry for a specific document.

        Reads vectors from catalog metadata.

        Returns the vector embedding for the document if it exists.
        """
        start_time = time.perf_counter()
        result = None

        if hasattr(self, "_docid_to_path") and documentId in self._docid_to_path:
            try:
                catalog = api.portal.get_tool("portal_catalog")
                metadata = catalog.getMetadataForRID(documentId)
                if metadata:
                    vectors_list = metadata.get("llm_vector")
                    if vectors_list:
                        result = np.array(vectors_list)
            except Exception as e:
                logger.debug(f"Could not get vectors from metadata: {e}")

        if result is None:
            result = default

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
        """Clear all index data."""
        self._docid_to_path = OOBTree()
        self.length = Length()
        self.document_count = Length()
        self.indexed_with_model = None

    security.declareProtected(search_zcatalog, "getITQHashes")

    def getITQHashes(self, docid):
        """Get the ITQ hashes for all chunks of a document.

        Reads from content annotations.

        Args:
            docid: Document ID (RID)

        Returns:
            tuple: List of (high_64bit, low_64bit) tuples, one per chunk
                   Returns None if not found
        """
        if hasattr(self, "_docid_to_path") and docid in self._docid_to_path:
            path = self._docid_to_path[docid]
            try:
                portal = api.portal.get()
                obj = portal.unrestrictedTraverse(path, None)
                if obj:
                    hashes = get_itq_hashes_from_annotations(obj)
                    if hashes:
                        return tuple(hashes)
            except Exception as e:
                logger.debug(f"Could not get ITQ hashes from annotations: {e}")
        return None

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

        Reads from content annotations.

        Args:
            docid: Document ID (RID)

        Returns:
            tuple: List of (d1, d2, ..., d8) tuples, one per chunk
                   Returns None if not found
        """
        if hasattr(self, "_docid_to_path") and docid in self._docid_to_path:
            path = self._docid_to_path[docid]
            try:
                portal = api.portal.get()
                obj = portal.unrestrictedTraverse(path, None)
                if obj:
                    distances = get_pivot_distances_from_annotations(obj)
                    if distances:
                        return tuple(distances)
            except Exception as e:
                logger.debug(f"Could not get pivot distances from annotations: {e}")
        return None

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

        Reads stats from catalog indexes and metadata (no object traversal).

        Returns:
            dict: Statistics including counts and consistency info
        """
        itq_docs = 0
        itq_chunks = 0
        pivot_docs = 0
        pivot_chunks = 0

        try:
            catalog = api.portal.get_tool("portal_catalog")

            # Pivot stats from KeywordIndex (pivot1)
            if "pivot1" in catalog.Indexes:
                pivot_index = catalog.Indexes["pivot1"]
                pivot_docs = pivot_index.numObjects()
                if hasattr(pivot_index, "_unindex"):
                    for values in pivot_index._unindex.values():
                        if isinstance(values, (list, tuple, set)):
                            pivot_chunks += len(values)
                        else:
                            pivot_chunks += 1

            # ITQ stats from catalog metadata column
            if hasattr(self, "_docid_to_path"):
                for docid in self._docid_to_path:
                    try:
                        metadata = catalog.getMetadataForRID(docid)
                        itq_value = metadata.get("itq_hashes")
                        if itq_value:
                            itq_docs += 1
                            itq_chunks += len(itq_value)
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"Could not get stats from catalog: {e}")

        stats = {
            "documents": self.document_count()
            if hasattr(self, "document_count")
            else 0,
            "vectors": self.length() if hasattr(self, "length") else 0,
            "itq_hashes": itq_docs,
            "itq_hashes_chunks": itq_chunks,
            "pivot_distances": pivot_docs,
            "pivot_distances_chunks": pivot_chunks,
            "indexed_model": self.getIndexedModel(),
            "model_consistent": self.isModelConsistent(),
        }

        # Check if ITQ/pivot data is available for current model.
        # Use queryUtility() directly to avoid triggering _ensure_initialized()
        # which would load the heavy SentenceTransformer model just for stats.
        try:
            settings = self._get_settings()
            model_id = settings.get("embedding_model", "all-minilm-l6")
            provider = queryUtility(IEmbeddingModelProvider, name=model_id)
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
            stats["itq_data_available"] = itq_docs > 0
            stats["pivot_data_available"] = pivot_docs > 0

        return stats

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
