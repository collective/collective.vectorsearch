import time
from logging import getLogger

from AccessControl.class_init import InitializeClass
from AccessControl.Permissions import search_zcatalog
from AccessControl.SecurityInfo import ClassSecurityInfo
from Acquisition import Implicit
from App.special_dtml import DTMLFile
from BTrees.IIBTree import IIBucket, IISet, intersection
from BTrees.Length import Length
from BTrees.OOBTree import OOBTree
from OFS.SimpleItem import SimpleItem
from Persistence import Persistent
from Products.PluginIndexes.interfaces import IQueryIndex
from Products.ZCatalog.query import IndexQuery
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
from collective.vectorsearch.indexers import (
    batch_min_hamming_distance,
    binary_hash_to_integers,
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

        Detects model/chunk_size changes: if the user changes the embedding
        model in the control panel, the cached embedding is invalidated and
        re-created without requiring a process restart.
        """
        settings = self._get_settings()
        model_id = settings.get("embedding_model", "all-minilm-l6")
        chunk_size = settings.get("embedding_chunk_size", 500)
        approx_algo = settings.get("approximation_algorithm", "exhaustive_cosine")

        if getattr(self, "_v_embedding", None) is not None:
            # Check if model or chunk_size changed since last init
            cached_model = getattr(self, "_v_model_id", None)
            cached_chunk = getattr(self, "_v_chunk_size", None)
            if cached_model is None or (
                cached_model == model_id and cached_chunk == chunk_size
            ):
                # Same model (or externally set), just ensure ITQ/pivot data
                self._ensure_itq_pivot_loaded()
                return
            # Model changed — re-initialize
            logger.info(
                "Model setting changed from '%s' to '%s', re-initializing",
                cached_model,
                model_id,
            )

        # Get model provider utility
        model_provider = queryUtility(IEmbeddingModelProvider, name=model_id)

        if model_provider is None:
            logger.warning(
                f"Model provider '{model_id}' not found, using all-minilm-l6"
            )
            model_provider = queryUtility(IEmbeddingModelProvider, name="all-minilm-l6")

        self._v_model_provider = model_provider
        self._v_model_id = model_id
        self._v_chunk_size = chunk_size

        # Get prefixes from model provider
        prefix_query = getattr(model_provider, "query_prefix", None)
        prefix_passage = getattr(model_provider, "passage_prefix", None)

        # Get embedding instance from provider
        self._v_embedding = model_provider.get_embedding_instance(
            chunk_size=chunk_size,
            prefix_query=prefix_query,
            prefix_passage=prefix_passage,
        )

        # Load ITQ data if needed
        if approx_algo in ("itq_lsh_2stage", "itq_lsh_3stage"):
            self._v_itq_boundary = model_provider.get_itq_boundary()
            self._v_pivot_data = model_provider.get_pivot_data()

        # Initialize similarity algorithm
        self._v_similarity_algorithm = CosineSimilarityAlgorithm()

    def _ensure_itq_pivot_loaded(self):
        """Load ITQ boundary and pivot data if the algorithm requires it.

        Called when _ensure_initialized() detects embedding is already loaded
        but the user may have changed the approximation algorithm setting.
        """
        settings = self._get_settings()
        approx_algo = settings.get("approximation_algorithm", "exhaustive_cosine")
        if approx_algo not in ("itq_lsh_2stage", "itq_lsh_3stage"):
            return
        if getattr(self, "_v_itq_boundary", None) is not None:
            return
        model_provider = getattr(self, "_v_model_provider", None)
        if model_provider is None:
            return
        try:
            self._v_itq_boundary = model_provider.get_itq_boundary()
            self._v_pivot_data = model_provider.get_pivot_data()
            logger.info(
                "Loaded ITQ/pivot data for %s (itq=%s, pivot=%s)",
                approx_algo,
                self._v_itq_boundary is not None,
                self._v_pivot_data is not None,
            )
        except Exception as e:
            logger.warning("Failed to load ITQ/pivot data: %s", e)

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
                "itq_candidates": registry(
                    "collective.vectorsearch.itq_candidates", default=100
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
        implemented_algorithms = [
            "exhaustive_cosine",
            "itq_lsh_2stage",
            "itq_lsh_3stage",
        ]
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
            "itq_candidates": 100,
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
        """Dispatch vector search to the configured approximation algorithm.

        Supports:
        - exhaustive_cosine: Brute-force cosine similarity on all documents
        - itq_lsh_2stage: Hamming distance filtering → cosine similarity
        - itq_lsh_3stage: Pivot range filtering → Hamming → cosine similarity

        Falls back to exhaustive_cosine if required data (ITQ/pivot) is unavailable.
        """
        query_str = " ".join(record.keys)
        if not query_str:
            return None

        query_vectors = self.embedding.embed(query_str, query=True)
        settings = self._get_settings()
        algo = settings.get("approximation_algorithm", "exhaustive_cosine")

        # For ITQ/pivot algorithms, compute passage-space embedding for hash
        # computation. ITQ boundaries and pivot data were trained on passage
        # embeddings, so the query must be in the same vector space for
        # meaningful Hamming distance / pivot distance comparisons.
        # The query-prefixed embedding is still used for final cosine scoring.
        passage_vectors = None
        if algo in ("itq_lsh_2stage", "itq_lsh_3stage"):
            passage_vectors = self.embedding.embed(query_str, query=False)

        # Layered fallback: 3-stage → 2-stage → exhaustive
        if algo == "itq_lsh_3stage":
            itq = getattr(self, "_v_itq_boundary", None)
            pivot = getattr(self, "_v_pivot_data", None)
            if itq and pivot:
                result = self._query_itq_lsh_3stage(
                    query_vectors, passage_vectors, settings
                )
                if result is not None:
                    return result
            if itq:
                logger.warning("Pivot data unavailable, downgrading 3-stage to 2-stage")
                result = self._query_itq_lsh_2stage(
                    query_vectors, passage_vectors, settings
                )
                if result is not None:
                    return result
            logger.warning("Falling back to exhaustive cosine search")

        elif algo == "itq_lsh_2stage":
            itq = getattr(self, "_v_itq_boundary", None)
            if itq:
                result = self._query_itq_lsh_2stage(
                    query_vectors, passage_vectors, settings
                )
                if result is not None:
                    return result
            logger.warning(
                "ITQ data unavailable, falling back to exhaustive cosine search"
            )

        return self._query_exhaustive_cosine(query_vectors)

    def _query_exhaustive_cosine(self, query_vectors):
        """Exhaustive cosine similarity search on all documents.

        Reads all vectors from catalog metadata (llm_vector column).
        """
        docids, vectors = self._get_all_doc_vectors()
        if vectors.size == 0:
            return IIBucket()
        indices, scores = self.similarity_algorithm(vectors, query_vectors)
        bucket = IIBucket()
        for docid, score in zip(docids[indices], scores):
            int_docid = int(docid)
            if int_docid not in bucket:
                bucket[int_docid] = int(score * 100_000_000)
        return bucket

    def _query_itq_lsh_2stage(self, query_vectors, passage_vectors, settings):
        """2-stage search: Hamming distance ranking → cosine similarity.

        Stage 2: Scans itq_hashes from catalog METADATA for all documents,
                 ranks by Hamming distance and keeps top N candidates.
        Stage 3: Loads llm_vector from catalog METADATA for candidates only,
                 computes precise cosine similarity.

        Args:
            query_vectors: Query-prefixed embeddings (for cosine similarity)
            passage_vectors: Passage-prefixed embeddings (for ITQ hash computation)
            settings: Configuration dict
        """
        t0 = time.perf_counter()
        itq_boundary = self._v_itq_boundary
        itq_candidates = settings.get("itq_candidates", 100)

        # Compute query ITQ hash using passage-space embedding
        # (ITQ boundary was trained on passage embeddings)
        pv = passage_vectors[0] if passage_vectors.ndim == 2 else passage_vectors
        query_hash = binary_hash_to_integers(itq_boundary.compute_hash(pv))
        if query_hash[0] is None:
            logger.warning("Failed to compute query ITQ hash")
            return None

        # Stage 2: Scan itq_hashes from catalog METADATA, rank by Hamming distance
        try:
            catalog = api.portal.get_tool("portal_catalog")
        except Exception:
            logger.warning("Cannot access catalog for 2-stage search")
            return None

        scored_docs = []  # [(hamming_distance, docid), ...]
        total_docs = 0
        docs_with_hashes = 0
        docs_without_hashes = 0

        for docid in self._docid_to_path:
            total_docs += 1
            try:
                metadata = catalog.getMetadataForRID(docid)
                if not metadata:
                    continue
                doc_hashes = metadata.get("itq_hashes")
                if not doc_hashes:
                    docs_without_hashes += 1
                    continue
                docs_with_hashes += 1
                min_dist = batch_min_hamming_distance(query_hash, doc_hashes)
                scored_docs.append((min_dist, docid))
            except Exception:
                continue

        # Sort by Hamming distance (ascending) and take top N
        scored_docs.sort(key=lambda x: x[0])
        candidate_docids = [docid for _, docid in scored_docs[:itq_candidates]]

        t1 = time.perf_counter()
        top_distances = [d for d, _ in scored_docs[:itq_candidates]]
        logger.info(
            "2-stage Stage 2 (hamming ranking): %d -> %d candidates in %.4fs"
            " (with_hashes=%d, without=%d, top_distances=%s)",
            total_docs,
            len(candidate_docids),
            t1 - t0,
            docs_with_hashes,
            docs_without_hashes,
            top_distances[:20] if top_distances else "none",
        )

        if not candidate_docids:
            return IIBucket()

        # Stage 3: Cosine on candidates (reads llm_vector from METADATA)
        result = self._cosine_on_candidates(query_vectors, candidate_docids, catalog)
        t2 = time.perf_counter()
        logger.info(
            "2-stage Stage 3 (cosine METADATA): %d candidates -> %d results in %.4fs",
            len(candidate_docids),
            len(result),
            t2 - t1,
        )
        logger.info("Total 2-stage search: %.4fs", t2 - t0)
        return result

    def _query_itq_lsh_3stage(self, query_vectors, passage_vectors, settings):
        """3-stage search: Pivot INDEX → Hamming ranking → Cosine METADATA.

        Stage 1: Uses pivot1-8 KeywordIndex range queries to filter candidates.
        Stage 2: Reads itq_hashes from catalog METADATA for Stage 1 candidates,
                 ranks by Hamming distance and keeps top N.
        Stage 3: Loads llm_vector from catalog METADATA for Stage 2 candidates,
                 computes precise cosine similarity.

        Args:
            query_vectors: Query-prefixed embeddings (for cosine similarity)
            passage_vectors: Passage-prefixed embeddings (for ITQ hash and pivot computation)
            settings: Configuration dict
        """
        t0 = time.perf_counter()
        itq_boundary = self._v_itq_boundary
        pivot_data = self._v_pivot_data
        pivot_threshold = settings.get("pivot_threshold", 200)
        itq_candidates = settings.get("itq_candidates", 100)

        # Use passage-space embedding for pivot/ITQ (trained on passage embeddings)
        pv = passage_vectors[0] if passage_vectors.ndim == 2 else passage_vectors

        # Stage 1: Pivot range filter (uses catalog INDEXES: pivot1-8)
        query_pivot_distances = pivot_data.compute_distances(pv)
        query_pivot_distances_int = [
            int(round(d * 1000)) for d in query_pivot_distances
        ]

        stage1_candidates = self._pivot_filter(
            query_pivot_distances_int, pivot_threshold
        )
        # Only keep candidates in our VectorIndex
        our_docids = IISet(self._docid_to_path.keys())
        stage1_candidates = intersection(stage1_candidates, our_docids)

        t1 = time.perf_counter()
        logger.info(
            "3-stage Stage 1 (pivot INDEX): %d -> %d candidates in %.4fs",
            len(our_docids),
            len(stage1_candidates) if stage1_candidates else 0,
            t1 - t0,
        )

        if not stage1_candidates:
            return IIBucket()

        # Stage 2: Hamming ranking (uses catalog METADATA: itq_hashes)
        query_hash = binary_hash_to_integers(itq_boundary.compute_hash(pv))
        if query_hash[0] is None:
            logger.warning("Failed to compute query ITQ hash for 3-stage")
            return None

        try:
            catalog = api.portal.get_tool("portal_catalog")
        except Exception:
            logger.warning("Cannot access catalog for 3-stage search")
            return None

        scored_docs = []  # [(hamming_distance, docid), ...]
        for docid in stage1_candidates:
            try:
                metadata = catalog.getMetadataForRID(docid)
                if not metadata:
                    continue
                doc_hashes = metadata.get("itq_hashes")
                if not doc_hashes:
                    continue
                min_dist = batch_min_hamming_distance(query_hash, doc_hashes)
                scored_docs.append((min_dist, docid))
            except Exception:
                continue

        # Sort by Hamming distance (ascending) and take top N
        scored_docs.sort(key=lambda x: x[0])
        stage2_candidates = [docid for _, docid in scored_docs[:itq_candidates]]

        t2 = time.perf_counter()
        top_distances = [d for d, _ in scored_docs[:itq_candidates]]
        logger.info(
            "3-stage Stage 2 (hamming ranking): %d -> %d candidates in %.4fs"
            " (top_distances=%s)",
            len(stage1_candidates) if stage1_candidates else 0,
            len(stage2_candidates),
            t2 - t1,
            top_distances[:20] if top_distances else "none",
        )

        if not stage2_candidates:
            return IIBucket()

        # Stage 3: Cosine similarity (uses catalog METADATA: llm_vector)
        result = self._cosine_on_candidates(query_vectors, stage2_candidates, catalog)
        t3 = time.perf_counter()
        logger.info(
            "3-stage Stage 3 (cosine METADATA): %d candidates -> %d results in %.4fs",
            len(stage2_candidates),
            len(result),
            t3 - t2,
        )
        logger.info("Total 3-stage search: %.4fs", t3 - t0)
        return result

    def _pivot_filter(self, query_pivot_distances_int, pivot_threshold):
        """Stage 1: Pivot range filtering using KeywordIndex range queries.

        Uses catalog INDEX (pivot1-8 KeywordIndex) with range='min:max'.
        Each pivot's result is intersected to find documents matching ALL pivots.

        Args:
            query_pivot_distances_int: List of 8 integer pivot distances for query
            pivot_threshold: Integer threshold for range (e.g. 200 = ±0.200)

        Returns:
            IISet of candidate document RIDs
        """
        try:
            catalog = api.portal.get_tool("portal_catalog")
        except Exception:
            return IISet()

        candidate_set = None

        for i in range(8):
            pivot_name = f"pivot{i + 1}"
            if pivot_name not in catalog.Indexes:
                continue

            index = catalog.Indexes[pivot_name]
            q_dist = query_pivot_distances_int[i]
            min_val = max(0, q_dist - pivot_threshold)
            max_val = q_dist + pivot_threshold

            # Use IndexQuery for range query on KeywordIndex
            query_dict = {pivot_name: {"query": [min_val, max_val], "range": "min:max"}}
            index_query = IndexQuery(
                query_dict,
                pivot_name,
                index.query_options,
                index.operators,
                index.useOperator,
            )
            result = index.query_index(index_query)

            if result is None or len(result) == 0:
                return IISet()  # No matches for this pivot, short circuit

            if candidate_set is None:
                candidate_set = IISet(result)
            else:
                candidate_set = intersection(candidate_set, IISet(result))

            if not candidate_set:
                return IISet()

        return candidate_set or IISet()

    def _cosine_on_candidates(self, query_vectors, candidate_docids, catalog):
        """Stage 3: Cosine similarity on candidate documents only.

        Reads llm_vector from catalog METADATA for candidates, computes cosine
        similarity, and returns scored results as IIBucket.
        """
        all_docids = []
        all_vectors = []

        for docid in candidate_docids:
            try:
                metadata = catalog.getMetadataForRID(docid)
                if not metadata:
                    continue
                vectors_list = metadata.get("llm_vector")
                if not vectors_list:
                    continue
                vectors = np.array(vectors_list)
                all_docids.extend([docid] * len(vectors))
                all_vectors.append(vectors)
            except Exception:
                continue

        if not all_vectors:
            return IIBucket()

        vectors = np.concatenate(all_vectors)
        docids = np.array(all_docids, dtype=int)
        indices, scores = self.similarity_algorithm(vectors, query_vectors)

        bucket = IIBucket()
        for docid, score in zip(docids[indices], scores):
            int_docid = int(docid)
            if int_docid not in bucket:
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
