from logging import getLogger
import time
from App.special_dtml import DTMLFile
from BTrees.IOBTree import IOBTree
from BTrees.IIBTree import IIBucket
from BTrees.Length import Length
from OFS.SimpleItem import SimpleItem
from Acquisition import Implicit
from Persistence import Persistent
from zope.interface import implementer
from AccessControl.class_init import InitializeClass
from AccessControl.SecurityInfo import ClassSecurityInfo
from AccessControl.Permissions import search_zcatalog
from Products.ZCatalog.ZCatalog import manage_zcatalog_indexes
from Products.PluginIndexes.interfaces import IQueryIndex

try:
    from plone.app.contenttypes.indexers import SearchableText
except ImportError:
    SearchableText = None

import numpy as np
from sentence_transformers import SentenceTransformer
from plone import api

from collective.vectorsearch.interfaces import IVectorIndex, IEmbeddingModelProvider
from collective.vectorsearch.embedding import SentenceTransformerEmbedding
from collective.vectorsearch.similarity_algorithm import CosineSimilarityAlgorithm
from zope.component import queryUtility

logger = getLogger("collective.vectorsearch")


class ModelCache:
    """Singleton cache for SentenceTransformer models.

    This prevents loading the same model multiple times into memory,
    which can be very expensive (100MB+ per model).
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
        """
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
            'cached_models': list(self._models.keys()),
            'model_count': len(self._models)
        }


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

        # Handle indexed_attrs from extra parameter
        if extra is not None and isinstance(extra, dict):
            indexed_attrs = extra.get('indexed_attrs', '')
            if indexed_attrs:
                if isinstance(indexed_attrs, str):
                    self.indexed_attrs = [
                        attr.strip() for attr in indexed_attrs.split(',')
                    ]
                else:
                    self.indexed_attrs = indexed_attrs
            else:
                self.indexed_attrs = [self.id]
        else:
            self.indexed_attrs = [self.id]

        # Lazy initialization flags - model/embedding loaded on first use
        self._embedding = None
        self._model_provider = None
        self._similarity_algorithm = None
        self.itq_boundary = None
        self.pivot_data = None

    def _ensure_initialized(self):
        """Lazy initialization of embedding model and similarity algorithm.

        This is called on first use (index_doc or query_index) to avoid
        loading heavy ML models during Quickinstall.
        """
        if self._embedding is not None:
            return

        settings = self._get_settings()
        model_id = settings.get('embedding_model', 'gte-small')
        chunk_size = settings.get('embedding_chunk_size', 500)
        approx_algo = settings.get('approximation_algorithm', 'exhaustive_cosine')

        # Get model provider utility
        model_provider = queryUtility(IEmbeddingModelProvider, name=model_id)

        if model_provider is None:
            logger.warning(f"Model provider '{model_id}' not found, using gte-small")
            model_provider = queryUtility(IEmbeddingModelProvider, name='gte-small')

        self._model_provider = model_provider

        # Get prefixes from model provider
        prefix_query = getattr(model_provider, 'query_prefix', None)
        prefix_passage = getattr(model_provider, 'passage_prefix', None)

        # Get embedding instance from provider
        self._embedding = model_provider.get_embedding_instance(
            chunk_size=chunk_size,
            prefix_query=prefix_query,
            prefix_passage=prefix_passage
        )

        # Load ITQ data if needed (Phase 2: not implemented yet)
        if approx_algo in ('itq_lsh_2stage', 'itq_lsh_3stage'):
            self.itq_boundary = model_provider.get_itq_boundary()
            self.pivot_data = model_provider.get_pivot_data()

        # Initialize similarity algorithm
        self._similarity_algorithm = CosineSimilarityAlgorithm()

    @property
    def embedding(self):
        """Lazy-loaded embedding instance."""
        self._ensure_initialized()
        return self._embedding

    @property
    def model_provider(self):
        """Lazy-loaded model provider."""
        self._ensure_initialized()
        return self._model_provider

    @property
    def similarity_algorithm(self):
        """Lazy-loaded similarity algorithm."""
        self._ensure_initialized()
        return self._similarity_algorithm

    def _get_settings(self):
        """Retrieve all configuration from registry."""
        try:
            registry = api.portal.get_registry_record

            settings = {
                'embedding_model': registry(
                    'collective.vectorsearch.embedding_model',
                    default='gte-small'
                ),
                'embedding_chunk_size': registry(
                    'collective.vectorsearch.embedding_chunk_size',
                    default=500
                ),
                'storage_backend': registry(
                    'collective.vectorsearch.storage_backend',
                    default='btrees'
                ),
                'external_db_uri': registry(
                    'collective.vectorsearch.external_db_uri',
                    default=''
                ),
                'approximation_algorithm': registry(
                    'collective.vectorsearch.approximation_algorithm',
                    default='exhaustive_cosine'
                ),
                'pivot_threshold': registry(
                    'collective.vectorsearch.pivot_threshold',
                    default=20
                ),
                'hamming_distance_threshold': registry(
                    'collective.vectorsearch.hamming_distance_threshold',
                    default=3
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
        implemented_backends = ['btrees']
        if settings['storage_backend'] not in implemented_backends:
            logger.warning(
                f"Storage backend '{settings['storage_backend']}' is not yet implemented. "
                f"Current implementation: {', '.join(implemented_backends)}"
            )

        # Approximation algorithm implementation status
        implemented_algorithms = ['exhaustive_cosine']
        if settings['approximation_algorithm'] not in implemented_algorithms:
            logger.warning(
                f"Approximation algorithm '{settings['approximation_algorithm']}' is not yet implemented. "
                f"Current implementation: {', '.join(implemented_algorithms)}"
            )

    def _get_default_settings(self):
        """Fallback default settings."""
        return {
            'embedding_model': 'gte-small',
            'embedding_chunk_size': 500,
            'storage_backend': 'btrees',
            'external_db_uri': '',
            'approximation_algorithm': 'exhaustive_cosine',
            'pivot_threshold': 20,
            'hamming_distance_threshold': 3,
        }

    def _change_length(self, name, value):
        length_obj = getattr(self, name, None)
        if length_obj is not None:
            length_obj.change(value)
        else:
            setattr(self, name, Length(value))

    security.declareProtected(manage_zcatalog_indexes, 'index_object')
    def index_object(self, documentId, obj, threshold=None):
        count = 0
        if SearchableText is not None:
            try:
                text = SearchableText(obj)
                row = self.index_doc(documentId, text)
                count += row
            except Exception as e:
                logger.warning(
                    "Failed to index SearchableText for document %s: %s",
                    documentId, e
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
                    field, documentId, e
                )
        return count  # Number of vector rows

    def index_doc(self, docid, text):
        # Skip empty or invalid text
        if not text or not isinstance(text, str):
            logger.debug("Skipping empty or invalid text for document %s", docid)
            return 0

        old_vectors = self._docvectors.get(docid, None)
        if old_vectors is not None:
            self._change_length("document_count", -1)
            old_row, old_col = old_vectors.shape
            self._change_length("length", -old_row)

        try:
            vectors = self.embedding.embed(text)
        except Exception as e:
            logger.error("Failed to embed text for document %s: %s", docid, e)
            return 0

        row, col = vectors.shape
        self._change_length("document_count", 1)
        self._change_length("length", row)
        self._docvectors[docid] = vectors

        # Track which model was used (set on first index, or update if changed)
        settings = self._get_settings()
        current_model = settings.get('embedding_model', 'gte-small')
        if getattr(self, 'indexed_with_model', None) is None:
            self.indexed_with_model = current_model

        return row

    security.declareProtected(manage_zcatalog_indexes, 'unindex_object')
    def unindex_object(self, docid):
        old_vectors = self._docvectors.get(docid, None)
        if old_vectors is not None:
            self._change_length("document_count", -1)
            old_row, old_col = old_vectors.shape
            self._change_length("length", -old_row)
            try:
                del self._docvectors[docid]
            except KeyError:
                logger.warning("Document %s not found in index during unindexing", docid)

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

    security.declareProtected(search_zcatalog, 'query_index')
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

    security.declareProtected(search_zcatalog, 'getEntryForObject')
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
            elapsed
        )
        return result

    security.declareProtected(search_zcatalog, 'uniqueValues')
    def uniqueValues(self, name=None, withLengths=0):
        """Return unique values for the index.

        Vector indexes don't have traditional unique values like keyword indexes.
        Returns an empty tuple for compatibility with the catalog interface.
        """
        logger.debug("uniqueValues called: name=%s, withLengths=%s", name, withLengths)
        # Vector embeddings don't have discrete unique values
        # Return empty tuple for catalog compatibility
        return ()

    security.declareProtected(search_zcatalog, 'numObjects')
    def numObjects(self):
        return self.document_count()

    security.declareProtected(search_zcatalog, 'indexSize')
    def indexSize(self):
        return self.length()

    security.declareProtected(manage_zcatalog_indexes, 'clear')
    def clear(self):
        self._docvectors = IOBTree()
        self.length = Length()
        self.document_count = Length()
        self.indexed_with_model = None

    security.declareProtected(search_zcatalog, 'getIndexedModel')
    def getIndexedModel(self):
        """Return the model ID used to create the indexed vectors.

        Returns:
            str or None: Model ID if vectors exist, None if index is empty
        """
        return getattr(self, 'indexed_with_model', None)

    security.declareProtected(search_zcatalog, 'getIndexSourceNames')
    def getIndexSourceNames(self):
        """Return the list of indexed attribute names."""
        return getattr(self, "indexed_attrs", [self.id])

    security.declareProtected(search_zcatalog, 'getIndexQueryNames')
    def getIndexQueryNames(self):
        return (self.id,)

    security.declareProtected(search_zcatalog, 'getIndexType')
    def getIndexType(self):
        """Return the type of this index."""
        start_time = time.perf_counter()
        result = "VectorIndex"
        elapsed = time.perf_counter() - start_time
        logger.debug("getIndexType called, time=%.4f seconds", elapsed)
        return result


InitializeClass(VectorIndex)


# Note: manage_addVectorIndex is now handled by AddVectorIndexView in browser/add_vector_index.py
