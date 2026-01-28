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
from Products.PluginIndexes.interfaces import IQueryIndex

try:
    from plone.app.contenttypes.indexers import SearchableText
except ImportError:
    SearchableText = None

import numpy as np
from sentence_transformers import SentenceTransformer
from plone import api

from collective.vectorsearch.interfaces import IVectorIndex
from collective.vectorsearch.embedding import SentenceTransformerEmbedding
from collective.vectorsearch.similarity_algorithm import CosineSimilarityAlgorithm

logger = getLogger("collective.vectorsearch")


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

        # Get settings from registry with fallback defaults
        settings = self._get_settings()

        model_name = settings.get('embedding_model_name', 'thenlper/gte-small')
        prefix_query = settings.get('embedding_prefix_query', 'query: ')
        chunk_size = settings.get('embedding_chunk_size', 500)
        similarity_algo = settings.get('similarity_algorithm', 'cosine')

        # Initialize embedding
        model = SentenceTransformer(model_name)
        self.embedding = SentenceTransformerEmbedding(
            model, chank_size=chunk_size, prefix_query=prefix_query
        )

        # Initialize similarity algorithm
        if similarity_algo == 'cosine':
            self.similarity_algorithm = CosineSimilarityAlgorithm()
        else:
            # Default to cosine for now
            self.similarity_algorithm = CosineSimilarityAlgorithm()

    def _get_settings(self):
        """Get settings from registry with error handling and fallback defaults."""
        try:
            registry = api.portal.get_registry_record
            return {
                'embedding_model_name': registry(
                    'collective.vectorsearch.embedding_model_name',
                    default='thenlper/gte-small'
                ),
                'embedding_prefix_query': registry(
                    'collective.vectorsearch.embedding_prefix_query',
                    default='query: '
                ),
                'embedding_chunk_size': registry(
                    'collective.vectorsearch.embedding_chunk_size',
                    default=500
                ),
                'similarity_algorithm': registry(
                    'collective.vectorsearch.similarity_algorithm',
                    default='cosine'
                ),
            }
        except Exception as e:
            logger.warning(
                f"Could not read registry settings: {e}. Using defaults."
            )
            return {
                'embedding_model_name': 'thenlper/gte-small',
                'embedding_prefix_query': 'query: ',
                'embedding_chunk_size': 500,
                'similarity_algorithm': 'cosine',
            }

    def _change_length(self, name, value):
        length_obj = getattr(self, name, None)
        if length_obj is not None:
            length_obj.change(value)
        else:
            setattr(self, name, Length(value))

    def index_object(self, documentId, obj, threshold=None):
        count = 0
        if SearchableText is not None:
            text = SearchableText(obj)
            row = self.index_doc(documentId, text)
            count += row
        fields = self.getIndexSourceNames()
        for field in fields:
            value = getattr(obj, field, None)
            if value is not None:
                row = self.index_doc(documentId, value)
                count += row
        return count  # Number of vector rows

    def index_doc(self, docid, text):
        old_vectors = self._docvectors.get(docid, None)
        if old_vectors is not None:
            self._change_length("document_count", -1)
            old_row, old_col = old_vectors.shape
            self._change_length("length", -old_row)
        vectors = self.embedding.embed(text)
        row, col = vectors.shape
        self._change_length("document_count", 1)
        self._change_length("length", row)
        self._docvectors[docid] = vectors
        return row

    def unindex_object(self, docid):
        old_vectors = self._docvectors.get(docid, None)
        if old_vectors is not None:
            self._change_length("document_count", -1)
            old_row, old_col = old_vectors.shape
            self._change_length("length", -old_row)
        del self._docvectors[docid]

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
        items = self._docvectors.items()
        vectors = np.concatenate([v for k, v in items])
        docids = np.concatenate([[k] * v.shape[0] for k, v in items])
        return docids, vectors

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

    def uniqueValues(self, name=None, withLengths=0):
        """Return unique values for the index.

        Vector indexes don't have traditional unique values like keyword indexes.
        Returns an empty tuple for compatibility with the catalog interface.
        """
        logger.debug("uniqueValues called: name=%s, withLengths=%s", name, withLengths)
        # Vector embeddings don't have discrete unique values
        # Return empty tuple for catalog compatibility
        return ()

    def numObjects(self):
        return self.document_count()

    def indexSize(self):
        return self.length()

    def clear(self):
        self._docvectors = IOBTree()
        self.length = Length()
        self.document_count = Length()

    def getIndexSourceNames(self):
        """Return the list of indexed attribute names."""
        return getattr(self, "indexed_attrs", [self.id])

    def getIndexQueryNames(self):
        return (self.id,)

    def getIndexType(self):
        """Return the type of this index."""
        start_time = time.perf_counter()
        result = "VectorIndex"
        elapsed = time.perf_counter() - start_time
        logger.debug("getIndexType called, time=%.4f seconds", elapsed)
        return result


InitializeClass(VectorIndex)
manage_addVectorIndexForm = DTMLFile("dtml/addVectorIndex", globals())


def manage_addVectorIndex(self, id, extra=None, REQUEST=None, RESPONSE=None, URL3=None):
    """Add a vector index"""
    return self.manage_addIndex(
        id, "VectorIndex", extra=extra, REQUEST=REQUEST, RESPONSE=RESPONSE, URL1=URL3
    )
