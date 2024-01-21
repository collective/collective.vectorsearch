from logging import getLogger
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

        model_name = "thenlper/gte-small"  # TODO: Make it configurable
        prefix_query = "query: "  # TODO: Make it configurable
        model = SentenceTransformer(model_name)
        self.embedding = SentenceTransformerEmbedding(
            model, chank_size=500, prefix_query=prefix_query
        )
        self.similarity_algorithm = (
            CosineSimilarityAlgorithm()
        )  # TODO: Make it configurable

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
        logger.debug(
            "TODO: timing check ", "_apply_index:: ", request
        )  # TODO: timing check

    @security.protected(search_zcatalog)
    def query(self, query, nbest=10):
        logger.debug(
            "TODO: timing check ", "query:: ", query, nbest
        )  # TODO: timing check
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
                bucket[int_docid] = int(
                    score * 100_000_000
                )  # TODO Is it okay? Zope needs int
        return bucket

    def _get_all_doc_vectors(self):
        items = self._docvectors.items()
        vectors = np.concatenate([v for k, v in items])
        docids = np.concatenate([[k] * v.shape[0] for k, v in items])
        return docids, vectors

    def getEntryForObject(self, documentId, default=None):
        logger.debug(
            "TODO: timing check: ",
            "getEntryForObject:: ",
            documentId,
            default,
        )  # TODO: timing check

    def uniqueValues(self, name=None, withLengths=0):
        logger.debug(
            "TODO: timing check: ", "uniqueValues:: ", name, withLengths
        )  # TODO: timing check
        raise NotImplementedError

    def numObjects(self):
        return self.document_count()

    def indexSize(self):
        return self.length()

    def clear(self):
        self._docvectors = IOBTree()
        self.length = Length()
        self.document_count = Length()

    def getIndexSourceNames(self):
        return getattr(self, "indexed_attrs", [self.id])  # TODO: Not using it now?

    def getIndexQueryNames(self):
        return (self.id,)

    def getIndexType(self):
        logger.debug("TODO: timing check: getIndexType:: ")  # TODO: timing check
        return "VectorIndex"


InitializeClass(VectorIndex)
manage_addVectorIndexForm = DTMLFile("dtml/addVectorIndex", globals())


def manage_addVectorIndex(self, id, extra=None, REQUEST=None, RESPONSE=None, URL3=None):
    """Add a vector index"""
    return self.manage_addIndex(
        id, "VectorIndex", extra=extra, REQUEST=REQUEST, RESPONSE=RESPONSE, URL1=URL3
    )
