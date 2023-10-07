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
import torch
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer

from collective.vectorsearch.interfaces import IVectorIndex

logger = getLogger("collective.vectorsearch")


# model_name = "intfloat/multilingual-e5-large"
model_name = "thenlper/gte-small"
MODEL = SentenceTransformer(model_name)
PREFIX_QUERY = "query: "


CHUNK_SIZE = 500


def get_embeddings(text: str, query=False) -> np.ndarray:
    if query:
        text = PREFIX_QUERY + text
    texts = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    embeddings = MODEL.encode(texts)
    return embeddings


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
        vectors = get_embeddings(text)
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
        query = get_embeddings(query_str, query=True)
        docids, vectors = self._get_all_doc_vectors()
        indices, scores = self._get_similarities(vectors, query)
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

    def _get_similarities(self, vectors, query, k=10):
        if vectors.shape[0] < k:
            k = vectors.shape[0]
        t_vectors = torch.tensor(vectors, dtype=torch.float32)
        t_query = torch.tensor(query, dtype=torch.float32)
        similarities = cosine_similarity(t_vectors, t_query)
        top10_values, top10_indices = torch.topk(similarities, k)
        return top10_indices.numpy(), top10_values.numpy()

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
