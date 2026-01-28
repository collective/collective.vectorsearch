# -*- coding: utf-8 -*-
"""Module where all interfaces, events and exceptions live."""

from zope.publisher.interfaces.browser import IDefaultBrowserLayer
from zope.interface import Interface
from zope import schema
from zope.schema.vocabulary import SimpleVocabulary, SimpleTerm
from collective.vectorsearch import _


class ICollectiveVectorsearchLayer(IDefaultBrowserLayer):
    """Marker interface that defines a browser layer."""


class IVectorIndex(Interface):
    """Marker interface for IVectorIndex"""


class IVectorSearchSettings(Interface):
    """Vector Search configuration settings."""

    embedding_model_name = schema.TextLine(
        title=_(u"Embedding Model Name"),
        description=_(
            u"SentenceTransformer model name to use for embeddings "
            u"(e.g., thenlper/gte-small, all-MiniLM-L6-v2)"
        ),
        default=u"thenlper/gte-small",
        required=True,
    )

    embedding_prefix_query = schema.TextLine(
        title=_(u"Query Prefix"),
        description=_(
            u"Prefix to add to query text before embedding. "
            u"Some models benefit from query-specific prefixes."
        ),
        default=u"query: ",
        required=False,
    )

    embedding_chunk_size = schema.Int(
        title=_(u"Chunk Size"),
        description=_(u"Maximum character length for text chunks during embedding."),
        default=500,
        required=True,
        min=100,
        max=10000,
    )

    similarity_algorithm = schema.Choice(
        title=_(u"Similarity Algorithm"),
        description=_(u"Algorithm to use for similarity calculation."),
        default=u"cosine",
        required=True,
        vocabulary=SimpleVocabulary([
            SimpleTerm(value=u"cosine", title=_(u"Cosine Similarity")),
            # Future: Add more algorithms as they are implemented
        ]),
    )
