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

    embedding_model = schema.Choice(
        title=_(u"Embedding Model"),
        description=_(u"Select the embedding model to use for vector search"),
        vocabulary="collective.vectorsearch.embedding_models",
        default=u"gte-small",
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


class IEmbeddingModelProvider(Interface):
    """Marker interface for embedding model providers.

    Named utilities implementing this interface will be available
    for selection in the control panel.
    """

    id = schema.ASCIILine(
        title=_(u"Model ID"),
        description=_(u"Unique identifier for this embedding model"),
        required=True,
    )

    title = schema.TextLine(
        title=_(u"Display Title"),
        description=_(u"Human-readable name shown in UI"),
        required=True,
    )

    description = schema.Text(
        title=_(u"Description"),
        description=_(u"Description of this model and its use cases"),
        required=False,
    )

    model_name = schema.TextLine(
        title=_(u"Model Name"),
        description=_(u"Internal model name (e.g., HuggingFace model ID)"),
        required=True,
    )

    use_cache_dir = schema.Bool(
        title=_(u"Use Cache Directory"),
        description=_(u"Whether to cache model files for faster loading"),
        default=False,
        required=True,
    )

    vector_dimensions = schema.Int(
        title=_(u"Vector Dimensions"),
        description=_(u"Dimensionality of embedding vectors produced by this model"),
        required=True,
    )

    hash_length = schema.Int(
        title=_(u"Hash Length"),
        description=_(u"Length of binary hash for ITQ quantization"),
        default=128,
        required=True,
    )

    itq_boundary_name = schema.TextLine(
        title=_(u"ITQ Boundary Name"),
        description=_(u"Name of pre-trained ITQ boundary vector (if available)"),
        required=False,
    )

    available_similarity_methods = schema.List(
        title=_(u"Available Similarity Methods"),
        description=_(u"List of similarity calculation methods supported"),
        value_type=schema.Choice(values=[u'cosine', u'itq_hamming', u'euclidean']),
        required=True,
    )

    embedding_class = schema.ASCIILine(
        title=_(u"Embedding Class"),
        description=_(u"Name of embedding class to use (e.g., SentenceTransformerEmbedding, FastEmbedEmbedding)"),
        default='SentenceTransformerEmbedding',
        required=True,
    )

    def get_embedding_instance(chunk_size=500, prefix_query=None):
        """Factory method to create an embedding instance.

        Returns:
            EmbeddingBase instance configured for this model
        """

    def get_itq_boundary():
        """Load and return ITQ pre-trained boundary vector.

        Returns:
            numpy.ndarray or None if not available
        """

    def get_pivot_data():
        """Load and return pre-prepared pivot calculation data.

        Returns:
            numpy.ndarray or None if not available
        """
