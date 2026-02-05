# -*- coding: utf-8 -*-
"""Module where all interfaces, events and exceptions live."""

from zope.publisher.interfaces.browser import IDefaultBrowserLayer
from zope.interface import Interface, invariant, Invalid
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
        description=_(
            u"Select the embedding model for vector search. "
            u"WARNING: Changing this model requires clearing all existing vectors and reindexing all content. "
            u"Different models produce incompatible vector dimensions. "
            u"This operation cannot be undone."
        ),
        vocabulary="collective.vectorsearch.embedding_models",
        default=u"all-minilm-l6",
        required=True,
    )

    embedding_chunk_size = schema.Int(
        title=_(u"Text Chunk Size"),
        description=_(
            u"Maximum character length for splitting long text into chunks. "
            u"Long documents are divided into multiple chunks of this size for vectorization. "
            u"Each chunk is embedded separately."
        ),
        default=500,
        required=True,
        min=100,
        max=10000,
    )

    storage_backend = schema.Choice(
        title=_(u"Storage Backend"),
        description=_(
            u"Storage system for vector data. "
            u"BTrees (internal), or external vector databases (FAISS, DuckDB, etc.)"
        ),
        vocabulary="collective.vectorsearch.storage_backends",
        default=u"btrees",
        required=True,
    )

    external_db_uri = schema.URI(
        title=_(u"External Database URI"),
        description=_(
            u"URI for external vector database connection. "
            u"Required when storage backend is not 'btrees'. "
            u"Examples: 'http://localhost:8080/faiss', 'duckdb:///path/to/db.duckdb'"
        ),
        required=False,
    )

    approximation_algorithm = schema.Choice(
        title=_(u"Approximation Algorithm"),
        description=_(
            u"Search strategy for similarity calculation. "
            u"Based on LSH cascade research (lsh-cascade-poc). "
            u"Currently only 'Exhaustive Cosine' is implemented."
        ),
        vocabulary="collective.vectorsearch.approximation_algorithms",
        default=u"exhaustive_cosine",
        required=True,
    )

    pivot_threshold = schema.Int(
        title=_(u"Pivot Threshold (Stage 1)"),
        description=_(
            u"Threshold for pivot-based filtering in Stage 1. "
            u"Higher values = more candidates retained. "
            u"Recommended: 20 (89.8% recall) or 15 (85.9% recall)."
        ),
        default=20,
        required=False,
        min=1,
        max=100,
    )

    hamming_distance_threshold = schema.Int(
        title=_(u"Hamming Distance Threshold (Stage 2)"),
        description=_(
            u"Maximum Hamming distance for candidate filtering in Stage 2. "
            u"Lower values = stricter filtering, fewer candidates. "
            u"Recommended: 2-10. Default: 3."
        ),
        default=3,
        required=False,
        min=0,
        max=128,
    )

    @invariant
    def validate_external_db_uri(obj):
        """Validate that external_db_uri is provided when storage_backend is not btrees."""
        if obj.storage_backend != u'btrees' and not obj.external_db_uri:
            raise Invalid(
                _(u"External Database URI is required when storage backend is not BTrees.")
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

    query_prefix = schema.TextLine(
        title=_(u"Query Prefix"),
        description=_(u"Prefix to add to query text (e.g., 'query: ' for E5 models). None if not needed."),
        required=False,
    )

    passage_prefix = schema.TextLine(
        title=_(u"Passage Prefix"),
        description=_(u"Prefix to add to passage/document text (e.g., 'passage: ' for E5 models). None if not needed."),
        required=False,
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
