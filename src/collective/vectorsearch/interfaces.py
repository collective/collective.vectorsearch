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
        description=_(u"Select the embedding model to use for vector search"),
        vocabulary="collective.vectorsearch.embedding_models",
        default=u"gte-small",
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

    stage1_retrieval_count = schema.Int(
        title=_(u"Stage 1: Hamming Distance Pre-filtering"),
        description=_(
            u"Number of candidates to retrieve in the first stage using ITQ binary hash Hamming distance. "
            u"Applied only when using multi-stage ITQ LSH algorithms. "
            u"Leave empty to skip this stage."
        ),
        required=False,
        min=1,
        max=100000,
    )

    stage2_retrieval_count = schema.Int(
        title=_(u"Stage 2: Cosine Similarity Re-ranking (Top-K)"),
        description=_(
            u"Number of top candidates from Stage 1 to re-rank using full cosine similarity. "
            u"Applied only when using multi-stage ITQ LSH algorithms. "
            u"Leave empty to skip this stage."
        ),
        required=False,
        min=1,
        max=100000,
    )

    stage3_retrieval_count = schema.Int(
        title=_(u"Stage 3: Final Result Count"),
        description=_(
            u"Number of final results to return after all stages. "
            u"Applied to all algorithms. "
            u"Leave empty to use system default."
        ),
        required=False,
        min=1,
        max=10000,
    )

    pivot_threshold = schema.Int(
        title=_(u"Pivot Threshold"),
        description=_(
            u"Threshold for pivot-based filtering (used in ITQ algorithms). "
            u"Higher values = more candidates retained. "
            u"Recommended: 20 (46.2% reduction, 89.8% recall) or 15 (73.9% reduction, 85.9% recall)."
        ),
        default=20,
        required=False,
        min=1,
        max=100,
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
