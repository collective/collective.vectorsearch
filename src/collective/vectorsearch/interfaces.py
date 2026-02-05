# -*- coding: utf-8 -*-
"""Module where all interfaces, events and exceptions live."""

from zope import schema
from zope.interface import Interface, Invalid, invariant
from zope.publisher.interfaces.browser import IDefaultBrowserLayer

from collective.vectorsearch import _


class ICollectiveVectorsearchLayer(IDefaultBrowserLayer):
    """Marker interface that defines a browser layer."""


class IVectorIndex(Interface):
    """Marker interface for IVectorIndex"""


class IVectorSearchSettings(Interface):
    """Vector Search configuration settings."""

    embedding_model = schema.Choice(
        title=_("Embedding Model"),
        description=_(
            "Select the embedding model for vector search. "
            "WARNING: Changing this model requires clearing all existing vectors and reindexing all content. "
            "Different models produce incompatible vector dimensions. "
            "This operation cannot be undone."
        ),
        vocabulary="collective.vectorsearch.embedding_models",
        default="all-minilm-l6",
        required=True,
    )

    embedding_chunk_size = schema.Int(
        title=_("Text Chunk Size"),
        description=_(
            "Maximum character length for splitting long text into chunks. "
            "Long documents are divided into multiple chunks of this size for vectorization. "
            "Each chunk is embedded separately."
        ),
        default=500,
        required=True,
        min=100,
        max=10000,
    )

    storage_backend = schema.Choice(
        title=_("Storage Backend"),
        description=_(
            "Storage system for vector data. "
            "BTrees (internal), or external vector databases (FAISS, DuckDB, etc.)"
        ),
        vocabulary="collective.vectorsearch.storage_backends",
        default="btrees",
        required=True,
    )

    external_db_uri = schema.URI(
        title=_("External Database URI"),
        description=_(
            "URI for external vector database connection. "
            "Required when storage backend is not 'btrees'. "
            "Examples: 'http://localhost:8080/faiss', 'duckdb:///path/to/db.duckdb'"
        ),
        required=False,
    )

    approximation_algorithm = schema.Choice(
        title=_("Approximation Algorithm"),
        description=_(
            "Search strategy for similarity calculation. "
            "Based on LSH cascade research (lsh-cascade-poc). "
            "Currently only 'Exhaustive Cosine' is implemented."
        ),
        vocabulary="collective.vectorsearch.approximation_algorithms",
        default="exhaustive_cosine",
        required=True,
    )

    pivot_threshold = schema.Int(
        title=_("Pivot Threshold (Stage 1)"),
        description=_(
            "Threshold for pivot-based filtering in Stage 1. "
            "Higher values = more candidates retained. "
            "Recommended: 20 (89.8% recall) or 15 (85.9% recall)."
        ),
        default=20,
        required=False,
        min=1,
        max=100,
    )

    hamming_distance_threshold = schema.Int(
        title=_("Hamming Distance Threshold (Stage 2)"),
        description=_(
            "Maximum Hamming distance for candidate filtering in Stage 2. "
            "Lower values = stricter filtering, fewer candidates. "
            "Recommended: 2-10. Default: 3."
        ),
        default=3,
        required=False,
        min=0,
        max=128,
    )

    @invariant
    def validate_external_db_uri(obj):
        """Validate that external_db_uri is provided when storage_backend is not btrees."""
        if obj.storage_backend != "btrees" and not obj.external_db_uri:
            raise Invalid(
                _(
                    "External Database URI is required when storage backend is not BTrees."
                )
            )


class IEmbeddingModelProvider(Interface):
    """Marker interface for embedding model providers.

    Named utilities implementing this interface will be available
    for selection in the control panel.
    """

    id = schema.ASCIILine(
        title=_("Model ID"),
        description=_("Unique identifier for this embedding model"),
        required=True,
    )

    title = schema.TextLine(
        title=_("Display Title"),
        description=_("Human-readable name shown in UI"),
        required=True,
    )

    description = schema.Text(
        title=_("Description"),
        description=_("Description of this model and its use cases"),
        required=False,
    )

    model_name = schema.TextLine(
        title=_("Model Name"),
        description=_("Internal model name (e.g., HuggingFace model ID)"),
        required=True,
    )

    use_cache_dir = schema.Bool(
        title=_("Use Cache Directory"),
        description=_("Whether to cache model files for faster loading"),
        default=False,
        required=True,
    )

    vector_dimensions = schema.Int(
        title=_("Vector Dimensions"),
        description=_("Dimensionality of embedding vectors produced by this model"),
        required=True,
    )

    hash_length = schema.Int(
        title=_("Hash Length"),
        description=_("Length of binary hash for ITQ quantization"),
        default=128,
        required=True,
    )

    itq_boundary_name = schema.TextLine(
        title=_("ITQ Boundary Name"),
        description=_("Name of pre-trained ITQ boundary vector (if available)"),
        required=False,
    )

    available_similarity_methods = schema.List(
        title=_("Available Similarity Methods"),
        description=_("List of similarity calculation methods supported"),
        value_type=schema.Choice(values=["cosine", "itq_hamming", "euclidean"]),
        required=True,
    )

    embedding_class = schema.ASCIILine(
        title=_("Embedding Class"),
        description=_(
            "Name of embedding class to use (e.g., SentenceTransformerEmbedding, FastEmbedEmbedding)"
        ),
        default="SentenceTransformerEmbedding",
        required=True,
    )

    query_prefix = schema.TextLine(
        title=_("Query Prefix"),
        description=_(
            "Prefix to add to query text (e.g., 'query: ' for E5 models). None if not needed."
        ),
        required=False,
    )

    passage_prefix = schema.TextLine(
        title=_("Passage Prefix"),
        description=_(
            "Prefix to add to passage/document text (e.g., 'passage: ' for E5 models). None if not needed."
        ),
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
