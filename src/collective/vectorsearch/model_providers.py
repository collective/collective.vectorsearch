# -*- coding: utf-8 -*-
"""Embedding model provider implementations."""

import logging

from zope.component import getUtilitiesFor
from zope.interface import implementer

from collective.vectorsearch import embedding as emb_module
from collective.vectorsearch.data_loader import (
    load_itq_data,
    load_pivot_data,
    load_voronoi_data,
    validate_itq_data,
    validate_pivot_data,
    validate_voronoi_data,
)
from collective.vectorsearch.interfaces import IEmbeddingModelProvider
from collective.vectorsearch.vector_index import ModelCache

logger = logging.getLogger("collective.vectorsearch")


# Check for optional dependencies
try:
    from fastembed import TextEmbedding  # noqa: F401
    from fastembed.common.model_description import ModelSource, PoolingType

    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def check_fastembed_available():
    """Check if fastembed is available."""
    return HAS_FASTEMBED


def check_sentence_transformers_available():
    """Check if sentence_transformers (GPU support) is available."""
    return HAS_SENTENCE_TRANSFORMERS


_CUSTOM_MODELS_REGISTERED = set()


def _register_fastembed_custom_model(
    model_name,
    dim,
    pooling,
    normalization,
    hf_source=None,
    model_file="onnx/model.onnx",
    additional_files=None,
):
    """Register a custom model with FastEmbed if not already registered.

    Idempotent - safe to call multiple times.
    """
    if model_name in _CUSTOM_MODELS_REGISTERED:
        return

    if not HAS_FASTEMBED:
        return

    try:
        TextEmbedding.add_custom_model(
            model=model_name,
            pooling=pooling,
            normalization=normalization,
            sources=ModelSource(hf=hf_source or model_name),
            dim=dim,
            model_file=model_file,
            additional_files=additional_files or [],
        )
    except ValueError:
        # Already registered (add_custom_model raises ValueError for duplicates)
        pass

    _CUSTOM_MODELS_REGISTERED.add(model_name)


@implementer(IEmbeddingModelProvider)
class BaseEmbeddingModelProvider:
    """Base class for embedding model providers.

    Subclasses should override class attributes to define their specific
    model configuration. External packages can extend this to add new models.

    Backend Attributes:
        backend: Backend identifier ('fastembed' or 'sentence_transformers')
        backend_name: Human-readable name for display
        requires_gpu: Whether this model requires/benefits from GPU
        extras_name: Buildout extras name to install (None for default)
    """

    # Class attributes (subclasses override)
    id = None
    title = None
    description = None
    model_name = None
    use_cache_dir = False
    vector_dimensions = None
    hash_length = 128
    itq_boundary_name = None
    available_similarity_methods = ["cosine"]
    embedding_class = "SentenceTransformerEmbedding"  # Default
    query_prefix = None
    passage_prefix = None

    # Data file configuration
    # If None, uses self.id with hyphens converted to underscores
    # Multiple providers can share the same data_file_id (e.g., CPU/GPU variants)
    data_file_id = None

    # Backend configuration
    backend = "sentence_transformers"  # 'fastembed' or 'sentence_transformers'
    backend_name = "Sentence Transformers"  # Human-readable name
    requires_gpu = False  # True if GPU is required/recommended
    extras_name = None  # Buildout extras name (None = default, 'gpu' = [gpu])

    @classmethod
    def is_available(cls):
        """Check if this provider can be used with current installed packages.

        Returns:
            bool: True if the required backend is available
        """
        if cls.backend == "fastembed":
            return HAS_FASTEMBED
        elif cls.backend == "sentence_transformers":
            return HAS_SENTENCE_TRANSFORMERS
        return False

    @classmethod
    def get_unavailable_reason(cls):
        """Get reason why this provider is unavailable.

        Returns:
            str or None: Reason string if unavailable, None if available
        """
        if cls.is_available():
            return None

        if cls.extras_name:
            return (
                f"{cls.backend_name} is not installed. "
                f"Add [{cls.extras_name}] extras to enable."
            )
        else:
            return (
                f"{cls.backend_name} is not installed. "
                "Please reinstall collective.vectorsearch."
            )

    def get_embedding_instance(
        self, chunk_size=500, prefix_query=None, prefix_passage=None
    ):
        """Factory method - creates appropriate embedding instance based on embedding_class."""
        # Check availability first
        if not self.is_available():
            raise ImportError(self.get_unavailable_reason())

        # Get embedding class
        EmbeddingClass = getattr(emb_module, self.embedding_class)

        # Handle different initialization patterns
        if self.embedding_class == "SentenceTransformerEmbedding":
            cache = ModelCache()
            model = cache.get_model(self.model_name)
            return EmbeddingClass(
                model,
                chunk_size=chunk_size,
                prefix_query=prefix_query,
                prefix_passage=prefix_passage,
            )

        elif self.embedding_class == "FastEmbedEmbedding":
            # FastEmbed doesn't use ModelCache (has its own caching)
            return EmbeddingClass(
                self.model_name,
                chunk_size=chunk_size,
                prefix_query=prefix_query,
                prefix_passage=prefix_passage,
            )

        else:
            raise ValueError(f"Unknown embedding_class: {self.embedding_class}")

    def _get_data_file_id(self):
        """Get the data file identifier for this provider.

        Returns data_file_id if set, otherwise converts id from
        hyphens to underscores (e.g., 'all-minilm-l6' -> 'all_minilm_l6').
        """
        if self.data_file_id:
            return self.data_file_id
        return self.id.replace("-", "_") if self.id else None

    def get_itq_boundary(self):
        """Load and return ITQ boundary data for this model.

        Returns:
            ITQData instance or None if not available/invalid
        """
        data_id = self._get_data_file_id()
        if not data_id:
            return None

        itq_data = load_itq_data(data_id)
        if itq_data is None:
            return None

        # Validate dimensions
        if not validate_itq_data(itq_data, self.vector_dimensions, self.hash_length):
            return None

        return itq_data

    def get_pivot_data(self):
        """Load and return pivot data for this model.

        Returns:
            PivotData instance or None if not available/invalid
        """
        data_id = self._get_data_file_id()
        if not data_id:
            return None

        pivot_data = load_pivot_data(data_id)
        if pivot_data is None:
            return None

        # Validate dimensions (default 8 pivots)
        if not validate_pivot_data(pivot_data, self.vector_dimensions):
            return None

        return pivot_data

    def get_voronoi_data(self):
        """Load and return Voronoi centroid data for this model.

        Returns:
            VoronoiData instance or None if not available/invalid
        """
        data_id = self._get_data_file_id()
        if not data_id:
            return None

        voronoi_data = load_voronoi_data(data_id)
        if voronoi_data is None:
            return None

        if not validate_voronoi_data(voronoi_data, self.vector_dimensions):
            return None

        return voronoi_data


# =============================================================================
# Model Providers
# =============================================================================


class AllMiniLMProvider(BaseEmbeddingModelProvider):
    """Provider for all-MiniLM-L6-v2 with FastEmbed.

    Lightweight, fast model suitable for CPU-only environments.
    """

    id = "all-minilm-l6"
    title = "MiniLM L6 v2 (FastEmbed)"
    description = (
        "Sentence Transformers MiniLM - 384 dimensions, ONNX optimized, fast, English"
    )
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dimensions = 384
    itq_boundary_name = "minilm_itq"
    available_similarity_methods = ["cosine"]
    embedding_class = "FastEmbedEmbedding"
    use_cache_dir = True
    query_prefix = None
    passage_prefix = None

    # Backend configuration
    backend = "fastembed"
    backend_name = "FastEmbed (CPU/ONNX)"
    requires_gpu = False
    extras_name = None  # Default installation


class E5BaseMultilingualProvider(BaseEmbeddingModelProvider):
    """Provider for E5 Base Multilingual with FastEmbed.

    Multilingual model (100+ languages) with ONNX optimization for CPU.
    Uses add_custom_model() to register with FastEmbed since this model
    is not in FastEmbed's built-in supported list.
    """

    id = "e5-base-multilingual"
    title = "E5 Base Multilingual (FastEmbed)"
    description = (
        "E5 Base Multilingual - 768 dimensions, ONNX optimized, 100+ languages"
    )
    model_name = "intfloat/multilingual-e5-base"
    vector_dimensions = 768
    itq_boundary_name = "e5_base_itq"
    available_similarity_methods = ["cosine"]
    embedding_class = "FastEmbedEmbedding"
    use_cache_dir = True
    query_prefix = "query: "
    passage_prefix = "passage: "

    # Backend configuration
    backend = "fastembed"
    backend_name = "FastEmbed (CPU/ONNX)"
    requires_gpu = False
    extras_name = None  # Default installation

    def get_embedding_instance(
        self, chunk_size=500, prefix_query=None, prefix_passage=None
    ):
        """Create FastEmbed embedding after registering custom model.

        intfloat/multilingual-e5-base is not in FastEmbed's built-in model list,
        but ONNX files exist on HuggingFace Hub. We register it as a custom model
        before creating the embedding instance.
        """
        _register_fastembed_custom_model(
            model_name=self.model_name,
            dim=self.vector_dimensions,
            pooling=PoolingType.MEAN,
            normalization=True,
            model_file="onnx/model.onnx",
            additional_files=["sentencepiece.bpe.model"],
        )
        return super().get_embedding_instance(
            chunk_size=chunk_size,
            prefix_query=prefix_query,
            prefix_passage=prefix_passage,
        )


class E5BaseMultilingualGPUProvider(BaseEmbeddingModelProvider):
    """Provider for E5 Base Multilingual with SentenceTransformers (GPU).

    GPU-accelerated version for faster processing with CUDA.
    Requires the [gpu] extras to be installed.
    """

    id = "e5-base-multilingual-gpu"
    title = "E5 Base Multilingual (GPU)"
    description = (
        "E5 Base Multilingual - 768 dimensions, GPU accelerated, 100+ languages"
    )
    model_name = "intfloat/multilingual-e5-base"
    vector_dimensions = 768
    itq_boundary_name = "e5_base_itq"
    available_similarity_methods = ["cosine"]
    embedding_class = "SentenceTransformerEmbedding"
    query_prefix = "query: "
    passage_prefix = "passage: "

    # Share ITQ/pivot data with CPU variant
    data_file_id = "e5_base_multilingual"

    # Backend configuration
    backend = "sentence_transformers"
    backend_name = "Sentence Transformers (GPU)"
    requires_gpu = True
    extras_name = "gpu"


# =============================================================================
# Utility functions
# =============================================================================


def get_available_providers():
    """Get list of all available provider classes."""
    all_providers = [
        AllMiniLMProvider,
        E5BaseMultilingualProvider,
        E5BaseMultilingualGPUProvider,
    ]
    return [p for p in all_providers if p.is_available()]


def get_all_providers():
    """Get list of all provider classes (available and unavailable)."""
    return [
        AllMiniLMProvider,
        E5BaseMultilingualProvider,
        E5BaseMultilingualGPUProvider,
    ]


def get_backend_info():
    """Get backend information from registered providers.

    Collects unique backends from all registered IEmbeddingModelProvider
    utilities and returns their availability status.

    Returns:
        list of dict: Backend information sorted by extras_name (default first)
    """
    backends = {}

    # Collect unique backends from registered providers
    for _name, provider in getUtilitiesFor(IEmbeddingModelProvider):
        backend_id = provider.backend
        if backend_id not in backends:
            backends[backend_id] = {
                "id": backend_id,
                "name": provider.backend_name,
                "requires_gpu": provider.requires_gpu,
                "extras_name": provider.extras_name,
                "available": provider.is_available(),
            }

    # Convert to list and sort (default first, then by extras_name)
    result = list(backends.values())
    result.sort(key=lambda x: (x["extras_name"] or "", x["name"]))

    return result


# Legacy function for backwards compatibility
def get_backend_status():
    """Get status of available backends (legacy).

    Deprecated: Use get_backend_info() instead for more detailed information.

    Returns:
        dict: Backend availability status
    """
    return {
        "fastembed": {
            "available": HAS_FASTEMBED,
            "name": "FastEmbed (CPU/ONNX)",
            "extras": "(default)",
        },
        "sentence_transformers": {
            "available": HAS_SENTENCE_TRANSFORMERS,
            "name": "Sentence Transformers (GPU)",
            "extras": "[gpu]",
        },
    }
