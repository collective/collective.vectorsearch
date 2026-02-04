# -*- coding: utf-8 -*-
"""Embedding model provider implementations."""

from zope.interface import implementer
from collective.vectorsearch.interfaces import IEmbeddingModelProvider
from collective.vectorsearch import embedding as emb_module
from collective.vectorsearch.vector_index import ModelCache
import logging

logger = logging.getLogger('collective.vectorsearch')


@implementer(IEmbeddingModelProvider)
class BaseEmbeddingModelProvider:
    """Base class for embedding model providers."""

    # Class attributes (subclasses override)
    id = None
    title = None
    description = None
    model_name = None
    use_cache_dir = False
    vector_dimensions = None
    hash_length = 128
    itq_boundary_name = None
    available_similarity_methods = ['cosine']
    embedding_class = 'SentenceTransformerEmbedding'  # Default

    def get_embedding_instance(self, chunk_size=500, prefix_query=None):
        """Factory method - creates appropriate embedding instance based on embedding_class."""
        # Get embedding class
        EmbeddingClass = getattr(emb_module, self.embedding_class)

        # Handle different initialization patterns
        if self.embedding_class == 'SentenceTransformerEmbedding':
            cache = ModelCache()
            model = cache.get_model(self.model_name)
            return EmbeddingClass(model, chunk_size=chunk_size, prefix_query=prefix_query)

        elif self.embedding_class == 'FastEmbedEmbedding':
            # FastEmbed doesn't use ModelCache (has its own caching)
            return EmbeddingClass(
                self.model_name,
                chunk_size=chunk_size,
                prefix_query=prefix_query
            )

        else:
            raise ValueError(f"Unknown embedding_class: {self.embedding_class}")

    def get_itq_boundary(self):
        """Phase 1: Not implemented yet."""
        return None

    def get_pivot_data(self):
        """Phase 1: Not implemented yet."""
        return None


class GTESmallProvider(BaseEmbeddingModelProvider):
    """Provider for thenlper/gte-small model."""

    id = 'gte-small'
    title = u'GTE Small'
    description = u'General Text Embeddings (Small) - 384 dimensions, multilingual'
    model_name = 'thenlper/gte-small'
    vector_dimensions = 384
    itq_boundary_name = 'gte_small_itq'
    available_similarity_methods = ['cosine']  # Phase 1ではcosineのみ
    embedding_class = 'SentenceTransformerEmbedding'


class E5BaseProvider(BaseEmbeddingModelProvider):
    """Provider for intfloat/multilingual-e5-base model."""

    id = 'e5-base-multilingual'
    title = u'E5 Base Multilingual'
    description = u'E5 Base Multilingual - 768 dimensions, supports 100+ languages'
    model_name = 'intfloat/multilingual-e5-base'
    vector_dimensions = 768
    itq_boundary_name = 'e5_base_itq'
    available_similarity_methods = ['cosine']
    embedding_class = 'SentenceTransformerEmbedding'


class E5FastEmbedProvider(BaseEmbeddingModelProvider):
    """Provider for E5 Base Multilingual with FastEmbed."""

    id = 'e5-base-multilingual-fastembed'
    title = u'E5 Base Multilingual (FastEmbed)'
    description = u'E5 Base Multilingual with FastEmbed - 768 dimensions, ONNX optimized, 100+ languages'
    model_name = 'intfloat/multilingual-e5-base'
    vector_dimensions = 768
    itq_boundary_name = 'e5_base_itq'
    available_similarity_methods = ['cosine']
    embedding_class = 'FastEmbedEmbedding'
    use_cache_dir = True  # FastEmbed uses local model cache
