# -*- coding: utf-8 -*-
"""Vocabularies for collective.vectorsearch."""

from zope.interface import provider
from zope.schema.interfaces import IVocabularyFactory
from zope.schema.vocabulary import SimpleVocabulary, SimpleTerm
from zope.component import getUtilitiesFor
from collective.vectorsearch.interfaces import IEmbeddingModelProvider


@provider(IVocabularyFactory)
class EmbeddingModelsVocabulary:
    """
    Vocabulary factory that dynamically generates terms from
    all registered IEmbeddingModelProvider utilities.
    """

    def __call__(self, context):
        """
        Generate vocabulary from registered model providers.

        Returns:
            SimpleVocabulary with terms for each registered model
        """
        terms = []

        # Get all registered embedding model providers
        for name, provider in getUtilitiesFor(IEmbeddingModelProvider):
            term = SimpleTerm(
                value=provider.id,
                token=provider.id,
                title=provider.title
            )
            terms.append(term)

        # Sort by title for better UX
        terms.sort(key=lambda t: t.title)

        # Fallback if no providers registered
        if not terms:
            terms.append(SimpleTerm(
                value='gte-small',
                token='gte-small',
                title=u'GTE Small (default)'
            ))

        return SimpleVocabulary(terms)


# Create singleton instance
EmbeddingModelsVocabularyFactory = EmbeddingModelsVocabulary()


@provider(IVocabularyFactory)
class StorageBackendsVocabulary:
    """Vocabulary for storage backend options."""

    def __call__(self, context):
        return SimpleVocabulary([
            SimpleTerm(
                value=u'btrees',
                token=u'btrees',
                title=u'BTrees (Internal)'
            ),
            SimpleTerm(
                value=u'faiss',
                token=u'faiss',
                title=u'FAISS'
            ),
            SimpleTerm(
                value=u'duckdb',
                token=u'duckdb',
                title=u'DuckDB'
            ),
            SimpleTerm(
                value=u'annoy',
                token=u'annoy',
                title=u'Annoy'
            ),
        ])


StorageBackendsVocabularyFactory = StorageBackendsVocabulary()


@provider(IVocabularyFactory)
class ApproximationAlgorithmsVocabulary:
    """Vocabulary for approximation algorithm options."""

    def __call__(self, context):
        return SimpleVocabulary([
            SimpleTerm(
                value=u'exhaustive_cosine',
                token=u'exhaustive_cosine',
                title=u'Exhaustive Cosine Search'
            ),
            SimpleTerm(
                value=u'hnsw',
                token=u'hnsw',
                title=u'HNSW'
            ),
            SimpleTerm(
                value=u'itq_lsh_2stage',
                token=u'itq_lsh_2stage',
                title=u'ITQ LSH 2-stage'
            ),
            SimpleTerm(
                value=u'itq_lsh_3stage',
                token=u'itq_lsh_3stage',
                title=u'ITQ LSH 3-stage'
            ),
        ])


ApproximationAlgorithmsVocabularyFactory = ApproximationAlgorithmsVocabulary()
