# -*- coding: utf-8 -*-
"""Vocabularies for collective.vectorsearch."""

from zope.component import getUtilitiesFor
from zope.interface import provider
from zope.schema.interfaces import IVocabularyFactory
from zope.schema.vocabulary import SimpleTerm, SimpleVocabulary

from collective.vectorsearch.interfaces import IEmbeddingModelProvider


@provider(IVocabularyFactory)
class EmbeddingModelsVocabulary:
    """
    Vocabulary factory that dynamically generates terms from
    all registered IEmbeddingModelProvider utilities.

    Only shows models that are currently available (have required packages installed).
    """

    def __call__(self, context):
        """
        Generate vocabulary from registered model providers.

        Returns:
            SimpleVocabulary with terms for each available model
        """
        terms = []

        # Get all registered embedding model providers
        for _name, model_provider in getUtilitiesFor(IEmbeddingModelProvider):
            # Only include available providers
            if (
                hasattr(model_provider, "is_available")
                and not model_provider.is_available()
            ):
                continue

            term = SimpleTerm(
                value=model_provider.id,
                token=model_provider.id,
                title=model_provider.title,
            )
            terms.append(term)

        # Sort by title for better UX
        terms.sort(key=lambda t: t.title)

        # Fallback if no providers are available
        if not terms:
            terms.append(
                SimpleTerm(
                    value="all-minilm-l6",
                    token="all-minilm-l6",
                    title="MiniLM L6 v2 (unavailable - install fastembed)",
                )
            )

        return SimpleVocabulary(terms)


# Create singleton instance
EmbeddingModelsVocabularyFactory = EmbeddingModelsVocabulary()


@provider(IVocabularyFactory)
class AllEmbeddingModelsVocabulary:
    """
    Vocabulary factory that shows ALL registered models,
    including unavailable ones (with availability status in title).

    Used for displaying model information in the control panel.
    """

    def __call__(self, context):
        """
        Generate vocabulary from all registered model providers.

        Returns:
            SimpleVocabulary with terms for all models (available and unavailable)
        """
        terms = []

        # Get all registered embedding model providers
        for _name, model_provider in getUtilitiesFor(IEmbeddingModelProvider):
            is_available = (
                not hasattr(model_provider, "is_available")
                or model_provider.is_available()
            )

            if is_available:
                title = model_provider.title
            else:
                title = "{} (unavailable)".format(model_provider.title)

            term = SimpleTerm(
                value=model_provider.id, token=model_provider.id, title=title
            )
            terms.append(term)

        # Sort by title for better UX
        terms.sort(key=lambda t: t.title)

        return SimpleVocabulary(terms)


# Create singleton instance
AllEmbeddingModelsVocabularyFactory = AllEmbeddingModelsVocabulary()


@provider(IVocabularyFactory)
class StorageBackendsVocabulary:
    """Vocabulary for storage backend options."""

    def __call__(self, context):
        return SimpleVocabulary(
            [
                SimpleTerm(value="btrees", token="btrees", title="BTrees (Internal)"),
                SimpleTerm(value="faiss", token="faiss", title="FAISS"),
                SimpleTerm(value="duckdb", token="duckdb", title="DuckDB"),
                SimpleTerm(value="annoy", token="annoy", title="Annoy"),
            ]
        )


StorageBackendsVocabularyFactory = StorageBackendsVocabulary()


@provider(IVocabularyFactory)
class ApproximationAlgorithmsVocabulary:
    """Vocabulary for approximation algorithm options."""

    def __call__(self, context):
        return SimpleVocabulary(
            [
                SimpleTerm(
                    value="exhaustive_cosine",
                    token="exhaustive_cosine",
                    title="Exhaustive Cosine Search",
                ),
                SimpleTerm(value="hnsw", token="hnsw", title="HNSW"),
                SimpleTerm(
                    value="itq_lsh_2stage",
                    token="itq_lsh_2stage",
                    title="ITQ LSH 2-stage",
                ),
                SimpleTerm(
                    value="itq_lsh_3stage",
                    token="itq_lsh_3stage",
                    title="ITQ LSH 3-stage",
                ),
            ]
        )


ApproximationAlgorithmsVocabularyFactory = ApproximationAlgorithmsVocabulary()
