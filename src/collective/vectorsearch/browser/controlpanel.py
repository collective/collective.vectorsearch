# -*- coding: utf-8 -*-
"""Control panel for Vector Search settings."""

import logging

from plone import api
from plone.app.registry.browser.controlpanel import (
    ControlPanelFormWrapper,
    RegistryEditForm,
)
from plone.z3cform import layout
from Products.Five.browser.pagetemplatefile import ViewPageTemplateFile
from Products.statusmessages.interfaces import IStatusMessage
from z3c.form import button

from collective.vectorsearch import _
from collective.vectorsearch.interfaces import IVectorIndex, IVectorSearchSettings
from collective.vectorsearch.model_providers import get_backend_info

logger = logging.getLogger("collective.vectorsearch")


class VectorSearchControlPanelForm(RegistryEditForm):
    """Vector Search settings control panel form."""

    schema = IVectorSearchSettings
    schema_prefix = "collective.vectorsearch"
    label = _("Vector Search Settings")
    description = _(
        "Configure vector search parameters. "
        "Changing the embedding model will clear all existing vectors."
    )

    def _get_current_model(self):
        """Get current embedding model from registry."""
        try:
            return api.portal.get_registry_record(
                "collective.vectorsearch.embedding_model", default="all-minilm-l6"
            )
        except Exception:
            return "all-minilm-l6"

    def _get_vector_indexes(self):
        """Get all vector indexes from the catalog."""
        try:
            catalog = api.portal.get_tool("portal_catalog")
            indexes = []
            for index_id in catalog.indexes():
                index = catalog.Indexes.get(index_id)
                if index is not None and IVectorIndex.providedBy(index):
                    indexes.append((index_id, index))
            return indexes
        except Exception as e:
            logger.warning(f"Could not get vector indexes: {e}")
            return []

    def _get_incompatible_indexes(self):
        """Get indexes with vectors created by a different model.

        Returns:
            list of tuples: (index_id, indexed_model, current_model)
        """
        current_model = self._get_current_model()
        incompatible = []

        for index_id, index in self._get_vector_indexes():
            try:
                doc_count = index.numObjects()
                if doc_count > 0:
                    indexed_model = index.getIndexedModel()
                    if indexed_model and indexed_model != current_model:
                        incompatible.append((index_id, indexed_model, current_model))
            except Exception as e:
                logger.warning(f"Could not check index {index_id}: {e}")

        return incompatible

    def _clear_all_vector_indexes(self):
        """Clear all vector indexes."""
        indexes = self._get_vector_indexes()
        cleared_count = 0
        for index_id, index in indexes:
            try:
                index.clear()
                logger.info(f"Cleared vector index: {index_id}")
                cleared_count += 1
            except Exception as e:
                logger.error(f"Failed to clear index {index_id}: {e}")
        return cleared_count

    def _reindex_all_vector_indexes(self):
        """Trigger reindexing for all vector indexes."""
        indexes = self._get_vector_indexes()
        index_ids = [idx_id for idx_id, _ in indexes]

        if index_ids:
            try:
                catalog = api.portal.get_tool("portal_catalog")
                catalog.reindexIndex(index_ids, REQUEST=self.request)
                logger.info(f"Triggered reindex for vector indexes: {index_ids}")
            except Exception as e:
                logger.error(f"Failed to reindex: {e}")
                raise

        return len(index_ids)

    @property
    def vector_index_stats(self):
        """Get statistics for all vector indexes."""
        indexes = self._get_vector_indexes()
        current_model = self._get_current_model()
        stats = []
        for index_id, index in indexes:
            try:
                indexed_model = index.getIndexedModel()
                doc_count = index.numObjects()
                stats.append(
                    {
                        "id": index_id,
                        "document_count": doc_count,
                        "vector_count": index.indexSize(),
                        "indexed_model": indexed_model,
                        "is_compatible": (
                            doc_count == 0
                            or indexed_model is None
                            or indexed_model == current_model
                        ),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not get stats for {index_id}: {e}")
                stats.append(
                    {
                        "id": index_id,
                        "document_count": 0,
                        "vector_count": 0,
                        "indexed_model": None,
                        "is_compatible": True,
                        "error": str(e),
                    }
                )
        return stats

    @property
    def has_incompatible_indexes(self):
        """Check if any index has vectors created with a different model."""
        return len(self._get_incompatible_indexes()) > 0

    @property
    def incompatible_indexes(self):
        """Get list of incompatible indexes for display."""
        return self._get_incompatible_indexes()

    @property
    def total_indexes(self):
        """Total number of vector indexes."""
        return len(self.vector_index_stats)

    @property
    def total_documents(self):
        """Total documents across all vector indexes."""
        return sum(s["document_count"] for s in self.vector_index_stats)

    @property
    def total_vectors(self):
        """Total vectors across all vector indexes."""
        return sum(s["vector_count"] for s in self.vector_index_stats)

    @property
    def backend_info(self):
        """Get backend information from registered providers."""
        return get_backend_info()

    @property
    def has_available_backend(self):
        """Check if at least one backend is available."""
        backends = get_backend_info()
        return any(b["available"] for b in backends)

    @button.buttonAndHandler(_("Save"), name="save")
    def handleSave(self, action):
        """Handle save."""
        data, errors = self.extractData()
        if errors:
            self.status = self.formErrorsMessage
            return

        self.applyChanges(data)
        IStatusMessage(self.request).addStatusMessage(_("Changes saved."), "info")

        # Check for incompatible indexes after save
        incompatible = self._get_incompatible_indexes()
        if incompatible:
            index_info = ", ".join(
                [f"{idx[0]} (was: {idx[1]})" for idx in incompatible]
            )
            IStatusMessage(self.request).addStatusMessage(
                _(
                    "Warning: Some indexes have vectors created with a different model: ${indexes}. "
                    "Use 'Clear All Vectors' and then 'Reindex All' to update them.",
                    mapping={"indexes": index_info},
                ),
                "warning",
            )

        self.request.response.redirect(self.request.getURL())

    @button.buttonAndHandler(_("Reindex All"), name="reindex")
    def handleReindex(self, action):
        """Handle reindex all vector indexes."""
        # Check for incompatible indexes first
        incompatible = self._get_incompatible_indexes()
        if incompatible:
            IStatusMessage(self.request).addStatusMessage(
                _(
                    "Cannot reindex: Some indexes have vectors created with a different model. "
                    "Use 'Clear All Vectors' first to remove incompatible vectors."
                ),
                "error",
            )
            self.request.response.redirect(self.request.getURL())
            return

        try:
            count = self._reindex_all_vector_indexes()
            if count > 0:
                IStatusMessage(self.request).addStatusMessage(
                    _(
                        "Reindexing triggered for ${count} vector index(es). "
                        "This may take some time depending on the amount of content.",
                        mapping={"count": count},
                    ),
                    "info",
                )
            else:
                IStatusMessage(self.request).addStatusMessage(
                    _("No vector indexes found to reindex."), "warning"
                )
        except Exception as e:
            IStatusMessage(self.request).addStatusMessage(
                _("Error during reindex: ${error}", mapping={"error": str(e)}), "error"
            )
        self.request.response.redirect(self.request.getURL())

    @button.buttonAndHandler(_("Clear All Vectors"), name="clear")
    def handleClear(self, action):
        """Handle clear all vector indexes."""
        count = self._clear_all_vector_indexes()
        if count > 0:
            IStatusMessage(self.request).addStatusMessage(
                _(
                    "${count} vector index(es) have been cleared. Please reindex your content.",
                    mapping={"count": count},
                ),
                "warning",
            )
        else:
            IStatusMessage(self.request).addStatusMessage(
                _("No vector indexes found to clear."), "warning"
            )
        self.request.response.redirect(self.request.getURL())

    @button.buttonAndHandler(_("Cancel"), name="cancel")
    def handleCancel(self, action):
        """Handle cancel."""
        IStatusMessage(self.request).addStatusMessage(_("Changes canceled."), "info")
        portal_url = api.portal.get().absolute_url()
        self.request.response.redirect(f"{portal_url}/@@overview-controlpanel")


class VectorSearchControlPanelView(ControlPanelFormWrapper):
    """Custom wrapper with statistics display."""

    index = ViewPageTemplateFile("templates/controlpanel.pt")


# Create the wrapped view
VectorSearchControlPanel = layout.wrap_form(
    VectorSearchControlPanelForm, VectorSearchControlPanelView
)
