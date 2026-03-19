# -*- coding: utf-8 -*-
"""Control panel for Vector Search settings."""

import logging
import traceback

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
from collective.vectorsearch.annotations import clear_vector_data
from collective.vectorsearch.interfaces import IVectorIndex, IVectorSearchSettings
from collective.vectorsearch.model_providers import get_backend_info
from collective.vectorsearch.subscribers import compute_and_store_vectors

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_model(self):
        """Get current embedding model from registry."""
        try:
            return api.portal.get_registry_record(
                "collective.vectorsearch.embedding_model", default="all-minilm-l6"
            )
        except Exception:
            return "all-minilm-l6"

    def _get_all_brains(self):
        """Get all content brains from the catalog.

        Always uses path=portal_path to scope results to the current
        Plone site.  ZCatalog.searchResults() with no arguments returns
        an empty result set — this helper prevents that pitfall.
        """
        catalog = api.portal.get_tool("portal_catalog")
        portal_path = "/".join(api.portal.get().getPhysicalPath())
        return catalog.unrestrictedSearchResults(path=portal_path)

    def _get_vector_indexes(self):
        """Get all VectorIndex instances from the catalog."""
        try:
            catalog = api.portal.get_tool("portal_catalog")
            indexes = []
            for index_id in catalog.indexes():
                index = catalog.Indexes.get(index_id)
                if index is None:
                    continue

                is_vector_index = IVectorIndex.providedBy(index)
                meta_type = getattr(index, "meta_type", None)

                # Fall back to meta_type for pickled objects without interface
                if not is_vector_index and meta_type == "VectorIndex":
                    is_vector_index = True
                    logger.debug(
                        "VectorIndex '%s' detected by meta_type "
                        "(interface not applied)",
                        index_id,
                    )

                if is_vector_index:
                    indexes.append((index_id, index))

            return indexes
        except Exception as e:
            logger.warning("Could not get vector indexes: %s", e)
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
                logger.warning("Could not check index %s: %s", index_id, e)

        return incompatible

    # ------------------------------------------------------------------
    # Clear / Reindex operations
    # ------------------------------------------------------------------

    def _clear_all_vector_indexes(self):
        """Clear all vector indexes and related catalog data.

        Clears in order:
        1. VectorIndex internal data (paths, counts, model info)
        2. pivot1-8 KeywordIndex data
        3. Content annotations (vectors, ITQ hashes, pivot distances)

        After clearing, catalog metadata columns (itq_hashes, llm_vector)
        will return None for all objects on next access.
        """
        catalog = api.portal.get_tool("portal_catalog")
        cleared_count = 0

        # 1. Clear VectorIndex instances
        for index_id, index in self._get_vector_indexes():
            try:
                index.clear()
                cleared_count += 1
            except Exception as e:
                logger.error("Failed to clear index %s: %s", index_id, e)

        # 2. Clear pivot1-8 and voronoi_cells KeywordIndexes
        for idx_name in [f"pivot{i}" for i in range(1, 9)] + ["voronoi_cells"]:
            if idx_name in catalog.Indexes:
                try:
                    catalog.Indexes[idx_name].clear()
                except Exception as e:
                    logger.error("Failed to clear %s: %s", idx_name, e)

        # 3. Clear content annotations
        annotation_count = 0
        for brain in self._get_all_brains():
            try:
                obj = brain.getObject()
                clear_vector_data(obj)
                annotation_count += 1
            except Exception:
                continue

        logger.info(
            "Cleared %d vector index(es), annotations from %d objects",
            cleared_count,
            annotation_count,
        )
        return cleared_count

    def _reindex_all_vector_indexes(self):
        """Reindex all vector-related indexes and metadata.

        Two-phase approach per object:
          1. compute_and_store_vectors()  — populate annotations
          2. catalog.catalog_object()     — update indexes + metadata

        Uses catalog_object() directly (not obj.reindexObject()) because:
        - Bypasses collective.indexing queue for immediate execution
        - Annotations must exist before catalog_object() since ZCatalog
          updates metadata BEFORE processing indexes

        Returns:
            tuple: (reindexed_count, computed_count, error_count)
        """
        catalog = api.portal.get_tool("portal_catalog")
        indexes = self._get_vector_indexes()
        vector_index_ids = [idx_id for idx_id, _ in indexes]

        if not vector_index_ids:
            return 0, 0, 0

        pivot_ids = [f"pivot{i}" for i in range(1, 9) if f"pivot{i}" in catalog.Indexes]
        voronoi_ids = ["voronoi_cells"] if "voronoi_cells" in catalog.Indexes else []
        all_idxs = vector_index_ids + pivot_ids + voronoi_ids

        brains = list(self._get_all_brains())
        logger.info(
            "Starting vector reindex: %d objects, indexes: %s",
            len(brains),
            all_idxs,
        )

        reindexed = 0
        computed = 0
        errors = 0
        for brain in brains:
            try:
                obj = brain.getObject()
                uid = brain.getPath()

                # Phase 1: Populate annotations
                chunks = compute_and_store_vectors(obj)
                if chunks > 0:
                    computed += 1

                # Phase 2: Update catalog indexes + metadata
                catalog.catalog_object(obj, uid, idxs=all_idxs, update_metadata=1)

                reindexed += 1
            except Exception as e:
                logger.warning("Failed to reindex %s: %s", brain.getPath(), e)
                errors += 1

        logger.info(
            "Vector reindex complete: %d/%d objects, %d with vectors, %d errors",
            reindexed,
            len(brains),
            computed,
            errors,
        )
        return reindexed, computed, errors

    # ------------------------------------------------------------------
    # Template properties
    # ------------------------------------------------------------------

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
                vector_count = index.indexSize()

                itq_pivot_stats = {}
                if hasattr(index, "getITQPivotStats"):
                    itq_pivot_stats = index.getITQPivotStats()

                stats.append(
                    {
                        "id": index_id,
                        "document_count": doc_count,
                        "vector_count": vector_count,
                        "indexed_model": indexed_model,
                        "is_compatible": (
                            doc_count == 0
                            or indexed_model is None
                            or indexed_model == current_model
                        ),
                        "itq_hash_count": itq_pivot_stats.get("itq_hashes", 0),
                        "itq_hash_chunks": itq_pivot_stats.get("itq_hashes_chunks", 0),
                        "pivot_distance_count": itq_pivot_stats.get(
                            "pivot_distances", 0
                        ),
                        "pivot_distance_chunks": itq_pivot_stats.get(
                            "pivot_distances_chunks", 0
                        ),
                        "itq_data_available": itq_pivot_stats.get(
                            "itq_data_available", False
                        ),
                        "pivot_data_available": itq_pivot_stats.get(
                            "pivot_data_available", False
                        ),
                    }
                )
            except Exception as e:
                logger.warning(
                    "Could not get stats for %s: %s\n%s",
                    index_id,
                    e,
                    traceback.format_exc(),
                )
                stats.append(
                    {
                        "id": index_id,
                        "document_count": 0,
                        "vector_count": 0,
                        "indexed_model": None,
                        "is_compatible": True,
                        "itq_hash_count": 0,
                        "itq_hash_chunks": 0,
                        "pivot_distance_count": 0,
                        "pivot_distance_chunks": 0,
                        "itq_data_available": False,
                        "pivot_data_available": False,
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
    def backend_info(self):
        """Get backend information from registered providers."""
        return get_backend_info()

    @property
    def has_available_backend(self):
        """Check if at least one backend is available."""
        backends = get_backend_info()
        return any(b["available"] for b in backends)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    @button.buttonAndHandler(_("Save"), name="save")
    def handleSave(self, action):
        """Handle save."""
        data, errors = self.extractData()
        if errors:
            self.status = self.formErrorsMessage
            return

        self.applyChanges(data)
        IStatusMessage(self.request).addStatusMessage(_("Changes saved."), "info")

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
            reindexed, computed, errors = self._reindex_all_vector_indexes()
            if reindexed > 0:
                IStatusMessage(self.request).addStatusMessage(
                    _(
                        "Reindexed ${reindexed} objects: "
                        "${computed} with vectors computed, ${errors} errors.",
                        mapping={
                            "reindexed": reindexed,
                            "computed": computed,
                            "errors": errors,
                        },
                    ),
                    "info",
                )
            elif errors > 0:
                IStatusMessage(self.request).addStatusMessage(
                    _(
                        "Reindex failed: ${errors} errors occurred. "
                        "Check server logs for details.",
                        mapping={"errors": errors},
                    ),
                    "error",
                )
            else:
                if not self._get_vector_indexes():
                    IStatusMessage(self.request).addStatusMessage(
                        _("No vector indexes found in catalog."), "warning"
                    )
                else:
                    total = len(list(self._get_all_brains()))
                    IStatusMessage(self.request).addStatusMessage(
                        _(
                            "No content objects to reindex "
                            "(${total} objects in catalog).",
                            mapping={"total": total},
                        ),
                        "warning",
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
