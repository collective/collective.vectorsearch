# -*- coding: utf-8 -*-
import logging

from Products.Five.browser import BrowserView
from Products.statusmessages.interfaces import IStatusMessage

from collective.vectorsearch import _

logger = logging.getLogger("collective.vectorsearch")


class AddVectorIndexView(BrowserView):
    """Browser view for adding a VectorIndex to the catalog."""

    def __call__(self):
        """Handle form submission."""
        if self.request.method == "POST":
            return self.handle_submit()
        return self.index()

    def handle_submit(self):
        """Process the form submission and create the index."""
        # Get form values
        index_id = self.request.form.get("id", "").strip()
        indexed_attrs = self.request.form.get("indexed_attrs", "").strip()

        # Validate
        if not index_id:
            IStatusMessage(self.request).addStatusMessage(
                _("Please provide an index ID."), type="error"
            )
            return self.index()

        # Parse indexed_attrs (comma-separated)
        if indexed_attrs:
            attrs = [attr.strip() for attr in indexed_attrs.split(",")]
            extra = {"indexed_attrs": ",".join(attrs)}
        else:
            extra = None

        # Create the index
        try:
            # context is the portal_catalog's Indexes folder (IAdding)
            self.context.manage_addIndex(index_id, "VectorIndex", extra=extra)

            IStatusMessage(self.request).addStatusMessage(
                _("Vector index '${id}' added successfully.", mapping={"id": index_id}),
                type="info",
            )

            # Redirect back to the manage interface
            return self.request.response.redirect(
                self.context.absolute_url() + "/manage_workspace"
            )

        except Exception as e:
            logger.error(f"Error adding vector index: {e}", exc_info=True)
            IStatusMessage(self.request).addStatusMessage(
                _("Error adding index: ${error}", mapping={"error": str(e)}),
                type="error",
            )
            return self.index()
