# -*- coding: utf-8 -*-
"""Control panel for Vector Search settings."""
from plone.app.registry.browser.controlpanel import RegistryEditForm
from collective.vectorsearch.interfaces import IVectorSearchSettings
from collective.vectorsearch import _


class VectorSearchControlPanel(RegistryEditForm):
    """Vector Search settings control panel."""

    schema = IVectorSearchSettings
    schema_prefix = "collective.vectorsearch"
    label = _(u"Vector Search Settings")
    description = _(
        u"Configure vector search parameters based on LSH cascade research. "
        u"See field help text for implementation status and usage guidelines. "
        u"These settings apply to newly created indexes only."
    )
