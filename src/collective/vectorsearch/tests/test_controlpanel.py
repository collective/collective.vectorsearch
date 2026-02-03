# -*- coding: utf-8 -*-
"""Control panel tests for this package."""
from collective.vectorsearch.testing import (
    COLLECTIVE_VECTORSEARCH_INTEGRATION_TESTING,
)
from plone import api
from plone.app.testing import setRoles, TEST_USER_ID
from zope.component import getMultiAdapter

import unittest


class TestControlPanel(unittest.TestCase):
    """Test that Vector Search control panel is properly configured."""

    layer = COLLECTIVE_VECTORSEARCH_INTEGRATION_TESTING

    def setUp(self):
        """Custom shared utility setup for tests."""
        self.portal = self.layer["portal"]
        self.request = self.layer["request"]
        setRoles(self.portal, TEST_USER_ID, ['Manager'])

    def test_controlpanel_view_exists(self):
        """Test that control panel view is accessible."""
        view = getMultiAdapter(
            (self.portal, self.request),
            name="vectorsearch-settings"
        )
        self.assertIsNotNone(view)
        self.assertTrue(hasattr(view, 'schema'))

    def test_controlpanel_registered(self):
        """Test control panel is registered in portal_controlpanel."""
        cp = api.portal.get_tool('portal_controlpanel')
        actions = [a.id for a in cp.listActions()]
        self.assertIn('vectorsearch', actions)

    def test_controlpanel_view_protected(self):
        """Test that regular users cannot access control panel."""
        from AccessControl import Unauthorized

        # Set user as regular member
        setRoles(self.portal, TEST_USER_ID, ['Member'])

        # Attempt to access control panel should raise Unauthorized
        with self.assertRaises(Unauthorized):
            self.portal.restrictedTraverse('@@vectorsearch-settings')

    def test_controlpanel_settings_accessible(self):
        """Test that control panel can access settings."""
        view = getMultiAdapter(
            (self.portal, self.request),
            name="vectorsearch-settings"
        )

        # Check that the view has access to the schema
        from collective.vectorsearch.interfaces import IVectorSearchSettings
        self.assertEqual(view.schema, IVectorSearchSettings)


class TestControlPanelUninstall(unittest.TestCase):
    """Test that control panel is removed on uninstall."""

    layer = COLLECTIVE_VECTORSEARCH_INTEGRATION_TESTING

    def setUp(self):
        """Uninstall the package."""
        self.portal = self.layer["portal"]
        self.request = self.layer["request"]

        try:
            from Products.CMFPlone.utils import get_installer
            self.installer = get_installer(self.portal, self.request)
        except ImportError:
            self.installer = api.portal.get_tool("portal_quickinstaller")

        roles_before = api.user.get_roles(TEST_USER_ID)
        setRoles(self.portal, TEST_USER_ID, ["Manager"])
        self.installer.uninstall_product("collective.vectorsearch")
        setRoles(self.portal, TEST_USER_ID, roles_before)

    def test_controlpanel_removed(self):
        """Test that control panel configlet is removed."""
        cp = api.portal.get_tool('portal_controlpanel')
        actions = [a.id for a in cp.listActions()]
        self.assertNotIn('vectorsearch', actions)
