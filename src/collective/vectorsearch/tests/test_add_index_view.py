# -*- coding: utf-8 -*-
"""AddVectorIndexView tests for this package."""
from collective.vectorsearch.testing import (
    COLLECTIVE_VECTORSEARCH_INTEGRATION_TESTING,
)
from plone import api
from plone.app.testing import setRoles, TEST_USER_ID

import unittest


class TestAddVectorIndexView(unittest.TestCase):
    """Test that AddVectorIndexView works correctly."""

    layer = COLLECTIVE_VECTORSEARCH_INTEGRATION_TESTING

    def setUp(self):
        """Custom shared utility setup for tests."""
        self.portal = self.layer["portal"]
        self.request = self.layer["request"]
        self.catalog = api.portal.get_tool('portal_catalog')
        setRoles(self.portal, TEST_USER_ID, ['Manager'])

    def test_view_renders(self):
        """Test that the add index view renders."""
        view = self.catalog.Indexes.restrictedTraverse('@@add-vector-index.html')
        html = view()
        self.assertIn('Add Vector Index', html)
        self.assertIn('Indexed Attributes', html)

    def test_create_index_with_id_only(self):
        """Test creating index with just ID."""
        self.request.form = {
            'id': 'test_vector',
            'submit_add': 'Add'
        }
        self.request.method = 'POST'

        view = self.catalog.Indexes.restrictedTraverse('@@add-vector-index.html')
        view()

        # Verify index was created
        self.assertIn('test_vector', self.catalog.Indexes.objectIds())
        index = self.catalog.Indexes['test_vector']
        self.assertEqual(index.meta_type, 'VectorIndex')

        # Verify indexed_attrs defaults to index id
        self.assertEqual(index.indexed_attrs, ['test_vector'])

    def test_create_index_with_indexed_attrs(self):
        """Test creating index with indexed attributes."""
        self.request.form = {
            'id': 'test_vector2',
            'indexed_attrs': 'title, description, text',
            'submit_add': 'Add'
        }
        self.request.method = 'POST'

        view = self.catalog.Indexes.restrictedTraverse('@@add-vector-index.html')
        view()

        # Verify index was created
        self.assertIn('test_vector2', self.catalog.Indexes.objectIds())
        index = self.catalog.Indexes['test_vector2']

        # Verify indexed_attrs were parsed correctly
        self.assertEqual(
            index.indexed_attrs,
            ['title', 'description', 'text']
        )

    def test_validation_empty_id(self):
        """Test that empty ID shows error and doesn't create index."""
        initial_indexes = list(self.catalog.Indexes.objectIds())

        self.request.form = {
            'id': '',
            'submit_add': 'Add'
        }
        self.request.method = 'POST'

        view = self.catalog.Indexes.restrictedTraverse('@@add-vector-index.html')
        result = view()

        # Should return form (not redirect)
        self.assertIn('Add Vector Index', result)

        # No new index should be created
        self.assertEqual(
            initial_indexes,
            list(self.catalog.Indexes.objectIds())
        )

    def test_index_uses_registry_settings(self):
        """Test that newly created index uses registry settings."""
        # Create index
        self.request.form = {
            'id': 'test_vector3',
            'submit_add': 'Add'
        }
        self.request.method = 'POST'

        view = self.catalog.Indexes.restrictedTraverse('@@add-vector-index.html')
        view()

        # Get the index
        index = self.catalog.Indexes['test_vector3']

        # Verify it uses registry settings
        # The default chunk size should be 500
        self.assertEqual(index.embedding.chank_size, 500)

        # The similarity algorithm should be CosineSimilarityAlgorithm
        from collective.vectorsearch.similarity_algorithm import (
            CosineSimilarityAlgorithm
        )
        self.assertIsInstance(
            index.similarity_algorithm,
            CosineSimilarityAlgorithm
        )


class TestAddVectorIndexViewPermissions(unittest.TestCase):
    """Test that AddVectorIndexView has proper permissions."""

    layer = COLLECTIVE_VECTORSEARCH_INTEGRATION_TESTING

    def setUp(self):
        """Custom shared utility setup for tests."""
        self.portal = self.layer["portal"]
        self.request = self.layer["request"]
        self.catalog = api.portal.get_tool('portal_catalog')

    def test_view_protected(self):
        """Test that regular users cannot access the view."""
        from AccessControl import Unauthorized

        # Set user as regular member
        setRoles(self.portal, TEST_USER_ID, ['Member'])

        # Attempt to access view should raise Unauthorized
        with self.assertRaises(Unauthorized):
            self.catalog.Indexes.restrictedTraverse('@@add-vector-index.html')
