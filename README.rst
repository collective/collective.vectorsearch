.. This README is meant for consumption by humans and PyPI. PyPI can render rst files so please do not use Sphinx features.
   If you want to learn more about writing documentation, please check out: http://docs.plone.org/about/documentation_styleguide.html
   This text does not appear on PyPI or github. It is a comment.

**Language**: English | `日本語 <README-ja.rst>`_

.. image:: https://github.com/collective/collective.vectorsearch/actions/workflows/plone-package.yml/badge.svg
    :target: https://github.com/collective/collective.vectorsearch/actions/workflows/plone-package.yml

.. image:: https://coveralls.io/repos/github/collective/collective.vectorsearch/badge.svg?branch=main
    :target: https://coveralls.io/github/collective/collective.vectorsearch?branch=main
    :alt: Coveralls

.. image:: https://img.shields.io/pypi/v/collective.vectorsearch.svg
    :target: https://pypi.python.org/pypi/collective.vectorsearch/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/status/collective.vectorsearch.svg
    :target: https://pypi.python.org/pypi/collective.vectorsearch
    :alt: Egg Status

.. image:: https://img.shields.io/pypi/pyversions/collective.vectorsearch.svg?style=plastic
    :alt: Supported - Python Versions

.. image:: https://img.shields.io/pypi/l/collective.vectorsearch.svg
    :target: https://pypi.python.org/pypi/collective.vectorsearch/
    :alt: License


=======================
collective.vectorsearch
=======================

A Plone add-on that provides semantic vector search capabilities using LLM embeddings.
This package enables similarity-based content search by converting text into vector embeddings
and finding semantically similar content using cosine similarity.


Features
--------

- **VectorIndex for ZCatalog**: A custom catalog index that stores vector embeddings
- **Multiple Embedding Models**: Support for various embedding models:

  - MiniLM L6 v2 (default, lightweight, English, FastEmbed)
  - E5-base multilingual (100+ languages, FastEmbed)
  - E5-base multilingual GPU (GPU accelerated, requires ``[gpu]`` extras)

- **FastEmbed by Default**: CPU-friendly ONNX-optimized embeddings, no GPU required
- **Optional GPU Support**: Add ``[gpu]`` extras for GPU-accelerated processing
- **Control Panel**: Configure embedding models and search settings via Site Setup
- **Backend Status Display**: View available backends and installation status
- **Lazy Model Loading**: Models are loaded on first use, not during package installation
- **Pluggable Architecture**: Easy to add new embedding model providers


Requirements
------------

- Plone 6.0 or 6.1
- Python 3.9 - 3.13


Installation
------------

Install collective.vectorsearch by adding it to your buildout::

    [buildout]

    ...

    eggs =
        collective.vectorsearch


and then running ``bin/buildout``.

GPU Support (Optional)
~~~~~~~~~~~~~~~~~~~~~~

For GPU-accelerated embedding with PyTorch and Sentence Transformers::

    [buildout]

    ...

    eggs =
        collective.vectorsearch [gpu]


Quick Start
-----------

1. Install the package via Site Setup → Add-ons
2. Go to Site Setup → Vector Search to configure the embedding model
3. Check the "Embedding Backend Status" section to verify backends are available
4. The ``llm_vector`` index is automatically added to ``portal_catalog``
5. Reindex your content via the control panel or ZMI


Configuration
-------------

Access the control panel at Site Setup → Vector Search to configure:

- **Embedding Model**: Select the model for generating embeddings (only available models are shown)
- **Text Chunk Size**: Maximum characters per chunk for long documents
- **Storage Backend**: Currently supports BTrees (internal storage)
- **Approximation Algorithm**: Search algorithm (currently exhaustive cosine similarity)

The control panel displays:

- **Embedding Backend Status**: Shows which backends are installed (FastEmbed, Sentence Transformers)
- **Vector Index Statistics**: Document and vector counts for each index


Available Embedding Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+------------+------+------------------------+
| Model                     | Dimensions | GPU  | Extras                 |
+===========================+============+======+========================+
| MiniLM L6 v2 (FastEmbed)  | 384        | No   | (default)              |
+---------------------------+------------+------+------------------------+
| E5 Base Multilingual      | 768        | No   | (default)              |
| (FastEmbed)               |            |      |                        |
+---------------------------+------------+------+------------------------+
| E5 Base Multilingual      | 768        | Yes  | ``[gpu]``              |
| (GPU)                     |            |      |                        |
+---------------------------+------------+------+------------------------+


Usage
-----

The package adds a ``VectorIndex`` named ``llm_vector`` to the portal catalog.
You can query it programmatically::

    from plone import api

    catalog = api.portal.get_tool('portal_catalog')
    index = catalog.Indexes['llm_vector']

    # Query returns document IDs with similarity scores
    results = index.query_index(record)


Adding Custom Vector Indexes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can add additional VectorIndex instances via ZMI:

1. Navigate to ``/Plone/portal_catalog/manage_main``
2. Select "VectorIndex" from the index type dropdown
3. Enter an ID and optionally specify indexed attributes


Extending with Custom Model Providers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

External packages can add new embedding models by implementing ``IEmbeddingModelProvider``::

    from collective.vectorsearch.model_providers import BaseEmbeddingModelProvider

    class MyCustomProvider(BaseEmbeddingModelProvider):
        id = 'my-custom-model'
        title = u'My Custom Model'
        description = u'Custom model description'
        model_name = 'my-org/my-model'
        vector_dimensions = 768

        # Backend configuration
        backend = 'fastembed'  # or 'sentence_transformers'
        backend_name = u'FastEmbed (CPU/ONNX)'
        requires_gpu = False
        extras_name = None  # or 'gpu' for [gpu] extras

Register it in your package's ``configure.zcml``::

    <utility
        factory=".providers.MyCustomProvider"
        provides="collective.vectorsearch.interfaces.IEmbeddingModelProvider"
        name="my-custom-model"
    />


Important Notes
---------------

Reinstall and Upgrade
~~~~~~~~~~~~~~~~~~~~~

After reinstalling or upgrading this package, you **must restart** the Plone/Zope server.
Without a restart, the model provider utilities may not be properly registered,
which can cause reindexing to fail silently.

**Recommended procedure:**

1. Reinstall or upgrade the package via Site Setup → Add-ons
2. Restart the Plone/Zope server
3. Go to Site Setup → Vector Search
4. Click "Reindex All" to rebuild the vector index

Uninstall Behavior
~~~~~~~~~~~~~~~~~~

**Warning:** Uninstalling this package will delete all vector data from the catalog.
The ``llm_vector`` index and all its embeddings will be permanently removed.

If you need to preserve vector data while updating the package code, use the
**Upgrade** feature instead of uninstall/reinstall.


Development
-----------

To set up a development environment::

    git clone https://github.com/collective/collective.vectorsearch.git
    cd collective.vectorsearch
    make install

Run tests::

    make test

See ``DEVELOP.rst`` for detailed development instructions.


Author
------

- Manabu TERADA (`@terapyon <https://github.com/terapyon>`_)


Contributors
------------

- (Your name here)


Contribute
----------

- Issue Tracker: https://github.com/collective/collective.vectorsearch/issues
- Source Code: https://github.com/collective/collective.vectorsearch


Support
-------

If you are having issues, please open an issue on GitHub:
https://github.com/collective/collective.vectorsearch/issues


License
-------

The project is licensed under the GPLv2.
