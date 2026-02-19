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

A Plone add-on that brings semantic vector search to your content management system.
It converts text into vector embeddings using LLM-based models and finds semantically similar
content through multi-stage approximate nearest neighbor search.


Features
--------

- **VectorIndex for ZCatalog**: A custom catalog index that stores and searches vector embeddings alongside traditional Plone indexes
- **Multi-Stage Approximate Search**: Three search algorithms with automatic fallback:

  - **Exhaustive Cosine** (default): Brute-force cosine similarity on all documents
  - **ITQ-LSH 2-Stage**: Hamming distance ranking via ITQ binary hashes, then cosine similarity on top-K candidates
  - **ITQ-LSH 3-Stage**: Pivot-based triangle inequality filtering, then Hamming ranking, then cosine similarity

- **Multiple Embedding Models**:

  - All-MiniLM-L6-v2 (default, 384 dimensions, English, FastEmbed/CPU)
  - E5-Base Multilingual (768 dimensions, 100+ languages, FastEmbed/CPU)
  - E5-Base Multilingual GPU (768 dimensions, GPU-accelerated, requires ``[gpu]`` extras)

- **Annotation-Based Data Storage**: Vector data stored in content annotations as the single source of truth
- **FastEmbed by Default**: CPU-friendly ONNX-optimized embeddings, no GPU required
- **Optional GPU Support**: Install ``[gpu]`` extras for GPU-accelerated processing via PyTorch and Sentence Transformers
- **Control Panel**: Configure models, search algorithms, and parameters via Site Setup
- **Pluggable Architecture**: Add new embedding model providers from external packages


Requirements
------------

- Plone 6.0 or 6.1+
- Python 3.10 - 3.13


Installation
------------

Install collective.vectorsearch by adding it to your buildout::

    [buildout]

    ...

    eggs =
        collective.vectorsearch


and then running ``bin/buildout``.

Or install via pip::

    pip install collective.vectorsearch

GPU Support (Optional)
~~~~~~~~~~~~~~~~~~~~~~

For GPU-accelerated embedding with PyTorch and Sentence Transformers::

    pip install collective.vectorsearch[gpu]

Or in buildout::

    [buildout]

    ...

    eggs =
        collective.vectorsearch [gpu]


Quick Start
-----------

1. Install the package via Site Setup -> Add-ons
2. Go to Site Setup -> Vector Search to configure the embedding model
3. The ``llm_vector`` index is automatically added to ``portal_catalog``
4. Content is automatically vectorized when created or modified
5. Use the "Reindex All" button in the control panel to vectorize existing content


How It Works
------------

Architecture
~~~~~~~~~~~~

When content is created or modified, event subscribers automatically compute embeddings
and store them in content annotations. The catalog indexers then read from these annotations
to populate the VectorIndex and supporting indexes (pivot1-8, itq_hashes).

::

    Content created/modified
      |
      +-- Event subscriber: compute_and_store_vectors()
      |     +-- Embed text using configured model
      |     +-- Compute ITQ binary hashes (128-bit)
      |     +-- Compute pivot distances (8 pivots)
      |     +-- Store all data in content annotations
      |
      +-- Catalog indexing
            +-- VectorIndex: reads vectors from annotations
            +-- pivot1-8 KeywordIndex: reads pivot distances
            +-- itq_hashes metadata: reads ITQ hashes

Multi-Stage Search
~~~~~~~~~~~~~~~~~~

The package implements a multi-stage approximate nearest neighbor search based on
the `lsh-cascade-poc <https://github.com/cmscom/lsh-cascade-poc>`_ research:

**Exhaustive Cosine** (``exhaustive_cosine``):
  Computes cosine similarity against all indexed documents.
  Most accurate but slowest for large datasets.

**ITQ-LSH 2-Stage** (``itq_lsh_2stage``):
  1. Compute query ITQ hash and rank all documents by Hamming distance
  2. Compute cosine similarity on the top-K candidates (``itq_candidates``, default: 100)

**ITQ-LSH 3-Stage** (``itq_lsh_3stage``):
  1. **Pivot filtering**: Use 8 pivot distances with triangle inequality to narrow candidates via KeywordIndex range queries
  2. **Hamming ranking**: Rank remaining candidates by ITQ Hamming distance, keep top-K
  3. **Cosine similarity**: Precise scoring on final candidates

The system automatically falls back: 3-stage -> 2-stage -> exhaustive if the required
ITQ or pivot data is unavailable.


Configuration
-------------

Access the control panel at Site Setup -> Vector Search to configure:

- **Embedding Model**: Select the model for generating embeddings
- **Text Chunk Size**: Maximum characters per chunk (100-10,000, default: 500)
- **Approximation Algorithm**: Search strategy (exhaustive_cosine, itq_lsh_2stage, itq_lsh_3stage)
- **Pivot Threshold (Stage 1)**: Filtering threshold for pivot-based search (cosine distance x 1000, default: 200)
- **ITQ Candidates (Stage 2)**: Number of candidates after Hamming ranking (default: 100)
- **Storage Backend**: Currently supports BTrees (internal storage)


Available Embedding Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+------------+------+------------------------+
| Model                     | Dimensions | GPU  | Extras                 |
+===========================+============+======+========================+
| All-MiniLM-L6-v2          | 384        | No   | (default)              |
| (FastEmbed)               |            |      |                        |
+---------------------------+------------+------+------------------------+
| E5 Base Multilingual      | 768        | No   | (default)              |
| (FastEmbed)               |            |      |                        |
+---------------------------+------------+------+------------------------+
| E5 Base Multilingual      | 768        | Yes  | ``[gpu]``              |
| (GPU)                     |            |      |                        |
+---------------------------+------------+------+------------------------+


Usage
-----

Programmatic Search
~~~~~~~~~~~~~~~~~~~

The package adds a ``VectorIndex`` named ``llm_vector`` to the portal catalog.
You can query it programmatically::

    from plone import api

    catalog = api.portal.get_tool('portal_catalog')
    index = catalog.Indexes['llm_vector']

    # Search for similar content
    results = index.query_index(record)


Adding Custom Vector Indexes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can add additional VectorIndex instances via ZMI:

1. Navigate to ``/Plone/portal_catalog/manage_main``
2. Select "VectorIndex" from the index type dropdown
3. Enter an ID and optionally specify indexed attributes (comma-separated)


Extending with Custom Model Providers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Offline Model Download
~~~~~~~~~~~~~~~~~~~~~~

FastEmbed downloads models on first use. For offline environments, pre-download
models using the CLI command::

    vectorsearch-download

This downloads all supported models to ``~/.cache/fastembed``.
Set ``FASTEMBED_CACHE_PATH`` environment variable to use a different location.


Important Notes
---------------

Reinstall and Upgrade
~~~~~~~~~~~~~~~~~~~~~

After reinstalling or upgrading this package, you **must restart** the Plone/Zope server.
Without a restart, the model provider utilities may not be properly registered.

**Recommended procedure:**

1. Reinstall or upgrade the package via Site Setup -> Add-ons
2. Restart the Plone/Zope server
3. Go to Site Setup -> Vector Search
4. Click "Reindex All" to rebuild the vector index

Uninstall Behavior
~~~~~~~~~~~~~~~~~~

**Warning:** Uninstalling this package will delete all vector data from the catalog.
The ``llm_vector`` index, pivot indexes, and all embeddings will be permanently removed.

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
