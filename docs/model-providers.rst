====================================
Creating Custom Embedding Providers
====================================

This document explains how to create custom embedding model providers for
collective.vectorsearch. This is useful when you want to use a different
embedding model than the built-in options.

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
========

collective.vectorsearch uses a pluggable provider architecture for embedding models.
Each provider is a class that:

1. Describes the model (name, dimensions, backend requirements)
2. Creates embedding instances for generating vectors
3. Optionally provides ITQ/pivot data for approximate search

Providers are registered as Zope utilities and automatically appear in the
control panel's model selection dropdown.


Built-in Providers
==================

The package includes three providers:

+-------------------------------+------------+---------------------+-------------------+
| Provider                      | Dimensions | Backend             | ITQ/Pivot Support |
+===============================+============+=====================+===================+
| AllMiniLMProvider             | 384        | FastEmbed           | Yes               |
+-------------------------------+------------+---------------------+-------------------+
| E5BaseMultilingualProvider    | 768        | FastEmbed           | Yes               |
+-------------------------------+------------+---------------------+-------------------+
| E5BaseMultilingualGPUProvider | 768        | SentenceTransformers| Yes               |
+-------------------------------+------------+---------------------+-------------------+


Creating a Basic Provider
=========================

To create a custom provider, extend ``BaseEmbeddingModelProvider``:

.. code-block:: python

    # mypackage/providers.py
    from collective.vectorsearch.model_providers import BaseEmbeddingModelProvider

    class MyCustomProvider(BaseEmbeddingModelProvider):
        """Provider for my custom embedding model."""

        # Required: Unique identifier (used in registry)
        id = "my-custom-model"

        # Required: Display name for control panel
        title = "My Custom Model"

        # Required: Model description
        description = "Custom model - 512 dimensions, specialized for my domain"

        # Required: Model name/path (Hugging Face model ID)
        model_name = "my-org/my-custom-model"

        # Required: Output vector dimensionality
        vector_dimensions = 512

        # Backend configuration
        backend = "fastembed"  # or "sentence_transformers"
        backend_name = "FastEmbed (CPU/ONNX)"
        requires_gpu = False
        extras_name = None  # or "gpu" for [gpu] extras

        # Embedding class to use
        embedding_class = "FastEmbedEmbedding"  # or "SentenceTransformerEmbedding"


Provider Attributes Reference
=============================

Required Attributes
-------------------

**id** (str)
    Unique identifier for the provider. Used in registry configuration.
    Convention: lowercase with hyphens (e.g., ``"my-model-v2"``).

**title** (str)
    Human-readable name shown in the control panel.

**description** (str)
    Description of the model's capabilities, language support, etc.

**model_name** (str)
    The model identifier. For Hugging Face models, this is the repo ID
    (e.g., ``"sentence-transformers/all-MiniLM-L6-v2"``).

**vector_dimensions** (int)
    The dimensionality of output vectors. Must match the model's actual output.

Backend Attributes
------------------

**backend** (str)
    Backend identifier: ``"fastembed"`` or ``"sentence_transformers"``.

**backend_name** (str)
    Human-readable backend name for display.

**requires_gpu** (bool)
    Whether this model requires or benefits from GPU acceleration.

**extras_name** (str or None)
    Buildout extras required. ``None`` for default installation,
    ``"gpu"`` for ``[gpu]`` extras.

**embedding_class** (str)
    The embedding class to instantiate:

    - ``"FastEmbedEmbedding"``: Uses fastembed library (ONNX, CPU)
    - ``"SentenceTransformerEmbedding"``: Uses sentence-transformers (PyTorch)

Optional Attributes
-------------------

**query_prefix** (str or None)
    Prefix to add to query text. Some models require this
    (e.g., E5 models use ``"query: "``).

**passage_prefix** (str or None)
    Prefix to add to document/passage text
    (e.g., E5 models use ``"passage: "``).

**hash_length** (int)
    Binary hash length for ITQ. Default: ``128``.

**use_cache_dir** (bool)
    Whether the model uses a cache directory. Default: ``False``.

**data_file_id** (str or None)
    Override the data file identifier for ITQ/pivot data.
    If ``None``, uses ``id`` with hyphens converted to underscores.
    Useful when multiple providers share the same ITQ/pivot data
    (e.g., CPU and GPU variants of the same model).


Registering the Provider
========================

Register your provider as a Zope utility in your package's ``configure.zcml``:

.. code-block:: xml

    <configure xmlns="http://namespaces.zope.org/zope">

      <utility
          factory=".providers.MyCustomProvider"
          provides="collective.vectorsearch.interfaces.IEmbeddingModelProvider"
          name="my-custom-model"
      />

    </configure>

The ``name`` attribute must match the provider's ``id``.


Adding Prefix Support (E5-style Models)
=======================================

Some models like E5 require different prefixes for queries and documents:

.. code-block:: python

    class E5LargeProvider(BaseEmbeddingModelProvider):
        id = "e5-large"
        title = "E5 Large"
        model_name = "intfloat/e5-large"
        vector_dimensions = 1024

        # Prefix configuration
        query_prefix = "query: "
        passage_prefix = "passage: "

        backend = "fastembed"
        embedding_class = "FastEmbedEmbedding"

The embedding classes automatically apply these prefixes when generating vectors.


Adding ITQ/Pivot Support
========================

To enable approximate search for your custom model, you need to:

1. Generate ITQ and pivot data (see `approximate-search.rst <approximate-search.rst>`_)
2. Place the data files in the correct location
3. Set the ``data_file_id`` attribute if needed

File Placement
--------------

Place your data files in the package's data directory:

.. code-block:: text

    src/collective/vectorsearch/data/
    ├── itq/
    │   └── my_custom_model/
    │       ├── mean_vector.npy      # (512,)
    │       ├── pca_matrix.npy       # (512, 128)
    │       ├── rotation_matrix.npy  # (128, 128)
    │       └── metadata.npy         # optional
    └── pivot/
        └── my_custom_model.npy      # (8, 512)

Note: The directory/file name uses underscores (``my_custom_model``),
converted from the provider's ``id`` (``my-custom-model``).

Sharing Data Between Providers
------------------------------

If you have CPU and GPU variants of the same model, they can share ITQ/pivot data:

.. code-block:: python

    class MyModelCPUProvider(BaseEmbeddingModelProvider):
        id = "my-model-cpu"
        model_name = "my-org/my-model"
        vector_dimensions = 512
        backend = "fastembed"
        # Uses default data_file_id: "my_model_cpu"


    class MyModelGPUProvider(BaseEmbeddingModelProvider):
        id = "my-model-gpu"
        model_name = "my-org/my-model"
        vector_dimensions = 512
        backend = "sentence_transformers"

        # Share data with CPU variant
        data_file_id = "my_model_cpu"


Using the Provider
==================

Once registered, your provider will automatically appear in the control panel.
You can also use it programmatically:

.. code-block:: python

    from zope.component import getUtility
    from collective.vectorsearch.interfaces import IEmbeddingModelProvider

    # Get your provider
    provider = getUtility(IEmbeddingModelProvider, name="my-custom-model")

    # Check availability
    if provider.is_available():
        # Create embedding instance
        embedding = provider.get_embedding_instance(chunk_size=500)

        # Generate vectors
        vectors = list(embedding.embed_documents(["Hello world"]))

        # Get ITQ data (if available)
        itq = provider.get_itq_boundary()
        if itq:
            hash_code = itq.compute_hash(vectors[0])


Complete Example
================

Here's a complete example of a custom provider with all features:

.. code-block:: python

    # mypackage/providers.py
    from collective.vectorsearch.model_providers import BaseEmbeddingModelProvider


    class BGESmallProvider(BaseEmbeddingModelProvider):
        """Provider for BAAI BGE Small English model.

        A lightweight English embedding model optimized for retrieval tasks.
        """

        # Identity
        id = "bge-small-en"
        title = "BGE Small English (FastEmbed)"
        description = (
            "BAAI BGE Small - 384 dimensions, ONNX optimized, "
            "retrieval-focused English model"
        )

        # Model configuration
        model_name = "BAAI/bge-small-en-v1.5"
        vector_dimensions = 384
        hash_length = 128

        # Prefix configuration (BGE models need this)
        query_prefix = "Represent this sentence for searching relevant passages: "
        passage_prefix = None  # No prefix for documents

        # Backend configuration
        backend = "fastembed"
        backend_name = "FastEmbed (CPU/ONNX)"
        requires_gpu = False
        extras_name = None
        embedding_class = "FastEmbedEmbedding"
        use_cache_dir = True


And the registration:

.. code-block:: xml

    <!-- mypackage/configure.zcml -->
    <configure xmlns="http://namespaces.zope.org/zope">

      <include package="collective.vectorsearch" />

      <utility
          factory=".providers.BGESmallProvider"
          provides="collective.vectorsearch.interfaces.IEmbeddingModelProvider"
          name="bge-small-en"
      />

    </configure>


Troubleshooting
===============

Provider Not Appearing in Control Panel
---------------------------------------

1. Ensure the ZCML is loaded (check for ``<include package="..."/>``)
2. Restart the Zope server after changes
3. Check that ``is_available()`` returns ``True``

.. code-block:: python

    from mypackage.providers import BGESmallProvider
    provider = BGESmallProvider()
    print(f"Available: {provider.is_available()}")
    print(f"Reason: {provider.get_unavailable_reason()}")

ITQ/Pivot Data Not Loading
--------------------------

1. Verify file paths match ``data_file_id`` (underscores, not hyphens)
2. Check file shapes match ``vector_dimensions`` and ``hash_length``
3. Enable debug logging to see detailed error messages:

.. code-block:: python

    import logging
    logging.getLogger("collective.vectorsearch").setLevel(logging.DEBUG)

Model Download Fails
--------------------

For FastEmbed models, pre-download using the offline command:

.. code-block:: bash

    # Add your model to the download list or download manually
    python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"


See Also
========

- `Approximate Search <approximate-search.rst>`_ - ITQ/Pivot data generation and usage
- `Hugging Face Model Hub <https://huggingface.co/models>`_ - Find embedding models
- `FastEmbed Supported Models <https://github.com/qdrant/fastembed>`_ - ONNX-optimized models
