====================================
Approximate Nearest Neighbor Search
====================================

This document explains the approximate nearest neighbor (ANN) search capabilities
of collective.vectorsearch, including ITQ-LSH (Iterative Quantization - Locality
Sensitive Hashing) and Pivot-based filtering.

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
========

Vector search typically requires computing similarity (e.g., cosine similarity)
between a query vector and all document vectors in the index. For large datasets,
this exhaustive search becomes slow.

collective.vectorsearch supports approximate search algorithms that trade a small
amount of accuracy for significant speed improvements:

- **ITQ-LSH**: Converts vectors to compact binary hashes for fast Hamming distance comparison
- **Pivot-based filtering**: Uses triangle inequality to prune candidates before exact comparison

These techniques are based on research from the
`lsh-cascade-poc <https://github.com/cmscom/lsh-cascade-poc>`_ project.


ITQ (Iterative Quantization)
============================

ITQ transforms high-dimensional floating-point vectors into compact binary hashes.
This enables:

- **Reduced storage**: 768-dimensional vectors (3KB) → 128-bit hashes (16 bytes)
- **Fast comparison**: Hamming distance on binary data is extremely fast
- **Hardware-friendly**: Binary operations are optimized on modern CPUs

The transformation pipeline:

.. code-block:: text

    Input Vector (768 dims)
         ↓
    - mean_vector (centering)
         ↓
    @ pca_matrix → PCA projection (128 dims)
         ↓
    @ rotation_matrix → ITQ rotation (128 dims)
         ↓
    sign() → Binary hash (128 bits)

How ITQ Learning Works
----------------------

ITQ requires three learned components: mean vector, PCA matrix, and rotation matrix.
Each serves a specific purpose in the transformation.

**1. Mean Vector (Centering)**

The mean vector is simply the average of all training embeddings:

.. code-block:: text

    mean_vector = mean(training_embeddings, axis=0)

Centering is essential because:

- It shifts the data distribution to be centered at the origin
- Sign quantization (positive → 1, negative → 0) works best when data is balanced
  around zero
- Without centering, the hash bits would be biased toward one value

**2. PCA Matrix (Dimensionality Reduction)**

PCA (Principal Component Analysis) reduces 768 dimensions to 128 while preserving
maximum variance:

.. code-block:: text

    pca_matrix = PCA(n_components=128).fit(centered_data).components_.T

Why PCA?

- Reduces computation from O(768) to O(128) per comparison
- Keeps the most "informative" dimensions (highest variance)
- Decorrelates dimensions, making subsequent quantization more effective
- 128 bits provides a good balance between accuracy and storage

**3. Rotation Matrix (ITQ Optimization)**

The rotation matrix is the key innovation of ITQ. Simple sign quantization after
PCA loses information because:

- Some dimensions have values clustered near zero (uncertain bits)
- Bits may be correlated, wasting capacity

ITQ finds an optimal rotation that minimizes quantization error through iteration:

.. code-block:: text

    For each iteration:
        1. Compute binary codes: B = sign(V @ R)
        2. Find optimal rotation: R = argmin ||V @ R - B||
           (solved via SVD: R = V' @ U where SVD(B' @ V) = U @ S @ V')

The rotation spreads information evenly across all bits, maximizing the
discriminative power of each bit.

**Training Data Requirements**

- **Quantity**: 10,000+ embeddings recommended for stable statistics
- **Diversity**: Should represent the variety of documents in your corpus
- **Domain**: Training on in-domain data improves accuracy; general-purpose
  data works but may be suboptimal


Pivot-Based Filtering
=====================

Pivot filtering uses a small set of reference vectors (pivots) to quickly
eliminate candidates that cannot be similar to the query.

Given 8 pivot vectors, we precompute the distance from each document to all pivots.
At query time:

1. Compute query-to-pivot distances
2. For each document, check if ``max(|doc_pivot_dist - query_pivot_dist|) < threshold``
3. Only documents passing this filter proceed to exact similarity calculation

This leverages the triangle inequality: if two vectors are similar, their distances
to a shared pivot point must also be similar.

How Pivots are Selected
-----------------------

Pivots are chosen using k-means clustering on the training embeddings:

.. code-block:: text

    pivots = KMeans(n_clusters=8).fit(training_embeddings).cluster_centers_

Why k-means clustering?

- **Representative**: Cluster centers represent different "regions" of the
  embedding space
- **Well-spaced**: k-means naturally spreads pivots to cover the space efficiently
- **Stable**: Centers are robust to small changes in training data

Why 8 pivots?

- More pivots = better filtering but more storage per document (8 floats per doc)
- Fewer pivots = less filtering power
- 8 provides good balance: ~90% recall with 4x candidate reduction

**Triangle Inequality Explained**

For any three points A, B, P (pivot):

.. code-block:: text

    |dist(A, P) - dist(B, P)| ≤ dist(A, B)

If A (query) and B (document) are similar (small dist(A,B)), then their distances
to any pivot P must also be similar. If we find a pivot where distances differ
significantly, we can safely skip that document.

**Threshold Selection**

The threshold parameter controls the recall-speed tradeoff:

- **threshold = 0.20**: ~90% recall, filters out ~75% of candidates
- **threshold = 0.15**: ~86% recall, filters out ~85% of candidates
- **threshold = 0.25**: ~94% recall, filters out ~60% of candidates

Lower thresholds are faster but may miss some relevant results.


Included Data Files
===================

collective.vectorsearch includes pre-computed ITQ and pivot data for the
default embedding models:

**MiniLM L6 v2** (384 dimensions):

- ITQ data: ``data/itq/all_minilm_l6/``
- Pivot data: ``data/pivot/all_minilm_l6.npy``

**E5 Base Multilingual** (768 dimensions):

- ITQ data: ``data/itq/e5_base_multilingual/``
- Pivot data: ``data/pivot/e5_base_multilingual.npy``

These data files are automatically loaded when using the corresponding
embedding model providers.


Data File Structure
===================

ITQ Data Directory
------------------

Each model's ITQ data is stored in a subdirectory with the following files:

.. code-block:: text

    data/itq/{model_id}/
    ├── mean_vector.npy      # Shape: (vector_dims,)
    ├── pca_matrix.npy       # Shape: (vector_dims, 128)
    ├── rotation_matrix.npy  # Shape: (128, 128)
    └── metadata.npy         # Optional: dict with training info

**mean_vector.npy**: The mean of the training vectors, used for centering.

**pca_matrix.npy**: PCA projection matrix that reduces dimensionality while
preserving variance.

**rotation_matrix.npy**: ITQ rotation matrix optimized to minimize quantization
error.

**metadata.npy**: Optional dictionary containing training parameters and statistics.

Pivot Data File
---------------

Pivot data is stored as a single NumPy array:

.. code-block:: text

    data/pivot/{model_id}.npy  # Shape: (8, vector_dims)

Each row is a pivot vector (cluster centroid) computed from training data.


Placing Custom Data Files
=========================

If you have custom ITQ/pivot data files (e.g., from your own training), place them
in the package's data directory:

For ITQ data (rename your files accordingly):

.. code-block:: bash

    # Example for E5 model
    cp e5_base_itq_mean_vector.npy \
       src/collective/vectorsearch/data/itq/e5_base_multilingual/mean_vector.npy

    cp e5_base_itq_pca_matrix.npy \
       src/collective/vectorsearch/data/itq/e5_base_multilingual/pca_matrix.npy

    cp e5_base_itq_rotation_matrix.npy \
       src/collective/vectorsearch/data/itq/e5_base_multilingual/rotation_matrix.npy

    cp e5_base_itq_metadata.npy \
       src/collective/vectorsearch/data/itq/e5_base_multilingual/metadata.npy

For pivot data:

.. code-block:: bash

    cp e5_base_pivots_8.npy \
       src/collective/vectorsearch/data/pivot/e5_base_multilingual.npy

After placing the files, verify they load correctly:

.. code-block:: python

    from collective.vectorsearch.model_providers import E5BaseMultilingualProvider

    provider = E5BaseMultilingualProvider()
    itq = provider.get_itq_boundary()
    pivot = provider.get_pivot_data()

    print(f"ITQ loaded: {itq is not None}")
    print(f"Pivot loaded: {pivot is not None}")

    if itq:
        print(f"  mean_vector: {itq.mean_vector.shape}")
        print(f"  pca_matrix: {itq.pca_matrix.shape}")
        print(f"  rotation_matrix: {itq.rotation_matrix.shape}")

    if pivot:
        print(f"  pivots: {pivot.pivots.shape}")


Generating Your Own ITQ/Pivot Data
==================================

To generate ITQ and pivot data for a custom model or dataset, follow the
methodology from the `lsh-cascade-poc <https://github.com/cmscom/lsh-cascade-poc>`_
repository.

Requirements
------------

- Representative sample of document embeddings (10,000+ recommended)
- NumPy, scikit-learn
- The embedding model you want to support

Step 1: Generate Embeddings
---------------------------

First, generate embeddings for your representative document set:

.. code-block:: python

    import numpy as np
    from fastembed import TextEmbedding

    # Load your documents
    documents = [...]  # List of text strings

    # Generate embeddings
    model = TextEmbedding("your-model-name")
    embeddings = list(model.embed(documents))
    embeddings = np.array(embeddings)

    # Save for later use
    np.save("training_embeddings.npy", embeddings)

Step 2: Train ITQ
-----------------

Train the ITQ transformation. This involves three sub-steps: centering,
PCA projection, and rotation optimization.

.. code-block:: python

    import numpy as np
    from sklearn.decomposition import PCA

    # Load embeddings
    embeddings = np.load("training_embeddings.npy")
    print(f"Training data shape: {embeddings.shape}")  # e.g., (10000, 768)

    # Parameters
    hash_length = 128  # Number of bits in the hash
    n_iterations = 50  # ITQ optimization iterations

    # === Step 2a: Compute mean vector for centering ===
    # This shifts the data to be centered at the origin,
    # which is important for balanced sign quantization.
    mean_vector = embeddings.mean(axis=0)
    centered = embeddings - mean_vector

    # === Step 2b: PCA projection ===
    # Reduces dimensionality while preserving maximum variance.
    # components_ gives eigenvectors as rows, we need columns.
    pca = PCA(n_components=hash_length)
    projected = pca.fit_transform(centered)
    pca_matrix = pca.components_.T  # Shape: (original_dims, hash_length)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # === Step 2c: ITQ rotation optimization ===
    # Finds the optimal rotation that minimizes quantization error.
    # Uses alternating optimization: fix codes, optimize rotation; repeat.
    rotation = np.eye(hash_length)  # Start with identity rotation

    for i in range(n_iterations):
        # Given current rotation, compute best binary codes
        rotated = projected @ rotation
        binary = (rotated > 0).astype(float) * 2 - 1  # Convert to {-1, +1}

        # Given binary codes, find optimal rotation via SVD
        # This minimizes ||projected @ rotation - binary||^2
        U, S, Vt = np.linalg.svd(binary.T @ projected)
        rotation = (Vt.T @ U.T).T

        # Optional: track convergence
        if i % 10 == 0:
            error = np.mean((projected @ rotation - binary) ** 2)
            print(f"Iteration {i}: quantization error = {error:.4f}")

    rotation_matrix = rotation

    # Save ITQ data files
    np.save("mean_vector.npy", mean_vector.astype(np.float32))
    np.save("pca_matrix.npy", pca_matrix.astype(np.float32))
    np.save("rotation_matrix.npy", rotation_matrix.astype(np.float32))
    print("ITQ training complete!")

Step 3: Compute Pivots
----------------------

Compute pivot vectors using k-means clustering. The cluster centers serve
as reference points for triangle inequality filtering.

.. code-block:: python

    import numpy as np
    from sklearn.cluster import KMeans

    # Load embeddings (use the same training data as ITQ)
    embeddings = np.load("training_embeddings.npy")

    # === Cluster to find representative pivot points ===
    # 8 pivots provides good balance between filtering power and storage
    n_pivots = 8
    kmeans = KMeans(
        n_clusters=n_pivots,
        random_state=42,  # For reproducibility
        n_init=10,        # Run 10 times with different seeds
        max_iter=300
    )
    kmeans.fit(embeddings)

    pivots = kmeans.cluster_centers_  # Shape: (8, dims)
    print(f"Pivot shapes: {pivots.shape}")
    print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")

    # Save pivot data
    np.save("pivots.npy", pivots.astype(np.float32))
    print("Pivot computation complete!")

For more detailed examples, see the notebooks in the
`lsh-cascade-poc repository <https://github.com/cmscom/lsh-cascade-poc/tree/main/notebooks>`_.


Downloading FastEmbed Models for Offline Use
============================================

FastEmbed downloads ONNX models from Hugging Face Hub on first use. For offline
environments, you can pre-download models using the included CLI command:

.. code-block:: bash

    # Download all supported models
    vectorsearch-download

    # This downloads to ~/.cache/fastembed by default

To specify a custom cache directory:

.. code-block:: bash

    export FASTEMBED_CACHE_PATH=/path/to/cache
    vectorsearch-download

Or programmatically:

.. code-block:: python

    from collective.vectorsearch.offline import download_all_models
    from pathlib import Path

    download_all_models(cache_dir=Path("/path/to/cache"))

Model Sizes
-----------

+------------------------------------------+----------+
| Model                                    | Size     |
+==========================================+==========+
| sentence-transformers/all-MiniLM-L6-v2   | ~90 MB   |
+------------------------------------------+----------+
| intfloat/multilingual-e5-base            | ~1.1 GB  |
+------------------------------------------+----------+


Using the Data Classes Programmatically
=======================================

The ``ITQData`` and ``PivotData`` classes can be used directly for custom
search implementations:

.. code-block:: python

    from collective.vectorsearch.data_loader import load_itq_data, load_pivot_data
    import numpy as np

    # Load data for a model
    itq = load_itq_data("all_minilm_l6")
    pivot = load_pivot_data("all_minilm_l6")

    # Compute hash for a query vector
    query_vector = np.random.randn(384).astype(np.float32)
    query_hash = itq.compute_hash(query_vector)
    print(f"Query hash: {query_hash.shape}")  # (128,)

    # Compute pivot distances
    query_pivot_dist = pivot.compute_distances(query_vector)
    print(f"Pivot distances: {query_pivot_dist}")  # (8,)

    # Filter candidates
    doc_pivot_distances = np.random.rand(1000, 8)  # Pre-computed
    mask = pivot.filter_candidates(doc_pivot_distances, query_pivot_dist, threshold=0.2)
    print(f"Candidates remaining: {mask.sum()}/{len(mask)}")


References
==========

- `lsh-cascade-poc <https://github.com/cmscom/lsh-cascade-poc>`_ - Research repository
  with detailed notebooks on ITQ-LSH and pivot-based filtering
- `ITQ Paper <https://www.cs.toronto.edu/~fleet/research/Papers/gong_cvpr11.pdf>`_ -
  "Iterative Quantization: A Procrustean Approach to Learning Binary Codes"
- `Pivot-based Indexing <https://en.wikipedia.org/wiki/Metric_tree>`_ -
  Triangle inequality for metric space pruning
