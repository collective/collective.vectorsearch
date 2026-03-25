Changelog
=========


1.0a5 (unreleased)
------------------

Bug fixes:

- Pass ``cache_dir`` and ``local_files_only`` to FastEmbed ``TextEmbedding``.
  Previously, ``FastEmbedEmbedding`` did not forward these parameters, so the
  cache location was entirely dependent on the ``FASTEMBED_CACHE_PATH``
  environment variable at runtime. Model loading now tries local cache first
  (``local_files_only=True``) and falls back to network download, matching the
  existing ``ModelCache.get_model()`` behavior for Sentence Transformers.
  This also activates the previously unused ``use_cache_dir`` provider attribute.
  [terapyon]


1.0a4 (unreleased)
------------------

Features:

- **Experimental**: Add Voronoi partition search (``voronoi_2stage``) based on
  `lsh-cascade-poc <https://github.com/cmscom/lsh-cascade-poc>`_ research
  (notebooks 111, 114, 115). Voronoi partitioning showed better recall
  than ITQ-LSH at comparable candidate reduction ratios in PoC benchmarks.
  This feature is experimental and may be removed or significantly changed
  in future releases depending on real-world evaluation results.

  - Stage 1: Voronoi cell filtering via KeywordIndex (multi-probe)
  - Stage 2: Cosine similarity scoring on filtered candidates

  [terapyon]

- Add ``VoronoiData`` class for K-Means centroid loading and cell assignment.
  Pre-trained centroid data (256 clusters) included for All-MiniLM-L6-v2
  and E5-Base Multilingual models.
  [terapyon]

- Add ``voronoi_cells`` KeywordIndex with multi-assign support
  (each document chunk assigned to multiple nearest centroids).
  [terapyon]

- Add ``voronoi_n_assign`` and ``voronoi_n_probe`` registry settings
  for configuring document cell assignments and query probing.
  [terapyon]

- Add upgrade step 1005 -> 1006: creates ``voronoi_cells`` KeywordIndex
  and Voronoi registry keys.
  [terapyon]

- Add provisional JSON API endpoint (``@@vectorsearch-api``) for E2E testing
  of different approximation algorithms. Accepts ``q``, ``limit``, and
  ``algorithm`` parameters, returns search results as JSON. The ``algorithm``
  parameter temporarily overrides the registry setting for that request only.
  Not a public API; subject to removal or change without notice.
  Usage: ``GET /Plone/@@vectorsearch-api?q=search+terms&algorithm=voronoi_2stage&limit=10``
  [terapyon]

Bug fixes:

- Fix ``_reindex_all_vector_indexes()`` and ``_clear_all_vector_indexes()``
  in control panel to include ``voronoi_cells`` index.
  [terapyon]


1.0a3 (unreleased)
------------------

Breaking changes:

- Migrate build configuration from ``setup.py`` to ``pyproject.toml``.
  ``setup.py`` is kept as a minimal shim for ``zc.buildout`` compatibility.
  [terapyon]

- Update ``src/collective/__init__.py`` to gracefully handle missing
  ``pkg_resources`` (setuptools >= 82). Falls back to PEP 420 implicit
  namespace packages when ``pkg_resources`` is unavailable.
  [terapyon]

Internal:

- Add ``plone.autoinclude.plugin`` entry point alongside existing
  ``z3c.autoinclude.plugin`` entry point.
  [terapyon]

- Replace ``pkg_resources`` usage in ``locales/update.py`` with
  ``pathlib.Path``.
  [terapyon]


1.0a2 (unreleased)
------------------

Features:

- Add multi-stage approximate nearest neighbor search based on
  `lsh-cascade-poc <https://github.com/cmscom/lsh-cascade-poc>`_ research:

  - Exhaustive Cosine: brute-force cosine similarity (default)
  - ITQ-LSH 2-Stage: Hamming distance ranking + cosine similarity on top-K
  - ITQ-LSH 3-Stage: pivot-based triangle inequality filtering + Hamming ranking + cosine

  Automatic fallback from 3-stage to 2-stage to exhaustive when ITQ/pivot data is unavailable.
  [terapyon]

- Add annotation-based data storage architecture (Version 1003).
  Vectors, ITQ hashes, and pivot distances are stored in content annotations as the
  single source of truth, resolving race conditions between event subscribers and
  catalog indexers.
  [terapyon]

- Add ITQ (Iterative Quantization) binary hashing for fast Hamming distance comparison.
  Pre-trained ITQ data (PCA matrix, rotation matrix, mean vector) included for
  All-MiniLM-L6-v2 and E5-Base Multilingual models.
  [terapyon]

- Add pivot-based filtering with 8 KeywordIndexes (pivot1-8) for triangle inequality pruning.
  Pre-computed pivot vectors included for All-MiniLM-L6-v2 and E5-Base Multilingual models.
  [terapyon]

- Add upgrade steps for migration from Version 1001 to 1005:

  - 1002: Add ITQ/pivot indexes and metadata columns (FieldIndex -> KeywordIndex)
  - 1003: Migrate vector data from VectorIndex internal BTrees to content annotations
  - 1004: Add llm_vector metadata column for direct vector access
  - 1005: Migrate hamming_distance_threshold setting to itq_candidates (top-K ranking)

  [terapyon]

- Mark unimplemented options as "(not yet available)" in control panel vocabularies
  (storage backends: FAISS, DuckDB, Annoy; approximation algorithm: HNSW).
  Add interface invariants to prevent saving unimplemented options.
  [terapyon]


1.0a1 (unreleased)
------------------

Features:

- Add VectorIndex, a custom ZCatalog index for storing and searching vector embeddings.
  [terapyon]

- Add pluggable embedding model architecture with ``IEmbeddingModelProvider`` interface.
  Three built-in providers: All-MiniLM-L6-v2 (FastEmbed), E5-Base Multilingual (FastEmbed),
  and E5-Base Multilingual GPU (Sentence Transformers).
  [terapyon]

- Add ``plone.registry``-based configuration system with ``IVectorSearchSettings`` interface.
  Configurable: embedding model, chunk size, storage backend, approximation algorithm,
  pivot threshold, and ITQ candidates.
  [terapyon]

- Add control panel at Site Setup -> Vector Search for configuration management.
  Includes Reindex All, Clear All Vectors buttons, index statistics,
  and incompatible index detection.
  [terapyon]

- Add ``AddVectorIndexView`` for creating VectorIndex instances via ZMI.
  [terapyon]

- Add event subscribers for automatic vectorization on content creation and modification.
  [terapyon]

- Add FastEmbed (CPU/ONNX) as the default embedding backend, with optional GPU support
  via ``[gpu]`` extras (PyTorch + Sentence Transformers).
  [terapyon]

- Add ``vectorsearch-download`` CLI command for offline model pre-downloading.
  [terapyon]

- Add reinstall safety: registry settings preserved with ``purge="false"``,
  catalog data retained during reinstall.
  [terapyon]

- Support Python 3.10 - 3.13 and Plone 6.0+.
  [terapyon]

- Initial release.
  [terapyon]
