Changelog
=========


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

- Support Python 3.10 - 3.13 and Plone 6.0 / 6.1.
  [terapyon]

- Initial release.
  [terapyon]
