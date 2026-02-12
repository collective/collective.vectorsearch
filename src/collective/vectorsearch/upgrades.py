# -*- coding: utf-8 -*-
"""Upgrade steps for collective.vectorsearch."""

from logging import getLogger

from BTrees.OOBTree import OOBTree
from plone import api
from zope.annotation.interfaces import IAnnotations

from collective.vectorsearch.annotations import (
    ANNOTATION_KEY_ITQ_HASHES,
    ANNOTATION_KEY_MODEL_ID,
    ANNOTATION_KEY_PIVOT_DISTANCES,
    ANNOTATION_KEY_VECTORS,
)

logger = getLogger("collective.vectorsearch.upgrades")


def upgrade_to_1001(context):
    """Initial upgrade step for version tracking.

    This step does nothing but establishes the upgrade infrastructure.
    Future upgrades will build on this.
    """
    logger.info("collective.vectorsearch: Upgrade to version 1001 complete")


def upgrade_to_1002(context):
    """Add ITQ/Pivot indexes and metadata columns.

    This upgrade adds:
    - pivot1-8 KeywordIndexes for triangle inequality filtering
      (supports multiple values per document for multi-chunk vectors)
    - itq_hashes metadata column for Hamming distance calculation
      (list of (high, low) tuples for all chunks)
    """
    catalog = api.portal.get_tool("portal_catalog")

    # Add pivot KeywordIndexes (pivot1 through pivot8)
    # KeywordIndex supports multiple values per document
    pivot_indexes = [f"pivot{i}" for i in range(1, 9)]
    for index_name in pivot_indexes:
        if index_name in catalog.indexes():
            # Check if it's already a KeywordIndex
            existing_index = catalog.Indexes.get(index_name)
            if existing_index is not None:
                meta_type = getattr(existing_index, "meta_type", None)
                if meta_type == "FieldIndex":
                    # Convert from FieldIndex to KeywordIndex
                    catalog.delIndex(index_name)
                    catalog.addIndex(index_name, "KeywordIndex")
                    logger.info(
                        f"Converted {index_name} from FieldIndex to KeywordIndex"
                    )
                else:
                    logger.info(
                        f"Index {index_name} already exists as {meta_type}, skipping"
                    )
        else:
            catalog.addIndex(index_name, "KeywordIndex")
            logger.info(f"Added KeywordIndex: {index_name}")

    # Handle metadata columns
    existing_columns = catalog.schema()

    # Remove old columns if they exist
    old_columns = ["itq_hash_high", "itq_hash_low"]
    for column_name in old_columns:
        if column_name in existing_columns:
            catalog.delColumn(column_name)
            logger.info(f"Removed old metadata column: {column_name}")

    # Add new itq_hashes column
    if "itq_hashes" not in existing_columns:
        catalog.addColumn("itq_hashes")
        logger.info("Added metadata column: itq_hashes")
    else:
        logger.info("Column itq_hashes already exists, skipping")

    logger.info("collective.vectorsearch: Upgrade to version 1002 complete")


def upgrade_to_1003(context):
    """Migrate vector data from VectorIndex to content annotations.

    This upgrade:
    1. Reads existing vectors from VectorIndex._docvectors
    2. Reads ITQ hashes and pivot distances from VectorIndex BTrees
    3. Stores all data in content annotations
    4. Updates VectorIndex._docid_to_path mapping
    5. Triggers catalog reindex for new indexer structure

    Note: This upgrade may take a while for large catalogs as it needs
    to process each indexed document.
    """
    catalog = api.portal.get_tool("portal_catalog")
    portal = api.portal.get()

    # Get VectorIndex
    if "llm_vector" not in catalog.Indexes:
        logger.info("No llm_vector index found, skipping migration")
        return

    vector_index = catalog.Indexes["llm_vector"]

    # Ensure the new data structure exists
    if not hasattr(vector_index, "_docid_to_path"):
        vector_index._docid_to_path = OOBTree()

    # Get model ID for migration
    model_id = getattr(vector_index, "indexed_with_model", None)

    # Check if we have any data to migrate
    if not hasattr(vector_index, "_docvectors") or len(vector_index._docvectors) == 0:
        logger.info("No vectors in VectorIndex, skipping migration")
        return

    migrated = 0
    errors = 0
    skipped = 0

    logger.info(
        f"Starting migration of {len(vector_index._docvectors)} documents "
        f"from VectorIndex to annotations"
    )

    # Iterate over all documents in VectorIndex
    for docid, vectors in vector_index._docvectors.items():
        try:
            # Get content object
            path = catalog.getpath(docid)
            obj = portal.unrestrictedTraverse(path, None)

            if obj is None:
                logger.warning(f"Could not find object at path {path}, skipping")
                skipped += 1
                continue

            # Check if object supports annotations
            try:
                annotations = IAnnotations(obj)
            except TypeError:
                logger.debug(f"Object at {path} does not support annotations, skipping")
                skipped += 1
                continue

            # Convert numpy arrays to lists for ZODB
            if vectors is not None:
                vectors_list = vectors.tolist()
            else:
                vectors_list = None

            # Get existing ITQ/pivot data from VectorIndex BTrees
            itq_hashes = None
            if hasattr(vector_index, "_itq_hashes"):
                itq_hashes = vector_index._itq_hashes.get(docid, None)
                if itq_hashes is not None:
                    itq_hashes = list(itq_hashes)

            pivot_distances = None
            if hasattr(vector_index, "_pivot_distances"):
                pivot_distances = vector_index._pivot_distances.get(docid, None)
                if pivot_distances is not None:
                    pivot_distances = list(pivot_distances)

            # Store in annotations
            annotations[ANNOTATION_KEY_VECTORS] = vectors_list
            annotations[ANNOTATION_KEY_ITQ_HASHES] = itq_hashes
            annotations[ANNOTATION_KEY_PIVOT_DISTANCES] = pivot_distances
            annotations[ANNOTATION_KEY_MODEL_ID] = model_id

            # Update path mapping
            vector_index._docid_to_path[docid] = path

            migrated += 1

            # Log progress every 100 documents
            if migrated % 100 == 0:
                logger.info(f"Migrated {migrated} documents...")

        except Exception as e:
            logger.warning(f"Failed to migrate doc {docid}: {e}")
            errors += 1

    logger.info(
        f"Migration complete: {migrated} migrated, {errors} errors, {skipped} skipped"
    )

    # Reindex to populate new indexer structure
    # This ensures the catalog indexes (pivot1-8, itq_hashes) are updated
    # using the new annotation-based indexers
    logger.info("Triggering catalog reindex for pivot and ITQ indexes...")
    try:
        reindex_indexes = [
            "pivot1",
            "pivot2",
            "pivot3",
            "pivot4",
            "pivot5",
            "pivot6",
            "pivot7",
            "pivot8",
        ]
        for index_name in reindex_indexes:
            if index_name in catalog.indexes():
                catalog.reindexIndex(index_name, None)
                logger.info(f"Reindexed {index_name}")
    except Exception as e:
        logger.warning(f"Error during reindex: {e}")

    logger.info("collective.vectorsearch: Upgrade to version 1003 complete")


def upgrade_to_1004(context):
    """Add llm_vector metadata column and remove _docvectors cache.

    This upgrade:
    1. Adds llm_vector as a metadata column in portal_catalog
    2. Reindexes llm_vector to populate the metadata from annotations
    3. The _docvectors IOBTree in VectorIndex is no longer used;
       vector data is now read from catalog metadata instead.
    """
    catalog = api.portal.get_tool("portal_catalog")

    # Add llm_vector metadata column if not already present
    existing_columns = catalog.schema()
    if "llm_vector" not in existing_columns:
        catalog.addColumn("llm_vector")
        logger.info("Added metadata column: llm_vector")
    else:
        logger.info("Column llm_vector already exists, skipping")

    # Reindex llm_vector to populate the metadata column from annotations
    if "llm_vector" in catalog.indexes():
        catalog.reindexIndex("llm_vector", None)
        logger.info("Reindexed llm_vector to populate metadata")

    # Clean up _docvectors from VectorIndex if it still exists
    if "llm_vector" in catalog.Indexes:
        vector_index = catalog.Indexes["llm_vector"]
        if hasattr(vector_index, "_docvectors"):
            del vector_index._docvectors
            vector_index._p_changed = True
            logger.info("Removed _docvectors cache from VectorIndex")

    logger.info("collective.vectorsearch: Upgrade to version 1004 complete")


def upgrade_to_1005(context):
    """Migrate registry key: hamming_distance_threshold -> itq_candidates.

    The Hamming distance search strategy changed from threshold-based filtering
    (hamming_dist <= threshold) to top-K ranking (sort by Hamming distance,
    take top N candidates). This matches the PoC (lsh-cascade-poc) behavior.

    Old key: collective.vectorsearch.hamming_distance_threshold (default: 3)
    New key: collective.vectorsearch.itq_candidates (default: 100)

    IMPORTANT: Do NOT use runImportStepFromProfile("plone.app.registry") here,
    as it re-imports the full registry/main.xml and resets ALL settings
    (including embedding_model) to their XML defaults.
    """
    from plone.registry import Record, field as registry_field

    registry = api.portal.get_tool("portal_registry")
    old_key = "collective.vectorsearch.hamming_distance_threshold"
    new_key = "collective.vectorsearch.itq_candidates"

    # Remove old registry key if it exists
    if old_key in registry.records:
        del registry.records[old_key]
        logger.info(f"Removed old registry key: {old_key}")

    # Add new key directly (without re-importing the full profile)
    if new_key not in registry.records:
        registry.records[new_key] = Record(
            registry_field.Int(title="ITQ Candidates (Stage 2)", default=100),
            value=100,
        )
        logger.info(f"Created registry key: {new_key} = 100")
    else:
        logger.info(f"Registry key {new_key} already exists, skipping")

    logger.info("collective.vectorsearch: Upgrade to version 1005 complete")
