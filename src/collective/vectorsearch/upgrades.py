# -*- coding: utf-8 -*-
"""Upgrade steps for collective.vectorsearch."""

from logging import getLogger

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
    from plone import api

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
                    logger.info(f"Converted {index_name} from FieldIndex to KeywordIndex")
                else:
                    logger.info(f"Index {index_name} already exists as {meta_type}, skipping")
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
