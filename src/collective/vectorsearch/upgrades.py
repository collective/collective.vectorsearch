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
