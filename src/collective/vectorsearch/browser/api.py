"""Provisional JSON API for E2E testing of vector search algorithms.

NOT a public API. Subject to removal or change without notice.

Usage:
    GET /Plone/@@vectorsearch-api?q=search+terms&limit=10&algorithm=voronoi_2stage
"""

import json
import logging
import time

from plone import api
from Products.Five.browser import BrowserView

from collective.vectorsearch.interfaces import IVectorIndex

logger = logging.getLogger("collective.vectorsearch")

VALID_ALGORITHMS = [
    "exhaustive_cosine",
    "itq_lsh_2stage",
    "itq_lsh_3stage",
    "voronoi_2stage",
]


class VectorSearchAPIView(BrowserView):
    """JSON API view for testing vector search with different algorithms."""

    def __call__(self):
        query = self.request.get("q", "").strip()
        limit = int(self.request.get("limit", 20))
        algorithm = self.request.get("algorithm", "").strip() or None

        self.request.response.setHeader("Content-Type", "application/json")

        if not query:
            return json.dumps({"error": "Missing required parameter: q"})

        if algorithm and algorithm not in VALID_ALGORITHMS:
            return json.dumps({
                "error": f"Invalid algorithm: {algorithm}",
                "valid_algorithms": VALID_ALGORITHMS,
            })

        catalog = api.portal.get_tool("portal_catalog")

        # Find VectorIndex
        vector_index = None
        for index_id in catalog.indexes():
            idx = catalog.Indexes.get(index_id)
            if idx is not None and IVectorIndex.providedBy(idx):
                vector_index = idx
                break

        if vector_index is None:
            return json.dumps({"error": "No VectorIndex found in catalog"})

        # Apply algorithm override if requested
        if algorithm:
            vector_index._settings_override = {
                "approximation_algorithm": algorithm,
            }

        try:
            t0 = time.perf_counter()
            brains = catalog(llm_vector=query, sort_limit=limit)
            elapsed = time.perf_counter() - t0
        finally:
            if algorithm:
                vector_index._settings_override = None

        results = []
        for brain in brains[:limit]:
            modified = brain.modified
            if modified is not None:
                modified = modified.ISO8601()
            results.append({
                "title": brain.Title,
                "url": brain.getURL(),
                "path": brain.getPath(),
                "modified": modified,
                "type": brain.portal_type,
                "description": brain.Description,
            })

        effective_algorithm = algorithm or api.portal.get_registry_record(
            "collective.vectorsearch.approximation_algorithm",
            default="exhaustive_cosine",
        )

        return json.dumps({
            "query": query,
            "algorithm": effective_algorithm,
            "count": len(results),
            "elapsed_seconds": round(elapsed, 4),
            "results": results,
        }, ensure_ascii=False)
