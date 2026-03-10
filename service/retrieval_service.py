"""Retrieval service for query answering."""

import logging
import os
import sys
from typing import List, Dict, Any, Optional

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.online_pipeline import search_vectors, search_vectors_expanded
from stores.milvus_store import MilvusVectorStore
from config.settings import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for query retrieval and answering."""

    def __init__(self, store: Optional[MilvusVectorStore] = None):
        """Initialize retrieval service.

        Args:
            store: Optional Milvus store instance
        """
        self.store = store

    def search(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 5,
        search_type: str = "hybrid",
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        score_threshold: Optional[float] = None,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """Search knowledge base.

        Args:
            query: Query text
            namespace: Namespace
            top_k: Number of results
            search_type: Search type (dense, sparse, hybrid)
            enable_mqe: Enable MQE
            enable_hyde: Enable HyDE
            score_threshold: Score threshold
            rrf_k: RRF parameter

        Returns:
            Search results
        """
        if not query:
            return {
                "success": False,
                "results": [],
                "error": "Empty query",
            }

        # Use search_type from settings if not provided
        if search_type not in ["dense", "sparse", "hybrid"]:
            search_type = "hybrid"

        try:
            if enable_mqe or enable_hyde:
                results = search_vectors_expanded(
                    store=self.store,
                    query=query,
                    top_k=top_k,
                    namespace=namespace,
                    score_threshold=score_threshold,
                    enable_mqe=enable_mqe,
                    enable_hyde=enable_hyde,
                    search_type=search_type,
                )
            else:
                results = search_vectors(
                    store=self.store,
                    query=query,
                    top_k=top_k,
                    namespace=namespace,
                    score_threshold=score_threshold,
                    search_type=search_type,
                )

            # Format results
            formatted_results = []
            for r in results:
                formatted_results.append(
                    {
                        "id": r.get("id"),
                        "chunk_id": r.get("chunk_id"),
                        "text": r.get("text", ""),
                        "score": r.get("score", 0),
                        "document_id": r.get("document_id"),
                        "source_path": r.get("source_path"),
                        "heading_path": r.get("heading_path"),
                        "metadata": r.get("metadata", {}),
                    }
                )

            return {
                "success": True,
                "results": formatted_results,
                "count": len(formatted_results),
                "query": query,
                "search_type": search_type,
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "success": False,
                "results": [],
                "error": str(e),
            }

    def health_check(self) -> Dict[str, Any]:
        """Health check.

        Returns:
            Health status
        """
        try:
            if self.store is None:
                dimension = settings.MODEL_DIMENSION or 1024
                self.store = MilvusVectorStore(
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    user=settings.MILVUS_USER,
                    password=settings.MILVUS_PASSWORD,
                    collection_name=settings.MILVUS_COLLECTION_NAME,
                    dense_dimension=dimension,
                    metric_type=settings.METRIC_TYPE,
                )

            healthy = self.store.health_check()
            return {
                "success": healthy,
                "status": "healthy" if healthy else "unhealthy",
            }
        except Exception as e:
            return {
                "success": False,
                "status": "unhealthy",
                "error": str(e),
            }
