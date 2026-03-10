"""Ingestion service for document indexing."""

import asyncio
import logging
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.offline_pipeline import (
    load_and_chunk_texts,
    index_chunks,
    create_rag_pipeline,
)
from stores.milvus_store import MilvusVectorStore
from config.settings import settings

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for ingesting documents to vector store."""

    def __init__(self, store: Optional[MilvusVectorStore] = None):
        """Initialize ingestion service.

        Args:
            store: Optional Milvus store instance
        """
        self.store = store

    def ingest(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        batch_size: int = 32,
        async_processing: bool = False,
    ) -> Dict[str, Any]:
        """Ingest documents to vector store.

        Args:
            texts: List of text contents
            metadata_list: Optional metadata for each text
            namespace: Namespace for isolation
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            batch_size: Batch size for processing
            async_processing: Enable async processing (not used in sync context)

        Returns:
            Result dict with count and status
        """
        # Always use sync processing in FastAPI context
        return self._ingest_sync(
            texts=texts,
            metadata_list=metadata_list,
            namespace=namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
        )

    def _ingest_sync(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]],
        namespace: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Synchronous ingestion."""
        start_time = datetime.now()

        # Chunk texts
        chunks = load_and_chunk_texts(
            texts=texts,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=namespace,
            metadata_list=metadata_list,
        )

        if not chunks:
            return {
                "success": False,
                "count": 0,
                "message": "No valid chunks generated",
                "error": "Empty text input",
            }

        # Index chunks
        indexed = index_chunks(
            store=self.store,
            chunks=chunks,
            batch_size=batch_size,
            namespace=namespace,
        )

        duration = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "count": indexed,
            "chunks_generated": len(chunks),
            "duration_seconds": duration,
            "namespace": namespace,
        }

    async def _ingest_async(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]],
        namespace: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Asynchronous ingestion."""
        loop = asyncio.get_event_loop()

        # Run sync operations in thread pool
        result = await loop.run_in_executor(
            None,
            self._ingest_sync,
            texts,
            metadata_list,
            namespace,
            chunk_size,
            chunk_overlap,
            batch_size,
        )

        return result

    def delete(
        self,
        chunk_ids: List[str],
    ) -> Dict[str, Any]:
        """Delete chunks by IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Result dict
        """
        if not self.store:
            from ..core.embedding import get_dimension

            dimension = get_dimension(1024)
            self.store = MilvusVectorStore(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD,
                collection_name=settings.MILVUS_COLLECTION_NAME,
                dense_dimension=dimension,
                metric_type=settings.METRIC_TYPE,
            )

        success = self.store.delete_vectors(ids=chunk_ids)

        return {
            "success": success,
            "deleted_count": len(chunk_ids) if success else 0,
        }

    def clear(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Clear the vector store.

        Args:
            namespace: Optional namespace filter

        Returns:
            Result dict
        """
        if not self.store:
            from ..core.embedding import get_dimension

            dimension = get_dimension(1024)
            self.store = MilvusVectorStore(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD,
                collection_name=settings.MILVUS_COLLECTION_NAME,
                dense_dimension=dimension,
                metric_type=settings.METRIC_TYPE,
            )

        success = self.store.clear_collection()

        return {
            "success": success,
            "message": "Collection cleared"
            if success
            else "Failed to clear collection",
        }
