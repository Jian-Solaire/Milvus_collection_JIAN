"""FastAPI RAG API endpoints."""

import time
import uuid
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from service.ingestion_service import IngestionService
from service.retrieval_service import RetrievalService
from config.settings import settings

app = FastAPI(title="Milvus RAG API", version="1.0.0")

# Initialize services
ingestion_service = IngestionService()
retrieval_service = RetrievalService()


class ApiResponse(BaseModel):
    """Standard API response."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_detail: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Ingestion request model."""

    texts: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None


class FileIngestRequest(BaseModel):
    """File ingestion request model."""

    file_paths: List[str]
    options: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    """Query request model."""

    query: str
    options: Optional[Dict[str, Any]] = None


class DeleteRequest(BaseModel):
    """Delete request model."""

    chunk_ids: List[str]


def success_response(data: Any, meta: Optional[Dict[str, Any]] = None) -> ApiResponse:
    """Create success response."""
    return ApiResponse(
        success=True,
        data=data,
        error=None,
        error_detail=None,
        meta=meta or {"timestamp": datetime.now().isoformat()},
    )


def error_response(error: str, detail: Optional[str] = None) -> ApiResponse:
    """Create error response."""
    return ApiResponse(
        success=False,
        data=None,
        error=error,
        error_detail=detail if settings.ERROR_DETAIL_ENABLED else None,
        meta={"timestamp": datetime.now().isoformat()},
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    result = retrieval_service.health_check()
    if result.get("success"):
        return success_response({"status": "healthy"})
    return JSONResponse(
        status_code=503,
        content=error_response("Service unhealthy", result.get("error")).model_dump(),
    )


@app.post("/knowledge/ingest", response_model=ApiResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents to knowledge base."""
    try:
        # Parse options
        options = request.options or {}
        chunk_opts = options.get("chunking", {})
        embedding_opts = options.get("embedding", {})

        # Extract parameters
        namespace = embedding_opts.get("namespace", settings.DEFAULT_NAMESPACE)
        chunk_size = chunk_opts.get("size", settings.DEFAULT_CHUNK_SIZE)
        chunk_overlap = chunk_opts.get("overlap", settings.DEFAULT_CHUNK_OVERLAP)
        async_processing = options.get("async_processing", settings.ASYNC_ENABLED)
        batch_size = options.get("batch_size", settings.ASYNC_BATCH_SIZE)

        # Ingest
        result = ingestion_service.ingest(
            texts=request.texts,
            metadata_list=request.metadata,
            namespace=namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            async_processing=async_processing,
        )

        if result.get("success"):
            return success_response(
                data={
                    "indexed_count": result.get("count"),
                    "chunks_generated": result.get("chunks_generated"),
                    "duration_seconds": result.get("duration_seconds"),
                },
                meta={"namespace": namespace},
            )
        else:
            return error_response(
                error=result.get("error", "Ingestion failed"),
                detail=result.get("message"),
            )

    except Exception as e:
        return error_response("Ingestion failed", str(e))


@app.post("/knowledge/ingest/files", response_model=ApiResponse)
async def ingest_files(request: FileIngestRequest):
    """Ingest files (PDF, Word, Excel, etc.) to knowledge base."""
    try:
        from pipeline.offline_pipeline import (
            load_and_convert_files,
            load_and_chunk_texts,
            index_chunks,
        )

        start_time = time.time()

        # Parse options
        options = request.options or {}
        chunk_opts = options.get("chunking", {})
        namespace = options.get("namespace", settings.DEFAULT_NAMESPACE)
        chunk_size = chunk_opts.get("size", settings.DEFAULT_CHUNK_SIZE)
        chunk_overlap = chunk_opts.get("overlap", settings.DEFAULT_CHUNK_OVERLAP)

        # Convert files to text
        convert_start = time.time()
        converted = load_and_convert_files(request.file_paths)
        convert_time = time.time() - convert_start

        if not converted:
            return error_response(
                "No valid files found", "All files failed to convert or are unsupported"
            )

        # Extract texts and metadata
        texts = [item["text"] for item in converted]
        metadata_list = [item["metadata"] for item in converted]

        # Chunk texts
        chunk_start = time.time()
        chunks = load_and_chunk_texts(
            texts=texts,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=namespace,
            metadata_list=metadata_list,
        )
        chunk_time = time.time() - chunk_start

        if not chunks:
            return error_response("No chunks generated", "Failed to chunk documents")

        # Index chunks
        index_start = time.time()
        indexed = index_chunks(chunks=chunks, namespace=namespace)
        index_time = time.time() - index_start

        total_time = time.time() - start_time

        return success_response(
            data={
                "files_processed": len(converted),
                "chunks_generated": len(chunks),
                "indexed_count": indexed,
                "duration_seconds": round(total_time, 3),
            },
            meta={
                "namespace": namespace,
                "timing": {
                    "document_processing_seconds": round(convert_time, 3),
                    "chunking_seconds": round(chunk_time, 3),
                    "indexing_seconds": round(index_time, 3),
                },
            },
        )

    except Exception as e:
        return error_response("File ingestion failed", str(e))


@app.post("/query", response_model=ApiResponse)
async def query_knowledge(request: QueryRequest):
    """Query knowledge base."""
    try:
        start_time = time.time()

        # Parse options
        options = request.options or {}
        search_opts = options.get("search", {})
        enhance_opts = options.get("enhance", {})

        # Extract parameters
        namespace = search_opts.get("namespace", settings.DEFAULT_NAMESPACE)
        top_k = search_opts.get("top_k", 5)
        search_type = search_opts.get("type", "hybrid")
        rrf_k = search_opts.get("rrf_k", settings.RRF_K)
        score_threshold = search_opts.get("score_threshold")

        # Enhancement options (default to False)
        enable_mqe = enhance_opts.get("mqe", False)
        enable_hyde = enhance_opts.get("hyde", False)
        mqe_expansions = enhance_opts.get("mqe_expand_queries", 3)

        # Search
        result = retrieval_service.search(
            query=request.query,
            namespace=namespace,
            top_k=top_k,
            search_type=search_type,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
            score_threshold=score_threshold,
            rrf_k=rrf_k,
        )

        search_time = time.time() - start_time

        if result.get("success"):
            return success_response(
                data={
                    "results": result.get("results", []),
                    "count": result.get("count", 0),
                    "query": request.query,
                    "search_type": search_type,
                    "duration_seconds": round(search_time, 3),
                },
            )
        else:
            return error_response(
                error=result.get("error", "Search failed"),
                detail=result.get("details"),
            )

    except Exception as e:
        return error_response("Search failed", str(e))


@app.get("/knowledge/stats", response_model=ApiResponse)
async def get_stats():
    """Get knowledge base statistics."""
    try:
        from core.embedding import get_dimension
        from stores.milvus_store import MilvusVectorStore

        dimension = get_dimension(1024)
        store = MilvusVectorStore(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD,
            collection_name=settings.MILVUS_COLLECTION_NAME,
            dense_dimension=dimension,
            metric_type=settings.METRIC_TYPE,
        )

        stats = store.get_collection_stats()
        return success_response(data=stats)

    except Exception as e:
        return error_response("Failed to get stats", str(e))


@app.delete("/knowledge/delete", response_model=ApiResponse)
async def delete_chunks(request: DeleteRequest):
    """Delete chunks by IDs."""
    try:
        result = ingestion_service.delete(chunk_ids=request.chunk_ids)

        if result.get("success"):
            return success_response(
                data={"deleted_count": result.get("deleted_count")},
            )
        else:
            return error_response("Delete failed")

    except Exception as e:
        return error_response("Delete failed", str(e))


@app.delete("/knowledge/clear", response_model=ApiResponse)
async def clear_knowledge():
    """Clear knowledge base."""
    try:
        result = ingestion_service.clear()

        if result.get("success"):
            return success_response(data={"message": result.get("message")})
        else:
            return error_response("Clear failed")

    except Exception as e:
        return error_response("Clear failed", str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
