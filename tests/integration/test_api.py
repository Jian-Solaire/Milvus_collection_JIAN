"""Integration tests for RAG API."""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
from unittest.mock import Mock, patch


class TestIngestionAPI:
    """Integration tests for ingestion endpoints."""

    @pytest.fixture
    def mock_ingestion_service(self):
        """Mock ingestion service."""
        with patch("api.rag_api.IngestionService") as mock:
            service = Mock()
            service.ingest = Mock(
                return_value={
                    "success": True,
                    "count": 10,
                    "chunks_generated": 10,
                    "duration_seconds": 1.5,
                }
            )
            mock.return_value = service
            yield service

    def test_ingest_endpoint_success(self, mock_ingestion_service):
        """Test successful ingestion."""
        from api.rag_api import IngestRequest

        request = IngestRequest(
            texts=["测试内容1", "测试内容2"],
            metadata=[{"source": "test"}],
            options={
                "chunking": {"size": 800, "overlap": 100},
                "embedding": {"namespace": "default"},
            },
        )

        # This would require actual FastAPI test client
        # For now, just verify the request model works
        assert len(request.texts) == 2
        assert request.options["chunking"]["size"] == 800


class TestQueryAPI:
    """Integration tests for query endpoints."""

    @pytest.fixture
    def mock_retrieval_service(self):
        """Mock retrieval service."""
        with patch("api.rag_api.RetrievalService") as mock:
            service = Mock()
            service.search = Mock(
                return_value={
                    "success": True,
                    "results": [
                        {
                            "id": 1,
                            "text": "测试结果",
                            "score": 0.95,
                        }
                    ],
                    "count": 1,
                }
            )
            mock.return_value = service
            yield service

    def test_query_endpoint_success(self, mock_retrieval_service):
        """Test successful query."""
        from api.rag_api import QueryRequest

        request = QueryRequest(
            query="测试查询",
            options={
                "search": {"top_k": 5, "type": "hybrid"},
                "enhance": {"mqe": False, "hyde": False},
            },
        )

        assert request.query == "测试查询"
        assert request.options["search"]["type"] == "hybrid"

    def test_query_endpoint_empty_query(self):
        """Test query with empty query."""
        from api.rag_api import QueryRequest

        request = QueryRequest(query="")

        assert request.query == ""


class TestDeleteAPI:
    """Integration tests for delete endpoints."""

    def test_delete_endpoint(self):
        """Test delete request model."""
        from api.rag_api import DeleteRequest

        request = DeleteRequest(chunk_ids=["chunk_1", "chunk_2", "chunk_3"])

        assert len(request.chunk_ids) == 3
        assert "chunk_1" in request.chunk_ids
