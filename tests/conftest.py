"""Test fixtures for RAG tests."""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_milvus_store():
    """Mock Milvus store."""
    store = MagicMock()
    store.add_vectors = Mock(return_value=True)
    store.search_dense = Mock(return_value=[])
    store.search_sparse = Mock(return_value=[])
    store.search_hybrid = Mock(return_value=[])
    store.delete_vectors = Mock(return_value=True)
    store.clear_collection = Mock(return_value=True)
    store.health_check = Mock(return_value=True)
    store.get_collection_stats = Mock(
        return_value={
            "store_type": "milvus",
            "entities_count": 0,
        }
    )
    return store


@pytest.fixture
def sample_chunks():
    """Sample chunk data."""
    return [
        {
            "id": "chunk_1",
            "content": "这是第一个测试文本块",
            "metadata": {
                "document_id": "doc_1",
                "namespace": "default",
            },
        },
        {
            "id": "chunk_2",
            "content": "这是第二个测试文本块",
            "metadata": {
                "document_id": "doc_1",
                "namespace": "default",
            },
        },
    ]


@pytest.fixture
def sample_texts():
    """Sample text data."""
    return [
        "这是一个测试文档内容。包含多个段落。",
        "这是第二个文档内容。用于测试分块功能。",
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata."""
    return [
        {"source_path": "/test/doc1.md", "document_id": "doc_1"},
        {"source_path": "/test/doc2.md", "document_id": "doc_2"},
    ]
