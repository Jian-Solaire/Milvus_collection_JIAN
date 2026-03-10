"""Unit tests for chunking module."""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
from pipeline.offline_pipeline import MarkdownChunker, load_and_chunk_texts


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    def test_split_paragraphs_with_headings(self):
        """Test paragraph splitting with headings."""
        text = """# 标题1

这是第一段内容。

## 标题2

这是第二段内容。

# 标题3

这是第三段内容。
"""
        paragraphs = MarkdownChunker.split_paragraphs_with_headings(text)

        assert len(paragraphs) >= 2
        # Check headings are captured
        headings = [p.get("heading_path") for p in paragraphs if p.get("heading_path")]
        assert "标题1" in str(headings)

    def test_chunk_paragraphs(self):
        """Test paragraph chunking."""
        paragraphs = [
            {
                "content": "这是第一段内容" * 100,
                "heading_path": "标题1",
                "start": 0,
                "end": 500,
            },
            {
                "content": "这是第二段内容" * 100,
                "heading_path": "标题2",
                "start": 500,
                "end": 1000,
            },
        ]

        chunks = MarkdownChunker.chunk_paragraphs(
            paragraphs, chunk_tokens=500, overlap_tokens=50
        )

        assert len(chunks) >= 1
        assert "content" in chunks[0]

    def test_approx_token_len(self):
        """Test token length approximation."""
        text = "Hello world 这是中文"
        length = MarkdownChunker._approx_token_len(text)

        assert length > 0


class TestLoadAndChunkTexts:
    """Tests for load_and_chunk_texts function."""

    def test_chunk_single_text(self):
        """Test chunking single text."""
        texts = ["这是测试内容" * 100]

        chunks = load_and_chunk_texts(
            texts=texts,
            chunk_size=100,
            chunk_overlap=10,
            namespace="test",
        )

        assert len(chunks) > 0
        assert "id" in chunks[0]
        assert "content" in chunks[0]
        assert "metadata" in chunks[0]

    def test_chunk_multiple_texts(self):
        """Test chunking multiple texts."""
        texts = [
            "第一个文档内容" * 50,
            "第二个文档内容" * 50,
        ]

        chunks = load_and_chunk_texts(
            texts=texts,
            chunk_size=100,
            chunk_overlap=10,
            namespace="test",
        )

        assert len(chunks) >= 2

    def test_metadata_preservation(self):
        """Test metadata is preserved."""
        texts = ["测试内容"]
        metadata = [{"source_path": "/test/doc.md", "custom_field": "value"}]

        chunks = load_and_chunk_texts(
            texts=texts,
            metadata_list=metadata,
            namespace="test",
        )

        assert chunks[0]["metadata"]["source_path"] == "/test/doc.md"
        assert chunks[0]["metadata"]["custom_field"] == "value"

    def test_empty_text_handling(self):
        """Test empty text is handled."""
        texts = ["", "  ", "有效内容"]

        chunks = load_and_chunk_texts(
            texts=texts,
            namespace="test",
        )

        # Should only have chunks for valid content
        assert len(chunks) >= 1

    def test_namespace_isolation(self):
        """Test namespace is set correctly."""
        texts = ["测试"]

        chunks = load_and_chunk_texts(
            texts=texts,
            namespace="custom_ns",
        )

        assert chunks[0]["metadata"]["namespace"] == "custom_ns"
