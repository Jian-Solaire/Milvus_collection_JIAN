"""Offline pipeline: document -> chunk -> index."""

import hashlib
import logging
import os
import sys
from typing import List, Dict, Any, Optional
import re

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedding import get_bge_vectorizer, get_dimension
from stores.milvus_store import MilvusConnectionManager, MilvusVectorStore
from config.settings import settings

logger = logging.getLogger(__name__)


# ====== Multi-format document conversion ======


def _get_markitdown_instance():
    """Get a configured MarkItDown instance for document conversion."""
    try:
        from markitdown import MarkItDown

        return MarkItDown()
    except ImportError:
        logger.warning(
            "[WARNING] MarkItDown not available. Install with: pip install markitdown"
        )
        return None


def _is_supported_format(path: str) -> bool:
    """Check if file format is supported."""
    ext = (os.path.splitext(path)[1] or "").lower()
    supported = {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
        ".zip",
        ".tar",
        ".gz",
        ".rar",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".css",
        ".scss",
        ".log",
        ".conf",
        ".ini",
        ".cfg",
        ".yaml",
        ".yml",
        ".toml",
    }
    return ext in supported


def _convert_to_markdown(path: str) -> str:
    """Convert any supported file format to markdown text."""
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return ""

    ext = (os.path.splitext(path)[1] or "").lower()

    # PDF uses enhanced processing
    if ext == ".pdf":
        return _enhanced_pdf_processing(path)

    # Other formats use MarkItDown
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)

    try:
        result = md_instance.convert(path)
        text = getattr(result, "text_content", None)
        if isinstance(text, str) and text.strip():
            return text
        return ""
    except Exception as e:
        logger.warning(f"MarkItDown failed for {path}: {e}")
        return _fallback_text_reader(path)


def _enhanced_pdf_processing(path: str) -> str:
    """Enhanced PDF processing with post-processing cleanup."""
    import re

    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)

    try:
        result = md_instance.convert(path)
        raw_text = getattr(result, "text_content", None)
        if not raw_text or not raw_text.strip():
            return ""

        lines = raw_text.splitlines()
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(line) <= 2 and not line.isdigit():
                continue
            if re.match(r"^\d+$", line):
                continue
            cleaned_lines.append(line)

        merged = []
        for line in cleaned_lines:
            if len(line) < 60 and merged and not merged[-1].endswith(("：", ":")):
                merged[-1] = merged[-1] + " " + line
            else:
                merged.append(line)

        return "\n\n".join(merged)

    except Exception as e:
        logger.warning(f"Enhanced PDF processing failed for {path}: {e}")
        return _fallback_text_reader(path)


def _fallback_text_reader(path: str) -> str:
    """Simple fallback reader for text files."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


def load_and_convert_files(
    file_paths: List[str],
    parallel: bool = True,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """Load and convert multiple files to text.

    Args:
        file_paths: List of file paths
        parallel: Enable parallel loading (default True)
        max_workers: Max parallel workers (default 4)

    Returns:
        List of dicts with 'text' and 'metadata'
    """
    if not parallel or len(file_paths) <= 1:
        return _load_files_sequential(file_paths)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_convert_single_file, path): path for path in file_paths
        }
        for future in as_completed(future_to_path):
            result = future.result()
            if result:
                results.append(result)

    return results


def _load_files_sequential(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Load files sequentially (fallback)."""
    results = []
    for path in file_paths:
        result = _convert_single_file(path)
        if result:
            results.append(result)
    return results


def _convert_single_file(path: str) -> Optional[Dict[str, Any]]:
    """Convert a single file to text dict."""
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return None

    if not _is_supported_format(path):
        logger.warning(f"Unsupported format: {path}")
        return None

    text = _convert_to_markdown(path)
    if not text.strip():
        logger.warning(f"No content extracted from: {path}")
        return None

    return {
        "text": text,
        "metadata": {
            "source_path": path,
            "file_name": os.path.basename(path),
            "file_ext": os.path.splitext(path)[1],
        },
    }


class MarkdownChunker:
    """Markdown-aware text chunker."""

    @staticmethod
    def _is_cjk(ch: str) -> bool:
        """Check if character is CJK."""
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
        )

    @staticmethod
    def _approx_token_len(text: str) -> int:

        if not text:
            return 0

        # 1. 统计 CJK 字符 (中日韩算作单字 Token)
        # 使用正则表达式直接匹配所有中文字符
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))

        # 2. 统计英文单词/数字
        # \b[a-zA-Z0-9-]+\b 匹配完整的单词、数字或带连字符的词
        # 这样即使 "Hello世界" 连在一起，也能识别出 1个英文 + 2个中文
        eng_tokens = len(re.findall(r"[a-zA-Z0-9-]+", text))

        # 3. 统计主要标点符号（可选，RAG中通常也占位）
        # 如果你想更精确，可以把常用标点也算进去
        # punct_count = len(re.findall(r'[，。！？、；：""''（）(),.!?:]', text))

        return cjk_count + eng_tokens

    @staticmethod
    def split_paragraphs_with_headings(text: str) -> List[Dict]:
        """Split text into paragraphs with heading information."""
        lines = text.splitlines()
        heading_stack: List[str] = []
        paragraphs: List[Dict] = []
        buf: List[str] = []
        char_pos = 0

        def flush_buf(end_pos: int) -> None:
            if not buf:
                return
            content = "\n".join(buf).strip()
            if not content:
                return
            paragraphs.append(
                {
                    "content": content,
                    "heading_path": " > ".join(heading_stack)
                    if heading_stack
                    else None,
                    "start": max(0, end_pos - len(content)),
                    "end": end_pos,
                }
            )

        for ln in lines:
            raw = ln
            if raw.strip().startswith("#"):
                flush_buf(char_pos)
                level = len(raw) - len(raw.lstrip("#"))
                title = raw.lstrip("#").strip()
                if level <= 0:
                    level = 1
                if level <= len(heading_stack):
                    heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)
                char_pos += len(raw) + 1
                continue

            if raw.strip() == "":
                flush_buf(char_pos)
                buf = []
            else:
                buf.append(raw)
            char_pos += len(raw) + 1

        flush_buf(char_pos)
        if not paragraphs:
            paragraphs = [
                {"content": text, "heading_path": None, "start": 0, "end": len(text)}
            ]
        return paragraphs

    @staticmethod
    def chunk_paragraphs(
        paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int
    ) -> List[Dict]:
        """Chunk paragraphs into larger chunks."""
        chunks: List[Dict] = []
        cur: List[Dict] = []
        cur_tokens = 0
        i = 0

        while i < len(paragraphs):
            p = paragraphs[i]
            p_tokens = MarkdownChunker._approx_token_len(p["content"]) or 1

            # If paragraph is bigger than chunk size, skip it or add alone
            if p_tokens > chunk_tokens:
                # If we have current chunk, save it first
                if cur:
                    content = "\n\n".join(x["content"] for x in cur)
                    chunks.append(
                        {
                            "content": content,
                            "start": cur[0]["start"],
                            "end": cur[-1]["end"],
                            "heading_path": next(
                                (
                                    x["heading_path"]
                                    for x in reversed(cur)
                                    if x.get("heading_path")
                                ),
                                None,
                            ),
                        }
                    )
                    cur = []
                    cur_tokens = 0

                    # Handle overlap
                    if overlap_tokens > 0:
                        kept = []
                        kept_tokens = 0
                        for x in reversed(chunks[-1]["content"].split("\n\n")):
                            pass  # simplified

                # Add the big paragraph alone
                chunks.append(
                    {
                        "content": p["content"],
                        "start": p["start"],
                        "end": p["end"],
                        "heading_path": p.get("heading_path"),
                    }
                )
                i += 1
                continue

            if cur_tokens + p_tokens <= chunk_tokens or not cur:
                # Add paragraph to current chunk
                cur.append(p)
                cur_tokens += p_tokens
                i += 1
            else:
                # Current chunk is full, save it
                content = "\n\n".join(x["content"] for x in cur)
                chunks.append(
                    {
                        "content": content,
                        "start": cur[0]["start"],
                        "end": cur[-1]["end"],
                        "heading_path": next(
                            (
                                x["heading_path"]
                                for x in reversed(cur)
                                if x.get("heading_path")
                            ),
                            None,
                        ),
                    }
                )

                # Handle overlap
                if overlap_tokens > 0 and len(cur) > 1:
                    kept: List[Dict] = []
                    kept_tokens = 0
                    for x in reversed(cur):
                        t = MarkdownChunker._approx_token_len(x["content"]) or 1
                        if kept_tokens + t > overlap_tokens:
                            break
                        kept.append(x)
                        kept_tokens += t
                    cur = list(reversed(kept))
                    cur_tokens = kept_tokens
                else:
                    # No overlap, start fresh
                    cur = []
                    cur_tokens = 0

        # Don't forget the last chunk
        if cur:
            content = "\n\n".join(x["content"] for x in cur)
            chunks.append(
                {
                    "content": content,
                    "start": cur[0]["start"],
                    "end": cur[-1]["end"],
                    "heading_path": next(
                        (
                            x["heading_path"]
                            for x in reversed(cur)
                            if x.get("heading_path")
                        ),
                        None,
                    ),
                }
            )

        return chunks


def load_and_chunk_texts(
    texts: List[str],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    namespace: str = "default",
    metadata_list: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict]:
    """Load and chunk texts.

    Args:
        texts: List of text contents
        chunk_size: Chunk size in tokens
        chunk_overlap: Overlap in tokens
        namespace: Namespace for isolation
        metadata_list: Optional metadata for each text

    Returns:
        List of chunk dictionaries
    """
    logger.info(
        f"Chunking {len(texts)} texts, chunk_size={chunk_size}, overlap={chunk_overlap}"
    )

    chunks: List[Dict] = []
    metadata_list = metadata_list or [{} for _ in texts]

    for idx, (text, metadata) in enumerate(zip(texts, metadata_list)):
        if not text or not text.strip():
            logger.warning(f"Skipping empty text at index {idx}")
            continue

        # Split into paragraphs
        para = MarkdownChunker.split_paragraphs_with_headings(text)

        token_chunks = MarkdownChunker.chunk_paragraphs(
            para,
            chunk_tokens=max(1, chunk_size),
            overlap_tokens=max(0, chunk_overlap),
        )

        # Process each chunk
        for chunk_idx, ch in enumerate(token_chunks):
            content = ch["content"]
            if not content.strip():
                continue

            # Generate IDs
            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            doc_id = metadata.get("document_id", f"doc_{idx}")
            chunk_id = f"{doc_id}_{chunk_idx}_{content_hash[:8]}"

            chunk = {
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "document_id": doc_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(token_chunks),
                    "namespace": namespace,
                    "source_path": metadata.get("source_path", ""),
                    "heading_path": ch.get("heading_path"),
                    "start": ch.get("start", 0),
                    "end": ch.get("end", 0),
                    **metadata,
                },
            }
            chunks.append(chunk)

    logger.info(f"Chunked {len(texts)} texts into {len(chunks)} chunks")
    return chunks


def index_chunks(
    store: Optional[MilvusVectorStore] = None,
    chunks: Optional[List[Dict]] = None,
    batch_size: int = 0,
    namespace: str = "default",
) -> int:
    """Index chunks to Milvus.

    Args:
        store: Milvus store instance
        chunks: List of chunk dictionaries
        batch_size: Batch size for embedding
        namespace: Namespace

    Returns:
        Number of indexed chunks
    """
    if not chunks:
        logger.warning("No chunks to index")
        return 0

    # Use default batch size from settings if not specified
    if batch_size <= 0:
        batch_size = settings.EMBEDDING_BATCH_SIZE

    # Get or create store
    if store is None:
        dimension = get_dimension(1024)
        store = MilvusConnectionManager.get_instance(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD,
            collection_name=settings.MILVUS_COLLECTION_NAME,
            dense_dimension=dimension,
            metric_type=settings.METRIC_TYPE,
        )

    # Get vectorizer
    vectorizer = get_bge_vectorizer(
        model_name=settings.MODEL_NAME,
        cache_dir=settings.MODEL_CACHE_DIR,
        device=settings.MODEL_DEVICE,
        enable_dense=settings.MODEL_ENABLE_DENSE,
        enable_sparse=settings.MODEL_ENABLE_SPARSE,
    )

    # Process chunks in batches
    indexed_count = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [ch["content"] for ch in batch]

        # Generate embeddings
        embeddings = vectorizer.encode(
            texts=texts,
            batch_size=batch_size,
            enable_dense=True,
            enable_sparse=True,
        )

        dense_vectors = embeddings.get("dense", [])
        sparse_vectors = embeddings.get("sparse", [])

        # Prepare metadata
        metadata_list = []
        for ch in batch:
            meta = ch.get("metadata", {})
            meta.update(
                {
                    "chunk_id": ch["id"],
                    "namespace": namespace,
                }
            )
            metadata_list.append(meta)

        # Add to store
        success = store.add_vectors(
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            texts=texts,
            metadata=metadata_list,
        )

        if success:
            indexed_count += len(batch)
            logger.info(f"Indexed batch {i // batch_size + 1}: {len(batch)} chunks")
        else:
            logger.error(f"Failed to index batch {i // batch_size + 1}")

    return indexed_count


def create_rag_pipeline(
    milvus_host: Optional[str] = None,
    milvus_port: Optional[int] = None,
    collection_name: str = "rag_vectors",
    rag_namespace: str = "default",
) -> Dict[str, Any]:
    """Create a complete RAG pipeline with Milvus and BGE-M3.

    Args:
        milvus_host: Milvus host
        milvus_port: Milvus port
        collection_name: Collection name
        rag_namespace: Namespace

    Returns:
        Dict containing store, namespace, and helper functions
    """
    host = milvus_host or settings.MILVUS_HOST
    port = milvus_port or settings.MILVUS_PORT
    dimension = get_dimension(1024)

    store = MilvusConnectionManager.get_instance(
        host=host,
        port=port,
        user=settings.MILVUS_USER,
        password=settings.MILVUS_PASSWORD,
        collection_name=collection_name,
        dense_dimension=dimension,
        metric_type=settings.METRIC_TYPE,
    )

    def add_documents(
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> int:
        """Add documents to RAG pipeline."""
        chunks = load_and_chunk_texts(
            texts=texts,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=rag_namespace,
            metadata_list=metadata_list,
        )
        return index_chunks(
            store=store,
            chunks=chunks,
            namespace=rag_namespace,
        )

    def search(
        query: str,
        top_k: int = 8,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Search RAG knowledge base (hybrid by default)."""
        from pipeline.online_pipeline import search_vectors

        return search_vectors(
            store=store,
            query=query,
            top_k=top_k,
            namespace=rag_namespace,
            score_threshold=score_threshold,
        )

    def search_advanced(
        query: str,
        top_k: int = 8,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Advanced search with query expansion."""
        from pipeline.online_pipeline import search_vectors_expanded

        return search_vectors_expanded(
            store=store,
            query=query,
            top_k=top_k,
            namespace=rag_namespace,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
            score_threshold=score_threshold,
        )

    def get_stats() -> Dict[str, Any]:
        """Get pipeline statistics."""
        return store.get_collection_stats()

    def clear() -> bool:
        """Clear the collection."""
        return store.clear_collection()

    return {
        "store": store,
        "namespace": rag_namespace,
        "add_documents": add_documents,
        "search": search,
        "search_advanced": search_advanced,
        "get_stats": get_stats,
        "clear": clear,
    }
