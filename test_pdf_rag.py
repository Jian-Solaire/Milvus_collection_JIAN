"""Test PDF ingestion and query - detailed timing"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.offline_pipeline import (
    load_and_convert_files,
    load_and_chunk_texts,
    index_chunks,
)
from core.embedding import get_bge_vectorizer, get_dimension
from stores.milvus_store import MilvusVectorStore, MilvusConnectionManager
from config.settings import settings


DEFAULT_QUERY = "A22赛题面部驱动模型的具体要求"


def test_pdf_detailed(pdf_path: str):
    """Test PDF ingestion with detailed timing"""
    print("=" * 60)
    print("PDF INGESTION TEST - DETAILED TIMING")
    print("=" * 60)

    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        return

    # Get store
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

    # Pre-initialize model
    print("\n[0] Pre-initializing BGE-M3 model...")
    t0 = time.time()
    vectorizer = get_bge_vectorizer(
        model_name=settings.MODEL_NAME,
        cache_dir=settings.MODEL_CACHE_DIR,
        device=settings.MODEL_DEVICE,
        enable_dense=settings.MODEL_ENABLE_DENSE,
        enable_sparse=settings.MODEL_ENABLE_SPARSE,
    )
    t1 = time.time()
    print(f"    Model init: {t1 - t0:.2f}s")

    # Step 1: Load PDF
    print("\n[1] Loading PDF...")
    t2 = time.time()
    docs = load_and_convert_files([pdf_path])
    t3 = time.time()
    print(f"    [1a] PDF convert: {t3 - t2:.2f}s")

    if not docs:
        print("    ERROR: No content extracted!")
        return

    for i, doc in enumerate(docs):
        print(f"    Doc {i + 1}: {len(doc.get('text', ''))} chars")

    # Step 2: Chunk
    print("\n[2] Chunking...")
    t4 = time.time()
    chunks = load_and_chunk_texts(
        texts=[doc["text"] for doc in docs],
        chunk_size=800,
        chunk_overlap=0,
        namespace="default",
        metadata_list=[doc["metadata"] for doc in docs],
    )
    t5 = time.time()
    print(f"    [2a] Split + chunk: {t5 - t4:.2f}s")
    print(f"    Generated {len(chunks)} chunks")

    # Step 3: Index (detailed)
    print("\n[3] Indexing to Milvus...")
    store.clear_collection()

    t6 = time.time()
    # 3a. Get vectorizer (should be cached)
    vectorizer = get_bge_vectorizer(
        model_name=settings.MODEL_NAME,
        cache_dir=settings.MODEL_CACHE_DIR,
        device=settings.MODEL_DEVICE,
        enable_dense=settings.MODEL_ENABLE_DENSE,
        enable_sparse=settings.MODEL_ENABLE_SPARSE,
    )
    t7 = time.time()
    print(f"    [3a] Get vectorizer: {t7 - t6:.3f}s")

    # 3b. Prepare data
    texts = [ch["content"] for ch in chunks]
    t8 = time.time()
    print(f"    [3b] Prepare texts: {t8 - t7:.3f}s")

    # 3c. Generate embeddings (dense)
    t9 = time.time()
    embeddings = vectorizer.encode(
        texts=texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        enable_dense=True,
        enable_sparse=False,  # Disable sparse for timing test
    )
    t10 = time.time()
    dense_vectors = embeddings.get("dense", [])
    print(f"    [3c] Dense encoding ({len(texts)} texts): {t10 - t9:.2f}s")

    # 3d. Generate sparse vectors
    t11 = time.time()
    embeddings_sparse = vectorizer.encode(
        texts=texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        enable_dense=False,
        enable_sparse=True,
    )
    t12 = time.time()
    sparse_vectors = embeddings_sparse.get("sparse", [])
    print(f"    [3d] Sparse encoding: {t12 - t11:.2f}s")

    # 3e. Prepare metadata
    metadata_list = []
    for ch in chunks:
        meta = ch.get("metadata", {})
        meta.update({"chunk_id": ch["id"], "namespace": "default"})
        metadata_list.append(meta)
    t13 = time.time()
    print(f"    [3e] Prepare metadata: {t13 - t12:.3f}s")

    # 3f. Insert to Milvus
    t14 = time.time()
    success = store.add_vectors(
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        texts=texts,
        metadata=metadata_list,
    )
    t15 = time.time()
    print(f"    [3f] Milvus insert: {t15 - t14:.2f}s")

    total_index = t15 - t6
    print(f"\n    [3] Total indexing: {total_index:.2f}s")
    print(f"        Dense: {t10 - t9:.2f}s ({100 * (t10 - t9) / total_index:.1f}%)")
    print(f"        Sparse: {t12 - t11:.2f}s ({100 * (t12 - t11) / total_index:.1f}%)")
    print(f"        Milvus: {t15 - t14:.2f}s ({100 * (t15 - t14) / total_index:.1f}%)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PDF load:        {(t3 - t2):.2f}s")
    print(f"Chunking:       {(t5 - t4):.2f}s")
    print(f"Indexing:       {total_index:.2f}s")
    print(f"  - Dense:      {(t10 - t9):.2f}s")
    print(f"  - Sparse:     {(t12 - t11):.2f}s")
    print(f"  - Milvus:     {(t15 - t14):.2f}s")
    print("=" * 60)


def test_query_detailed():
    """Test query with detailed timing"""
    print("\n" + "=" * 60)
    print("QUERY TEST - DETAILED TIMING")
    print("=" * 60)

    query = DEFAULT_QUERY

    # Get store
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

    # 1. Get vectorizer
    t0 = time.time()
    vectorizer = get_bge_vectorizer(
        model_name=settings.MODEL_NAME,
        cache_dir=settings.MODEL_CACHE_DIR,
        device=settings.MODEL_DEVICE,
        enable_dense=settings.MODEL_ENABLE_DENSE,
        enable_sparse=settings.MODEL_ENABLE_SPARSE,
    )
    t1 = time.time()
    print(f"\n[1] Get vectorizer: {t1 - t0:.3f}s")

    # 2. Generate query dense embedding
    t2 = time.time()
    dense_result = vectorizer.encode(
        texts=[query],
        batch_size=1,
        enable_dense=True,
        enable_sparse=False,
    )
    t3 = time.time()
    dense_vector = dense_result.get("dense", [[]])[0]
    print(f"[2] Query dense encoding: {t3 - t2:.2f}s")

    # 3. Generate query sparse embedding
    t4 = time.time()
    sparse_result = vectorizer.encode(
        texts=[query],
        batch_size=1,
        enable_dense=False,
        enable_sparse=True,
    )
    t5 = time.time()
    sparse_vector = sparse_result.get("sparse", [{}])[0]
    print(f"[3] Query sparse encoding: {t5 - t4:.2f}s")

    # 4. Hybrid search
    t6 = time.time()
    results = store.search_hybrid_native(
        query_dense=dense_vector,
        query_sparse=sparse_vector,
        top_k=3,
        namespace="default",
        ranker_type="rrf",
    )
    t7 = time.time()
    print(f"[4] Native hybrid search: {t7 - t6:.2f}s")

    print(f"\n[QUERY] '{query}'")
    print(f"Found {len(results)} results in {t7 - t1:.2f}s:")
    for i, r in enumerate(results, 1):
        text = r.get("text", "")[:80]
        score = r.get("score", 0)
        print(f"  {i}. {text}... (score: {score:.4f})")

    print("\n" + "=" * 60)
    print("QUERY SUMMARY")
    print("=" * 60)
    print(f"Vectorizer:      {(t1 - t0):.3f}s")
    print(f"Dense encode:   {(t3 - t2):.2f}s")
    print(f"Sparse encode:  {(t5 - t4):.2f}s")
    print(f"Milvus search:  {(t7 - t6):.2f}s")
    print(f"Total:          {(t7 - t1):.2f}s")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test PDF RAG")
    parser.add_argument("pdf", nargs="?", help="PDF file path")
    parser.add_argument("--query", "-q", help="Query string")
    args = parser.parse_args()

    if args.pdf:
        test_pdf_detailed(args.pdf)

    test_query_detailed()


if __name__ == "__main__":
    main()
