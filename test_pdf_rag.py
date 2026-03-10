"""Test PDF ingestion and query - with model caching simulation"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.offline_pipeline import (
    load_and_convert_files,
    load_and_chunk_texts,
    index_chunks,
)
from pipeline.online_pipeline import search_vectors
from core.embedding import get_bge_vectorizer, get_dimension
from stores.milvus_store import MilvusVectorStore, MilvusConnectionManager
from config.settings import settings


DEFAULT_QUERY = "A22赛题面部驱动模型的具体要求"


def test_pdf_twice(pdf_path: str):
    """Test PDF ingestion twice - simulate real scenario with model caching"""
    print("=" * 60)
    print("PDF INGESTION TEST (2 runs - model cached after first)")
    print("=" * 60)

    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        return

    # Get store and clear collection
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

    # Pre-initialize model (simulates real scenario where model is already loaded)
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
    print(f"    Model initialized in {t1 - t0:.2f}s")

    # Load PDF once
    print(f"\n[1] Loading PDF: {pdf_path}")
    t2 = time.time()
    docs = load_and_convert_files([pdf_path])
    t3 = time.time()
    print(f"    Loaded {len(docs)} document(s) in {t3 - t2:.2f}s")

    if not docs:
        print("    ERROR: No content extracted!")
        return

    # Chunk once
    print("\n[2] Chunking...")
    t4 = time.time()
    chunks = load_and_chunk_texts(
        texts=[doc["text"] for doc in docs],
        chunk_size=200,
        chunk_overlap=50,
        namespace="default",
        metadata_list=[doc["metadata"] for doc in docs],
    )
    t5 = time.time()
    print(f"    Generated {len(chunks)} chunks in {t5 - t4:.2f}s")

    # ===== First run (model already loaded, fresh collection) =====
    print("\n" + "=" * 40)
    print("FIRST RUN")
    print("=" * 40)
    store.clear_collection()
    t6 = time.time()
    indexed1 = index_chunks(chunks=chunks, namespace="default")
    t7 = time.time()
    print(f"    Indexed: {indexed1} chunks in {t7 - t6:.2f}s")

    # ===== Second run (model already loaded, fresh collection) =====
    print("\n" + "=" * 40)
    print("SECOND RUN (model cached)")
    print("=" * 40)
    store.clear_collection()
    t8 = time.time()
    indexed2 = index_chunks(chunks=chunks, namespace="default")
    t9 = time.time()
    print(f"    Indexed: {indexed2} chunks in {t9 - t8:.2f}s")

    if t9 - t8 < t7 - t6:
        print(f"Time saved:      {(t7 - t6) - (t9 - t8):.2f}s (model cached)")
    else:
        print(f"Note: Second run slower (Milvus duplicate insert overhead)")
    print("=" * 60)


def query_loop():
    """Interactive query loop"""
    print("=" * 60)
    print("QUERY MODE")
    print(f"Default query: {DEFAULT_QUERY}")
    print("=" * 60)

    while True:
        query = input("\nEnter query (or 'q' to quit): ").strip()

        if query.lower() == "q":
            break

        if not query:
            query = DEFAULT_QUERY
            print(f"Using default: {query}")

        print(f"\n[QUERY] {query}")
        t0 = time.time()
        results = search_vectors(
            query=query, top_k=3, namespace="default", search_type="hybrid"
        )
        t1 = time.time()

        print(f"    Found {len(results)} results in {t1 - t0:.2f}s:")
        for i, r in enumerate(results, 1):
            text = r.get("text", "")[:100]
            score = r.get("score", 0)
            print(f"    {i}. {text}...")
            print(f"       Score: {score:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test PDF RAG")
    parser.add_argument("pdf", nargs="?", help="PDF file path")
    parser.add_argument("--query", "-q", help="Query string (optional, skips ingest)")
    args = parser.parse_args()

    if args.pdf:
        test_pdf_twice(args.pdf)

    if args.query or not args.pdf:
        if args.query:
            t0 = time.time()
            results = search_vectors(
                query=args.query, top_k=3, namespace="default", search_type="dense"
            )
            t1 = time.time()
            print(f"\n[QUERY] {args.query}")
            print(f"Found {len(results)} results in {t1 - t0:.2f}s:")
            for i, r in enumerate(results, 1):
                text = r.get("text", "")
                score = r.get("score", 0)
                print(f"  {i}. {text}... (score: {score:.4f})")
        else:
            query_loop()


if __name__ == "__main__":
    main()
