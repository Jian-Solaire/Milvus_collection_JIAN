"""Milvus vector store implementation with dense and sparse vectors support."""

import logging
import os
import sys
import threading
from typing import Dict, List, Optional, Any, Union

import numpy as np

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

logger = logging.getLogger(__name__)

# Try to import pymilvus
try:
    from pymilvus import (
        connections,
        Collection,
        FieldSchema,
        CollectionSchema,
        DataType,
        utility,
        AnnSearchRequest,
        RRFRanker,
        WeightedRanker,
    )
    from pymilvus.exceptions import MilvusException

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    MILVUSException = Exception
    connections = None
    Collection = None
    AnnSearchRequest = None
    RRFRanker = None
    WeightedRanker = None


class MilvusConnectionManager:
    """Milvus connection manager - singleton pattern to prevent duplicate connections."""

    _instances: Dict[str, "MilvusVectorStore"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        collection_name: str = "rag_vectors",
        dense_dimension: int = 1024,
        metric_type: str = "IP",
        timeout: int = 30,
        **kwargs,
    ) -> "MilvusVectorStore":
        """Get or create Milvus instance (singleton pattern)."""
        key = f"{host}:{port}:{collection_name}"

        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    logger.debug(f"Creating new Milvus connection: {collection_name}")
                    cls._instances[key] = MilvusVectorStore(
                        host=host,
                        port=port,
                        user=user,
                        password=password,
                        collection_name=collection_name,
                        dense_dimension=dense_dimension,
                        metric_type=metric_type,
                        timeout=timeout,
                        **kwargs,
                    )
                else:
                    logger.debug(
                        f"Reusing existing Milvus connection: {collection_name}"
                    )
        else:
            logger.debug(f"Reusing existing Milvus connection: {collection_name}")

        return cls._instances[key]


class MilvusVectorStore:
    """Milvus vector database implementation with dense + sparse vectors."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        collection_name: str = "rag_vectors",
        dense_dimension: int = 1024,
        metric_type: str = "IP",
        timeout: int = 30,
        **kwargs,
    ):
        """Initialize Milvus vector store.

        Args:
            host: Milvus host
            port: Milvus port
            user: Milvus user (optional)
            password: Milvus password (optional)
            collection_name: Collection name
            dense_dimension: Dense vector dimension
            metric_type: Distance metric type (IP, COSINE, L2)
            timeout: Connection timeout
        """
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus not installed. Please run: pip install pymilvus>=2.6.11"
            )

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.dense_dimension = dense_dimension
        self.metric_type = metric_type.upper()
        self.timeout = timeout
        self.client = None
        self.collection = None
        self.alias = "default"  # Use fixed alias
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Milvus client and collection."""
        try:
            # Connect to Milvus
            try:
                connections.connect(
                    alias=self.alias,
                    host=self.host,
                    port=self.port,
                    user=self.user or "",
                    password=self.password or "",
                    timeout=self.timeout,
                )
                logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            except Exception as e:
                logger.warning(f"Connection might already exist: {e}")

            # Create or get collection
            self._ensure_collection()

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _ensure_collection(self) -> None:
        """Ensure collection exists, create if not."""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name, using=self.alias):
                logger.info(f"Using existing collection: {self.collection_name}")
                self.collection = Collection(self.collection_name, using=self.alias)
                self.collection.load()
            else:
                # Create new collection with schema
                self._create_collection()

            # Ensure indexes exist
            self._ensure_indexes()

        except Exception as e:
            logger.error(f"Collection initialization failed: {e}")
            raise

    def _create_collection(self) -> None:
        """Create new collection with schema."""
        # Define fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(
                name="text_dense", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dimension
            ),
            FieldSchema(name="text_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
            # Metadata fields
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="namespace", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(
                name="heading_path",
                dtype=DataType.VARCHAR,
                max_length=1024,
                nullable=True,
            ),
            FieldSchema(
                name="metadata_json",
                dtype=DataType.VARCHAR,
                max_length=4096,
                nullable=True,
            ),
        ]

        # Create schema
        schema = CollectionSchema(
            fields=fields,
            description="RAG vector collection with dense and sparse vectors",
        )

        # Create collection
        self.collection = Collection(
            name=self.collection_name, schema=schema, using=self.alias
        )
        logger.info(f"Created collection: {self.collection_name}")

    def _ensure_indexes(self) -> None:
        """Create indexes for dense and sparse vectors."""
        try:
            # Dense vector index (AUTOINDEX)
            dense_index_params = {
                "index_type": "AUTOINDEX",
                "metric_type": self.metric_type,
            }
            self.collection.create_index(
                field_name="text_dense",
                index_params=dense_index_params,
            )
            logger.info("Created dense vector index")

            # Sparse vector index
            sparse_index_params = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": self.metric_type,
            }
            self.collection.create_index(
                field_name="text_sparse",
                index_params=sparse_index_params,
            )
            logger.info("Created sparse vector index")

            # Load collection
            self.collection.load()

        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            raise

    def add_vectors(
        self,
        dense_vectors: List[List[float]],
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        texts: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> bool:
        """Add vectors to Milvus.

        Args:
            dense_vectors: Dense vectors
            sparse_vectors: Sparse vectors as dicts {term_idx: weight}
            texts: Original texts
            metadata: Metadata list
            ids: Optional IDs

        Returns:
            bool: Success or not
        """
        try:
            if not dense_vectors:
                logger.warning("No vectors to add")
                return False

            # Prepare data
            texts = texts or ["" for _ in dense_vectors]
            metadata = metadata or [{} for _ in dense_vectors]

            # Initialize sparse vectors if not provided
            if sparse_vectors is None:
                sparse_vectors = [{} for _ in dense_vectors]
            elif len(sparse_vectors) != len(dense_vectors):
                raise ValueError(
                    "dense_vectors and sparse_vectors must have same length"
                )

            # Prepare rows
            rows = []
            for i, (dense, sparse, text, meta) in enumerate(
                zip(dense_vectors, sparse_vectors, texts, metadata)
            ):
                row = {
                    "text": text[:10000] if text else "",  # Truncate if too long
                    "text_dense": dense,
                    "text_sparse": sparse,
                    "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
                    "document_id": meta.get("document_id", ""),
                    "namespace": meta.get("namespace", self.alias),
                    "source_path": meta.get("source_path", ""),
                    "heading_path": meta.get("heading_path"),
                    "metadata_json": str(meta)[:4096] if meta else None,
                }
                rows.append(row)

            # Insert data
            self.collection.insert(rows)
            self.collection.flush()

            logger.info(f"Successfully added {len(rows)} vectors to Milvus")
            return True

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search using dense vectors.

        Args:
            query_vector: Query dense vector
            top_k: Number of results
            namespace: Namespace filter
            score_threshold: Score threshold

        Returns:
            List of search results
        """
        try:
            # Build search parameters
            search_params = {
                "metric_type": self.metric_type,
                "params": {},
            }

            # Build filter expression
            filter_expr = self._build_filter_expr(namespace=namespace)

            # Search
            results = self.collection.search(
                data=[query_vector],
                anns_field="text_dense",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=[
                    "id",
                    "text",
                    "chunk_id",
                    "document_id",
                    "namespace",
                    "source_path",
                    "heading_path",
                    "metadata_json",
                ],
            )

            # Process results
            return self._process_search_results(results, score_threshold)

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    def search_sparse(
        self,
        query_sparse: Dict[int, float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search using sparse vectors.

        Args:
            query_sparse: Query sparse vector as dict {term_idx: weight}
            top_k: Number of results
            namespace: Namespace filter
            score_threshold: Score threshold

        Returns:
            List of search results
        """
        try:
            # Build search parameters
            search_params = {
                "metric_type": self.metric_type,
                "params": {},
            }

            # Build filter expression
            filter_expr = self._build_filter_expr(namespace=namespace)

            # Search
            results = self.collection.search(
                data=[query_sparse],
                anns_field="text_sparse",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=[
                    "id",
                    "text",
                    "chunk_id",
                    "document_id",
                    "namespace",
                    "source_path",
                    "heading_path",
                    "metadata_json",
                ],
            )

            # Process results
            return self._process_search_results(results, score_threshold)

        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []

    def search_hybrid(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        rrf_k: int = 60,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search using both dense and sparse vectors with RRF fusion.

        Args:
            query_dense: Query dense vector
            query_sparse: Query sparse vector
            top_k: Number of results
            namespace: Namespace filter
            rrf_k: RRF parameter
            score_threshold: Score threshold

        Returns:
            List of fused results
        """
        # Get more candidates for better fusion
        candidate_pool = max(top_k * 4, 20)

        # Search dense
        dense_results = self.search_dense(
            query_vector=query_dense,
            top_k=candidate_pool,
            namespace=namespace,
        )

        # Search sparse
        sparse_results = self.search_sparse(
            query_sparse=query_sparse,
            top_k=candidate_pool,
            namespace=namespace,
        )

        # RRF fusion
        fused = self._rrf_fusion(dense_results, sparse_results, rrf_k)

        # Apply threshold and limit
        if score_threshold is not None:
            fused = [r for r in fused if r.get("score", 0) >= score_threshold]

        return fused[:top_k]

    def search_hybrid_native(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        ranker_type: str = "rrf",
        weights: List[float] = [0.7, 0.3],
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search using Milvus native hybrid_search API.

        This method performs multi-way ANN search on the server side
        and fuses results using RRF or Weighted ranker.

        Args:
            query_dense: Query dense vector
            query_sparse: Query sparse vector (dict format)
            top_k: Number of results
            namespace: Namespace filter
            ranker_type: "rrf" or "weighted"
            weights: [dense_weight, sparse_weight] for weighted mode
            score_threshold: Score threshold

        Returns:
            List of fused results
        """
        try:
            # Prepare data formats
            dense_data = (
                [query_dense] if not isinstance(query_dense[0], list) else query_dense
            )
            sparse_data = [query_sparse]

            # Dense ANN search request
            req_dense = AnnSearchRequest(
                data=dense_data,
                anns_field="text_dense",
                param={"metric_type": self.metric_type, "params": {"nprobe": 10}},
                limit=top_k * 2,
            )

            # Sparse ANN search request
            req_sparse = AnnSearchRequest(
                data=sparse_data,
                anns_field="text_sparse",
                param={"metric_type": self.metric_type},
                limit=top_k * 2,
            )

            # Configure ranker
            if ranker_type.lower() == "weighted":
                ranker = WeightedRanker(*weights)
            else:
                ranker = RRFRanker(k=60)

            # Execute hybrid search (without expr filter for now)
            res = self.collection.hybrid_search(
                reqs=[req_dense, req_sparse],
                rerank=ranker,
                limit=top_k,
                output_fields=[
                    "text",
                    "chunk_id",
                    "document_id",
                    "namespace",
                    "source_path",
                    "heading_path",
                    "metadata_json",
                ],
            )

            # Apply namespace filter in Python if needed
            results = self._process_search_results(res)
            if namespace:
                results = [r for r in results if r.get("namespace") == namespace]

            # Apply threshold if needed
            if score_threshold is not None:
                results = [r for r in results if r.get("score", 0) >= score_threshold]

            return results

        except Exception as e:
            logger.error(f"Native hybrid search failed: {e}")
            return []

    def _rrf_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion algorithm.

        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results
            rrf_k: RRF parameter

        Returns:
            Fused results
        """
        # Score map: id -> {dense_score, sparse_score, rank info}
        scores: Dict[str, Dict[str, Any]] = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            chunk_id = result.get("chunk_id", str(result.get("id", "")))
            rrf_score = 1.0 / (rrf_k + rank)

            if chunk_id not in scores:
                scores[chunk_id] = {
                    "id": result.get("id"),
                    "text": result.get("text", ""),
                    "chunk_id": chunk_id,
                    "document_id": result.get("document_id"),
                    "namespace": result.get("namespace"),
                    "source_path": result.get("source_path"),
                    "heading_path": result.get("heading_path"),
                    "metadata": result.get("metadata", {}),
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                }
            scores[chunk_id]["dense_score"] = result.get("score", 0)
            scores[chunk_id]["dense_rank"] = rank

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            chunk_id = result.get("chunk_id", str(result.get("id", "")))

            if chunk_id not in scores:
                scores[chunk_id] = {
                    "id": result.get("id"),
                    "text": result.get("text", ""),
                    "chunk_id": chunk_id,
                    "document_id": result.get("document_id"),
                    "namespace": result.get("namespace"),
                    "source_path": result.get("source_path"),
                    "heading_path": result.get("heading_path"),
                    "metadata": result.get("metadata", {}),
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                }
            scores[chunk_id]["sparse_score"] = result.get("score", 0)
            scores[chunk_id]["sparse_rank"] = rank

        # Calculate RRF scores
        fused_results = []
        for chunk_id, data in scores.items():
            rrf_score = 0.0

            if data["dense_rank"] is not None:
                rrf_score += 1.0 / (rrf_k + data["dense_rank"])
            if data["sparse_rank"] is not None:
                rrf_score += 1.0 / (rrf_k + data["sparse_rank"])

            # Also include original scores
            data["score"] = rrf_score
            data["dense_distance"] = data.get("dense_score", 0)
            data["sparse_distance"] = data.get("sparse_score", 0)
            fused_results.append(data)

        # Sort by RRF score
        fused_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return fused_results

    def _build_filter_expr(self, namespace: Optional[str] = None) -> Optional[str]:
        """Build filter expression for search.

        Args:
            namespace: Namespace filter

        Returns:
            Filter expression string
        """
        conditions = []

        if namespace:
            conditions.append(f'namespace == "{namespace}"')

        if conditions:
            return " and ".join(conditions)
        return None

    def _process_search_results(
        self,
        results: Any,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Process Milvus search results.

        Args:
            results: Raw search results
            score_threshold: Score threshold

        Returns:
            Processed results
        """
        processed = []

        if not results:
            return processed

        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "score": hit.distance,  # Distance (for IP, higher is better)
                    "text": hit.entity.get("text", ""),
                    "chunk_id": hit.entity.get("chunk_id", ""),
                    "document_id": hit.entity.get("document_id", ""),
                    "namespace": hit.entity.get("namespace", ""),
                    "source_path": hit.entity.get("source_path", ""),
                    "heading_path": hit.entity.get("heading_path"),
                    "metadata": {},
                }

                # Parse metadata_json if present
                metadata_json = hit.entity.get("metadata_json")
                if metadata_json:
                    try:
                        result["metadata"] = (
                            eval(metadata_json)
                            if isinstance(metadata_json, str)
                            else metadata_json
                        )
                    except Exception:
                        pass

                processed.append(result)

        return processed

    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs.

        Args:
            ids: List of chunk IDs to delete

        Returns:
            bool: Success or not
        """
        try:
            # Build delete expression
            id_list = '", "'.join(ids)
            delete_expr = f'chunk_id in ["{id_list}"]'

            self.collection.delete(delete_expr)
            self.collection.flush()

            logger.info(f"Deleted {len(ids)} vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    def clear_collection(self) -> bool:
        """Clear the collection.

        Returns:
            bool: Success or not
        """
        try:
            # Drop and recreate collection
            utility.drop_collection(self.collection_name, using=self.alias)
            self._ensure_collection()

            logger.info(f"Cleared collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information.

        Returns:
            Dict: Collection info
        """
        try:
            info = {
                "name": self.collection_name,
                "dense_dimension": self.dense_dimension,
                "metric_type": self.metric_type,
            }

            # Get stats
            if self.collection:
                stats = self.collection.num_entities
                info["entities_count"] = stats

            return info

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dict: Collection stats
        """
        info = self.get_collection_info()
        info["store_type"] = "milvus"
        return info

    def health_check(self) -> bool:
        """Health check.

        Returns:
            bool: Service is healthy or not
        """
        try:
            # Try to get collection list
            utility.list_collections(using=self.alias)
            return True
        except Exception as e:
            logger.error(f"Milvus health check failed: {e}")
            return False

    def __del__(self):
        """Destructor."""
        pass  # Let Milvus connection pool handle cleanup
