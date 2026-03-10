"""Online pipeline: query -> retrieve -> generate."""

import logging
import os
import sys
from typing import List, Dict, Any, Optional

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedding import get_bge_vectorizer, get_dimension
from stores.milvus_store import MilvusVectorStore
from config.settings import settings

logger = logging.getLogger(__name__)


def embed_query(query: str) -> Dict[str, Any]:
    """Embed query text.

    Args:
        query: Query text

    Returns:
        Dict with 'dense' and 'sparse' vectors
    """
    vectorizer = get_bge_vectorizer(
        model_name=settings.MODEL_NAME,
        cache_dir=settings.MODEL_CACHE_DIR,
        device=settings.MODEL_DEVICE,
        enable_dense=settings.MODEL_ENABLE_DENSE,
        enable_sparse=settings.MODEL_ENABLE_SPARSE,
    )
    return vectorizer.encode_query(query)


def search_vectors(
    store: Optional[MilvusVectorStore] = None,
    query: str = "",
    top_k: int = 8,
    namespace: Optional[str] = None,
    score_threshold: Optional[float] = None,
    search_type: str = "hybrid",
) -> List[Dict]:
    """Search vectors in Milvus.

    Args:
        store: Milvus store instance
        query: Query text
        top_k: Number of results
        namespace: Namespace filter
        score_threshold: Score threshold
        search_type: Search type (dense, sparse, hybrid)

    Returns:
        List of search results
    """
    if not query:
        return []

    # Get store
    if store is None:
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

    # Embed query
    query_embeddings = embed_query(query)
    dense_vector = query_embeddings.get("dense", [])
    sparse_vector = query_embeddings.get("sparse", {})

    if not dense_vector and not sparse_vector:
        logger.error("Failed to generate query embeddings")
        return []

    # Ensure dense_vector is a list
    if dense_vector and isinstance(dense_vector[0], list):
        dense_vector = dense_vector[0]
    if sparse_vector and isinstance(sparse_vector, list):
        sparse_vector = sparse_vector[0] if sparse_vector else {}

    # Search based on type
    if search_type == "dense":
        results = store.search_dense(
            query_vector=dense_vector,
            top_k=top_k,
            namespace=namespace,
            score_threshold=score_threshold,
        )
    elif search_type == "sparse":
        results = store.search_sparse(
            query_sparse=sparse_vector,
            top_k=top_k,
            namespace=namespace,
            score_threshold=score_threshold,
        )
    else:  # hybrid
        results = store.search_hybrid_native(
            query_dense=dense_vector,
            query_sparse=sparse_vector,
            top_k=top_k,
            namespace=namespace,
            ranker_type="rrf",
            score_threshold=score_threshold,
        )

    return results


def search_vectors_expanded(
    store: Optional[MilvusVectorStore] = None,
    query: str = "",
    top_k: int = 8,
    namespace: Optional[str] = None,
    score_threshold: Optional[float] = None,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
    search_type: str = "hybrid",
) -> List[Dict]:
    """Search with query expansion.

    Args:
        store: Milvus store instance
        query: Query text
        top_k: Number of results
        namespace: Namespace filter
        score_threshold: Score threshold
        enable_mqe: Enable multi-query expansion
        mqe_expansions: Number of expansions
        enable_hyde: Enable hypothetical document embedding
        candidate_pool_multiplier: Candidate pool multiplier
        search_type: Search type

    Returns:
        List of search results
    """
    if not query:
        return []

    # Query expansion
    expansions: List[str] = [query]

    if enable_mqe and mqe_expansions > 0:
        expansions.extend(_prompt_mqe(query, mqe_expansions))

    if enable_hyde:
        hyde_text = _prompt_hyde(query)
        if hyde_text:
            expansions.append(hyde_text)

    # Unique and trim
    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq[: max(1, len(uniq))]

    # Distribute pool
    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))

    # Collect hits
    agg: Dict[str, Dict] = {}
    for q in expansions:
        hits = search_vectors(
            store=store,
            query=q,
            top_k=per,
            namespace=namespace,
            score_threshold=score_threshold,
            search_type=search_type,
        )
        for h in hits:
            chunk_id = h.get("chunk_id", str(h.get("id")))
            score = float(h.get("score", 0.0))
            if chunk_id not in agg or score > float(agg[chunk_id].get("score", 0.0)):
                agg[chunk_id] = h

    # Sort by score
    merged = list(agg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]


def _prompt_mqe(query: str, n: int) -> List[str]:
    """Generate expanded queries using LLM.

    Args:
        query: Original query
        n: Number of expansions

    Returns:
        List of expanded queries
    """
    try:
        from core.llm import HelloAgentsLLM

        llm = HelloAgentsLLM()
        prompt = [
            {
                "role": "system",
                "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。",
            },
            {
                "role": "user",
                "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。",
            },
        ]
        text = llm.invoke(prompt)
        lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
        outs = [ln for ln in lines if ln]
        return outs[:n] or [query]
    except Exception:
        return [query]


def _prompt_hyde(query: str) -> Optional[str]:
    """Generate hypothetical document using LLM.

    Args:
        query: User query

    Returns:
        Hypothetical document text
    """
    try:
        from core.llm import HelloAgentsLLM

        llm = HelloAgentsLLM()
        prompt = [
            {
                "role": "system",
                "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。",
            },
            {
                "role": "user",
                "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。",
            },
        ]
        return llm.invoke(prompt)
    except Exception:
        return None


class OnlinePipeline:
    """Online query processing pipeline."""

    def __init__(
        self,
        store: Optional[MilvusVectorStore] = None,
        namespace: str = "default",
    ):
        """Initialize online pipeline.

        Args:
            store: Milvus store instance
            namespace: Namespace
        """
        self.store = store
        self.namespace = namespace

    def search(
        self,
        query: str,
        top_k: int = 8,
        search_type: str = "hybrid",
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Search knowledge base.

        Args:
            query: Query text
            top_k: Number of results
            search_type: Search type
            enable_mqe: Enable MQE
            enable_hyde: Enable HyDE
            score_threshold: Score threshold

        Returns:
            List of search results
        """
        if enable_mqe or enable_hyde:
            return search_vectors_expanded(
                store=self.store,
                query=query,
                top_k=top_k,
                namespace=self.namespace,
                score_threshold=score_threshold,
                enable_mqe=enable_mqe,
                enable_hyde=enable_hyde,
                search_type=search_type,
            )

        return search_vectors(
            store=self.store,
            query=query,
            top_k=top_k,
            namespace=self.namespace,
            score_threshold=score_threshold,
            search_type=search_type,
        )
