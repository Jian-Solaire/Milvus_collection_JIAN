"""Vector search for similarity matching."""
from core.vector_store import VectorStore


class VectorSearch:
    """Search vectors in vector store."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """Search similar chunks."""
        pass
