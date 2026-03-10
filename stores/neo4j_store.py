"""Neo4j vector store implementation (for graph relationships)."""
from core.vector_store import VectorStore


class Neo4jStore(VectorStore):
    """Neo4j vector database implementation."""

    def add(self, chunks: list[dict]) -> None:
        """Add chunks to Neo4j."""
        pass

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """Search in Neo4j."""
        pass

    def delete(self, ids: list[str]) -> None:
        """Delete from Neo4j."""
        pass
