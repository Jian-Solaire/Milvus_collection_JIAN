"""Vector store abstraction interface."""
from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add(self, chunks: list[dict]) -> None:
        """Add chunks to vector store."""
        pass

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """Search similar chunks."""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete chunks by IDs."""
        pass
