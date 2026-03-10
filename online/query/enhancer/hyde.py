"""HyDE (Hypothetical Document Embeddings) strategy."""
from abc import ABC, abstractmethod


class QueryEnhancer(ABC):
    """Base query enhancer."""

    @abstractmethod
    def enhance(self, query: str) -> list[str]:
        """Return enhanced queries."""
        pass


class HyDEEnhancer(QueryEnhancer):
    """Generate hypothetical document and use it for retrieval."""

    def enhance(self, query: str) -> list[str]:
        """Enhance query using HyDE."""
        pass
