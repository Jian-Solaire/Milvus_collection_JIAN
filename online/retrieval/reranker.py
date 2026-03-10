"""Reranker for ranking retrieved results."""
from abc import ABC, abstractmethod


class Reranker(ABC):
    """Base reranker."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Rerank chunks by relevance."""
        pass
