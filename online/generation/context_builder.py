"""Build context from retrieved chunks for LLM."""
from typing import Any


class ContextBuilder:
    """Build prompt context from retrieved chunks."""

    def build(self, query: str, chunks: list[dict]) -> str:
        """Build context string."""
        pass
