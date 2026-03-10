"""MQE (Multi-Query Expansion) strategy."""
from online.query.enhancer.hyde import QueryEnhancer


class MQEEnhancer(QueryEnhancer):
    """Expand query into multiple sub-queries."""

    def enhance(self, query: str) -> list[str]:
        """Enhance query using MQE."""
        pass
