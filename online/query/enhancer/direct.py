"""Direct query strategy (no enhancement)."""
from online.query.enhancer.hyde import QueryEnhancer


class DirectEnhancer(QueryEnhancer):
    """No enhancement, use query directly."""

    def enhance(self, query: str) -> list[str]:
        """Return original query."""
        return [query]
