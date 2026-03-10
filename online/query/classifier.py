"""Query classifier for determining query type."""
from models.query import QueryRequest


class QueryClassifier:
    """Classify query as QA or search."""

    def classify(self, query: QueryRequest) -> str:
        """Return 'qa' or 'search'."""
        pass
