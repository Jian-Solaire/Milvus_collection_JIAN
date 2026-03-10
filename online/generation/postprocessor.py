"""Post-process generated response."""
from models.response import QueryResponse


class PostProcessor:
    """Validate and format generated response."""

    def process(self, response: QueryResponse) -> QueryResponse:
        """Process and validate response."""
        pass
