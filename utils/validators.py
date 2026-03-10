"""Input validation utilities."""
from typing import Any


def validate_query(query: str) -> bool:
    """Validate query input."""
    return bool(query and query.strip())
