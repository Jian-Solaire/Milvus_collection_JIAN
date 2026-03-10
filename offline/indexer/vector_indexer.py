"""Vector indexer for storing chunks in vector database."""
from core.vector_store import VectorStore
from models.chunk import Chunk


class VectorIndexer:
    """Index chunks to vector store."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def index(self, chunks: list[Chunk]) -> None:
        """Index chunks to vector store."""
        pass
