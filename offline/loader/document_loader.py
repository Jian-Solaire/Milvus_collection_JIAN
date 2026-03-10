"""Document loader for various file formats."""
from models.document import Document


class DocumentLoader:
    """Load documents from various sources."""

    def load(self, file_path: str) -> Document:
        """Load document from file."""
        pass

    def load_batch(self, file_paths: list[str]) -> list[Document]:
        """Load multiple documents."""
        pass
