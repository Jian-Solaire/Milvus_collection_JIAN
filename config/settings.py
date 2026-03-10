"""Global settings for vector store, LLM, and embedding configuration."""

import os
from typing import Optional


class Settings:
    """Global application settings."""

    # Milvus Configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    MILVUS_SECURE: bool = False
    MILVUS_COLLECTION_NAME: str = "rag_vectors"

    # BGE-M3 Model Configuration
    MODEL_NAME: str = "BAAI/bge-m3"
    MODEL_CACHE_DIR: str = r"C:\Users\JAQ\.cache\huggingface\hub"
    MODEL_DEVICE: str = "cuda"
    MODEL_ENABLE_SPARSE: bool = True
    MODEL_ENABLE_DENSE: bool = True
    EMBEDDING_BATCH_SIZE: int = 32  # Optimal for GPU

    # Model Configuration
    MODEL_DIMENSION: Optional[int] = 1024
    DEFAULT_CHUNK_SIZE: int = 800
    DEFAULT_CHUNK_OVERLAP: int = 100
    DEFAULT_NAMESPACE: str = "default"
    RRF_K: int = 60

    # Async Configuration
    ASYNC_ENABLED: bool = True
    ASYNC_BATCH_SIZE: int = 100

    # Error Configuration
    ERROR_DETAIL_ENABLED: bool = True

    # Index Configuration
    DENSE_INDEX_TYPE: str = "AUTOINDEX"
    SPARSE_INDEX_TYPE: str = "SPARSE_INVERTED_INDEX"
    METRIC_TYPE: str = "IP"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        settings = cls()
        settings.MILVUS_HOST = os.getenv("MILVUS_HOST", settings.MILVUS_HOST)
        settings.MILVUS_PORT = int(os.getenv("MILVUS_PORT", str(settings.MILVUS_PORT)))
        settings.MILVUS_USER = os.getenv("MILVUS_USER") or None
        settings.MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD") or None
        settings.MILVUS_SECURE = os.getenv("MILVUS_SECURE", "false").lower() == "true"
        settings.MILVUS_COLLECTION_NAME = os.getenv(
            "MILVUS_COLLECTION_NAME", settings.MILVUS_COLLECTION_NAME
        )
        settings.MODEL_CACHE_DIR = os.getenv(
            "MODEL_CACHE_DIR", settings.MODEL_CACHE_DIR
        )
        settings.MODEL_DEVICE = os.getenv("MODEL_DEVICE", settings.MODEL_DEVICE)
        settings.RRF_K = int(os.getenv("RRF_K", str(settings.RRF_K)))
        settings.ERROR_DETAIL_ENABLED = (
            os.getenv("ERROR_DETAIL_ENABLED", "true").lower() == "true"
        )
        settings.ASYNC_ENABLED = os.getenv("ASYNC_ENABLED", "true").lower() == "true"
        return settings


settings = Settings.from_env()
