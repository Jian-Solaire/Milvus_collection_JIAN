"""Unit tests for settings module."""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
from config.settings import Settings


class TestSettings:
    """Tests for Settings."""

    def test_default_values(self):
        """Test default values are set."""
        settings = Settings()

        assert settings.MILVUS_HOST == "localhost"
        assert settings.MILVUS_PORT == 19530
        assert settings.DEFAULT_CHUNK_SIZE == 800
        assert settings.DEFAULT_CHUNK_OVERLAP == 100
        assert settings.DEFAULT_NAMESPACE == "default"
        assert settings.RRF_K == 60

    def test_from_env_with_defaults(self):
        """Test loading from environment with defaults."""
        import os

        # Clear any existing env vars
        env_vars = [
            "MILVUS_HOST",
            "MILVUS_PORT",
            "RRF_K",
            "MODEL_CACHE_DIR",
            "ERROR_DETAIL_ENABLED",
        ]
        for var in env_vars:
            os.environ.pop(var, None)

        settings = Settings.from_env()

        assert settings.MILVUS_HOST == "localhost"
        assert settings.RRF_K == 60

    def test_from_env_override(self):
        """Test environment variable override."""
        import os

        os.environ["MILVUS_HOST"] = "192.168.1.100"
        os.environ["MILVUS_PORT"] = "19531"
        os.environ["RRF_K"] = "50"

        try:
            settings = Settings.from_env()

            assert settings.MILVUS_HOST == "192.168.1.100"
            assert settings.MILVUS_PORT == 19531
            assert settings.RRF_K == 50
        finally:
            # Cleanup
            os.environ.pop("MILVUS_HOST", None)
            os.environ.pop("MILVUS_PORT", None)
            os.environ.pop("RRF_K", None)

    def test_model_cache_dir_default(self):
        """Test model cache directory default."""
        settings = Settings()

        assert settings.MODEL_CACHE_DIR == "./models/bge-m3"

    def test_async_config(self):
        """Test async configuration."""
        settings = Settings()

        assert settings.ASYNC_ENABLED is True
        assert settings.ASYNC_BATCH_SIZE == 100

    def test_error_config(self):
        """Test error configuration."""
        settings = Settings()

        assert settings.ERROR_DETAIL_ENABLED is True
