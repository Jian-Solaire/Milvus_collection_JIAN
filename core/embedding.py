"""BGE-M3 embedding model with dense and sparse vector support."""

import os
import sys
import logging
import re
from typing import List, Dict, Any, Union

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)

_bge_model = None
_model_dimensions = None


class BGEVectorizer:
    """BGE-M3 model for generating dense and sparse vectors."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str = None,
        device: str = "cuda",
        enable_dense: bool = True,
        enable_sparse: bool = True,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or settings.MODEL_CACHE_DIR
        self.device = device
        self.enable_dense = enable_dense
        self.enable_sparse = enable_sparse
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        global _bge_model, _model_dimensions

        if _bge_model is not None:
            self.model = _bge_model
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading BGE-M3 from: {self.cache_dir}")
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device,
            )

            _model_dimensions = self.model.get_sentence_embedding_dimension()
            logger.info(f"BGE-M3 loaded. Dim: {_model_dimensions}")

            _bge_model = self.model

        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise

    @property
    def dense_dimension(self) -> int:
        global _model_dimensions
        if _model_dimensions is None:
            self._load_model()
        return _model_dimensions or 1024

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        enable_dense: bool = True,
        enable_sparse: bool = True,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(texts, str):
            texts = [texts]

        enable_dense = enable_dense and self.enable_dense
        enable_sparse = enable_sparse and self.enable_sparse

        result: Dict[str, Any] = {}

        if enable_dense:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
            )
            if len(embeddings.shape) == 2:
                result["dense"] = embeddings.tolist()
            else:
                result["dense"] = [embeddings.tolist()]

        if enable_sparse:
            result["sparse"] = self._generate_sparse_vectors(texts)

        return result

    def _generate_sparse_vectors(self, texts: List[str]) -> List:
        """Generate sparse vectors using scipy sparse matrices."""
        import scipy.sparse as sp
        import numpy as np

        sparse_vectors = []

        # Build vocabulary across all texts
        all_tokens = []
        for text in texts:
            tokens = re.findall(r"\b\w+\b", text.lower())
            tokens = [t for t in tokens if len(t) > 1]
            all_tokens.append(tokens)

        # Build vocabulary
        vocab = {}
        for tokens in all_tokens:
            for token in set(tokens):
                if token not in vocab:
                    vocab[token] = len(vocab)

        vocab_size = len(vocab)
        if vocab_size == 0:
            # Return empty sparse matrices
            for _ in texts:
                sparse_vectors.append(sp.csr_matrix((1, 1)))
            return sparse_vectors

        # Create sparse matrices for each text
        for tokens in all_tokens:
            if not tokens:
                sparse_vectors.append(sp.csr_matrix((1, vocab_size)))
                continue

            # Term frequency
            indices = []
            values = []
            for token in set(tokens):
                if token in vocab:
                    indices.append(vocab[token])
                    values.append(tokens.count(token))

            # Normalize
            if values:
                max_freq = max(values)
                values = [v / max_freq for v in values]

            # Create sparse row vector
            sparse_vec = sp.csr_matrix(
                (values, ([0] * len(indices), indices)), shape=(1, vocab_size)
            )
            sparse_vectors.append(sparse_vec)

        return sparse_vectors

    def encode_query(self, query: str) -> Dict[str, Any]:
        return self.encode([query])


def get_bge_vectorizer(**kwargs) -> BGEVectorizer:
    global _bge_model

    if _bge_model is None:
        _bge_model = BGEVectorizer(**kwargs)

    return _bge_model


def get_dimension(default: int = 1024) -> int:
    global _model_dimensions
    if _model_dimensions is not None:
        return _model_dimensions
    try:
        v = get_bge_vectorizer()
        return v.dense_dimension
    except Exception:
        return default


class EmbeddingModel(VectorStore):
    def __init__(self):
        self.vectorizer = get_bge_vectorizer()

    def embed_text(self, text: str) -> list[float]:
        result = self.vectorizer.encode_query(text)
        dense = result.get("dense", [])
        if dense and isinstance(dense[0], list):
            return dense[0]
        return dense

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self.vectorizer.encode(texts)
        return result.get("dense", [])


def get_text_embedder():
    return EmbeddingModel()
