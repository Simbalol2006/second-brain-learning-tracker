"""Text embeddings on CPU via fastembed (ONNX runtime, no PyTorch)."""

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from secondbrain.errors import EmbeddingError

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns texts into unit-length vectors."""

    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return one vector of shape (dim,) for a search query."""
        ...


class FastEmbedEmbedder:
    """Wraps a fastembed model. The model is downloaded to cache_dir on first use."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        from fastembed import TextEmbedding

        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading embedding model %s", model_name)
        try:
            self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
            self.dim = int(TextEmbedding.get_embedding_size(model_name))
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{model_name}'. The first run needs internet "
                "to download it; also check that EMBEDDING_MODEL is a model fastembed supports. "
                f"Original error: {exc}"
            ) from exc
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        try:
            vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed adds the query prefix for models that were trained with one.
        try:
            vector = np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Embedding the query failed: {exc}") from exc
        return normalize(vector)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length. Zero rows are left as zeros."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def mean_vector(vectors: np.ndarray) -> np.ndarray:
    """Unit-length mean of the rows. Used as the item-level vector."""
    return normalize(vectors.mean(axis=0))
