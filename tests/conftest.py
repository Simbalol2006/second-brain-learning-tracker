"""Shared test helpers. All tests here run offline."""

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from secondbrain import db
from secondbrain.config import Settings
from secondbrain.embed import normalize
from secondbrain.models import Segment


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder. No model download."""

    model_name = "fake-embedder"
    dim = 64

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                digest = hashlib.md5(word.encode()).digest()
                out[i, digest[0] % self.dim] += 1.0
        return normalize(out)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.chdir(tmp_path)  # no stray .env is picked up
    return Settings(data_dir=tmp_path / "data", _env_file=None)


@pytest.fixture
def conn(settings: Settings) -> Iterator[sqlite3.Connection]:
    connection = db.connect(settings.db_path)
    yield connection
    connection.close()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_segments(n: int, words_each: int = 8, seconds_each: float = 3.0) -> list[Segment]:
    """Synthetic caption snippets: n snippets of words_each words."""
    return [
        Segment(
            text=" ".join(f"w{i}_{j}" for j in range(words_each)),
            start=i * seconds_each,
            end=(i + 1) * seconds_each,
        )
        for i in range(n)
    ]
