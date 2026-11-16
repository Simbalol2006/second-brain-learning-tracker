"""Shared test helpers. All tests here run offline."""

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from secondbrain import db, items
from secondbrain.config import Settings
from secondbrain.embed import normalize
from secondbrain.llm.base import LLMResponse
from secondbrain.ingest import pipeline
from secondbrain.models import Metadata, Segment, Transcript


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

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FakeProvider:
    """Scripted LLM: returns the given answers in order and records every prompt."""

    name = "fake"
    model = "fake-model"

    def __init__(self, answers: list[str | Exception]) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if not self.answers:
            raise AssertionError("FakeProvider was called more often than scripted")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return LLMResponse(answer, input_tokens=100, output_tokens=20)


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


def add_text_item(
    conn: sqlite3.Connection,
    settings: Settings,
    embedder: FakeEmbedder,
    source_id: str,
    title: str,
    text: str,
    repeat: int = 30,
) -> int:
    """Save an item with the given text and run chunk + embed + link (no network)."""
    item_id, _ = items.get_or_create(conn, "youtube", source_id, f"https://www.youtube.com/watch?v={source_id}")
    items.save_metadata(conn, item_id, Metadata(title=title, duration_sec=600))
    segments = [Segment(text, start=i * 5.0, end=(i + 1) * 5.0) for i in range(repeat)]
    items.save_transcript(conn, item_id, Transcript(segments, "auto", "en"))
    conn.commit()
    pipeline.process_item(conn, settings, lambda: embedder, item_id)
    return item_id
