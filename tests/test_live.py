"""Live checks. Need internet, so they are skipped by default.

Run on the laptop:
    uv run pytest -m live
    SB_LIVE_YOUTUBE_URL="https://www.youtube.com/watch?v=..." uv run pytest -m live
"""

import os
import sqlite3

import numpy as np
import pytest

from secondbrain import items, vectors
from secondbrain.config import Settings
from secondbrain.embed import FastEmbedEmbedder
from secondbrain.ingest import pipeline

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def real_embedder(tmp_path_factory: pytest.TempPathFactory) -> FastEmbedEmbedder:
    return FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path_factory.mktemp("models"))


def test_real_embedding_model(real_embedder: FastEmbedEmbedder) -> None:
    out = real_embedder.embed_documents(
        [
            "Gradient descent updates weights to reduce the loss.",
            "Backpropagation computes gradients for neural network training.",
            "The recipe needs two cups of flour and one egg.",
        ]
    )
    assert out.shape == (3, 384)
    assert np.linalg.norm(out, axis=1) == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)
    assert float(out[0] @ out[1]) > float(out[0] @ out[2])


def test_real_youtube_ingest(
    conn: sqlite3.Connection, settings: Settings, real_embedder: FastEmbedEmbedder
) -> None:
    url = os.environ.get("SB_LIVE_YOUTUBE_URL")
    if not url:
        pytest.skip("Set SB_LIVE_YOUTUBE_URL to a video that has English captions.")
    result = pipeline.ingest_url(conn, settings, lambda: real_embedder, url)
    assert result.status == "embedded", result.note
    item = items.get(conn, result.item_id)
    assert item["title"]
    assert item["duration_sec"] and item["duration_sec"] > 0
    assert len(items.load_chunks(conn, result.item_id)) >= 1
    assert vectors.has_item_vector(conn, result.item_id)
