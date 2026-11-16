"""Live checks. Need internet, so they are skipped by default.

Run on the laptop:
    uv run pytest -m live
    SB_LIVE_YOUTUBE_URL="https://www.youtube.com/watch?v=..." uv run pytest -m live

The LLM check uses the provider configured in the project's .env (it reads the real
.env on purpose) and makes one small paid call when LLM_PROVIDER=anthropic.
"""

import os
import sqlite3

import numpy as np
import pytest

from secondbrain import analyze, items, vectors
from secondbrain.config import Settings
from secondbrain.embed import FastEmbedEmbedder
from secondbrain.errors import LLMError
from secondbrain.ingest import pipeline
from secondbrain.llm import build_provider
from secondbrain.models import Segment, Transcript

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


LIVE_TEXT = (
    "Gradient descent is an optimization method used to train machine learning models. "
    "It starts from random weights and repeatedly moves them a small step in the direction "
    "that reduces the loss. The size of the step is called the learning rate. If the learning "
    "rate is too large the training can diverge, and if it is too small training is slow. "
    "Stochastic gradient descent estimates the gradient from a small batch of examples, "
    "which makes each step much cheaper."
)


def test_real_llm_analysis(conn: sqlite3.Connection) -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real = Settings(_env_file=os.path.join(project_root, ".env"))
    try:
        provider = build_provider(real)
    except LLMError as exc:
        pytest.skip(f"LLM not configured: {exc}")
    item_id, _ = items.get_or_create(conn, "article", "live-llm", "https://example.com/live")
    items.save_transcript(conn, item_id, Transcript([Segment(LIVE_TEXT)], "article", "en"))
    conn.commit()

    result = analyze.analyze_item(conn, real, provider, item_id)
    assert result.summary and len(result.key_points) >= 2 and result.topics
    usage = items.llm_usage(conn, item_id)
    print(f"\nprovider={provider.name} model={provider.model} calls={usage['calls']} "
          f"valid={usage['valid_calls']} tokens={usage['input_tokens']}/{usage['output_tokens']}")
    print(result.model_dump_json(indent=2))
