"""End-to-end pipeline tests with fake fetchers and a fake embedder."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeEmbedder, make_segments

from secondbrain import items, vectors
from secondbrain.config import Settings
from secondbrain.errors import FetchError, IngestError, InvalidUrlError, TranscriptUnavailable
from secondbrain.ingest import pipeline
from secondbrain.models import Metadata, Segment, Transcript

URL = "https://youtu.be/dQw4w9WgXcQ"
ARTICLE_URL = "https://example.com/post"
META = Metadata(title="A talk", author="A channel", duration_sec=600, published_at="2026-01-15")


def patch_youtube(
    monkeypatch: pytest.MonkeyPatch,
    captions: Transcript | Exception,
    meta: Metadata | Exception = META,
) -> dict[str, int]:
    """Replace the YouTube fetchers. Returns a call counter."""
    calls = {"metadata": 0, "captions": 0}

    def fetch_metadata(video_id: str) -> Metadata:
        calls["metadata"] += 1
        if isinstance(meta, Exception):
            raise meta
        return meta

    def fetch_captions(video_id: str, languages: list[str]) -> Transcript:
        calls["captions"] += 1
        if isinstance(captions, Exception):
            raise captions
        return captions

    monkeypatch.setattr(pipeline.youtube, "fetch_metadata", fetch_metadata)
    monkeypatch.setattr(pipeline.youtube, "fetch_captions", fetch_captions)
    return calls


def auto_captions(n: int = 100) -> Transcript:
    return Transcript(make_segments(n), source="auto", language="en")


def run(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, url: str = URL, **kw: Any) -> pipeline.IngestResult:
    return pipeline.ingest_url(conn, settings, lambda: embedder, url, **kw)


def test_youtube_happy_path(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder
) -> None:
    patch_youtube(monkeypatch, auto_captions())
    result = run(conn, settings, embedder)

    assert (result.status, result.created, result.note) == ("embedded", True, None)
    item = items.get(conn, result.item_id)
    assert item["title"] == "A talk"
    assert item["duration_sec"] == 600
    assert item["transcript_source"] == "auto"
    assert item["language"] == "en"
    assert item["progress_status"] == "to_watch"
    assert item["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    chunks = items.load_chunks(conn, result.item_id)
    assert len(chunks) > 1
    assert chunks[0]["start_sec"] == 0.0
    assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == len(chunks)
    assert vectors.has_item_vector(conn, result.item_id)
    assert len(items.load_segments(conn, result.item_id) or []) == 100

    stages = [t["stage"] for t in items.load_timings(conn, result.item_id)]
    assert stages == ["metadata", "captions", "chunk", "embed"]


def test_stored_vectors_are_searchable(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder
) -> None:
    patch_youtube(monkeypatch, auto_captions())
    result = run(conn, settings, embedder)
    target = items.load_chunks(conn, result.item_id)[1]
    query = embedder.embed_documents([target["text"]])[0]
    nearest_id, distance = vectors.knn_chunks(conn, query.tolist(), k=1)[0]
    assert nearest_id == target["id"]
    assert distance == pytest.approx(0.0, abs=1e-5)


def test_same_url_is_not_ingested_twice(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder
) -> None:
    calls = patch_youtube(monkeypatch, auto_captions())
    first = run(conn, settings, embedder)
    second = run(conn, settings, embedder, url="https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s")
    assert second.item_id == first.item_id
    assert (second.status, second.created) == ("embedded", False)
    assert calls == {"metadata": 1, "captions": 1}
    assert embedder.calls == 1
    assert conn.execute("SELECT count(*) FROM items").fetchone()[0] == 1


def test_no_captions_marks_item_and_keeps_metadata(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder
) -> None:
    patch_youtube(monkeypatch, TranscriptUnavailable("Captions are disabled for this video."))
    result = run(conn, settings, embedder)
    assert result.status == "no_transcript"
    assert "disabled" in (result.note or "")
    item = items.get(conn, result.item_id)
    assert item["title"] == "A talk"
    assert item["transcript_source"] == "none"
    assert items.load_chunks(conn, result.item_id) == []
    assert embedder.calls == 0


def test_whisper_fills_in_when_asked(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, tmp_path: Path
) -> None:
    patch_youtube(monkeypatch, TranscriptUnavailable("none"))
    audio = tmp_path / "audio.webm"
    audio.write_bytes(b"x")
    monkeypatch.setattr(pipeline.youtube, "download_audio", lambda video_id, out_dir: audio)
    monkeypatch.setattr(
        pipeline, "transcribe_audio", lambda path, s: Transcript(make_segments(40), "whisper", "en")
    )

    first = run(conn, settings, embedder)
    assert first.status == "no_transcript"

    status = pipeline.process_item(conn, settings, lambda: embedder, first.item_id, transcribe=True)
    assert status == "embedded"
    item = items.get(conn, first.item_id)
    assert item["transcript_source"] == "whisper"
    assert item["error"] is None
    assert not audio.exists()  # audio is removed after a successful transcription
    stages = [t["stage"] for t in items.load_timings(conn, first.item_id)]
    assert stages[-4:] == ["audio_download", "whisper", "chunk", "embed"]


def test_fetch_failure_is_recorded_and_raised(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder
) -> None:
    patch_youtube(monkeypatch, FetchError("YouTube blocked the caption request"))
    with pytest.raises(IngestError, match="stage 'fetch'.*blocked") as excinfo:
        run(conn, settings, embedder)
    item = items.get(conn, excinfo.value.item_id)
    assert item["ingest_status"] == "failed"
    assert item["error"] == "[fetch] YouTube blocked the caption request"
    assert item["title"] == "A talk"  # metadata fetched before the failure is kept


def test_failed_item_resumes_without_refetching(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings
) -> None:
    calls = patch_youtube(monkeypatch, auto_captions())

    class BrokenEmbedder(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> Any:
            raise RuntimeError("out of memory")

    with pytest.raises(IngestError, match="stage 'embed'.*out of memory") as excinfo:
        run(conn, settings, BrokenEmbedder())
    item_id = excinfo.value.item_id
    assert items.get(conn, item_id)["ingest_status"] == "failed"
    assert len(items.load_chunks(conn, item_id)) > 1  # earlier stages were kept

    result = run(conn, settings, FakeEmbedder())
    assert (result.item_id, result.status, result.note) == (item_id, "embedded", None)
    assert calls == {"metadata": 1, "captions": 1}


def test_empty_transcript_fails_at_chunk_stage(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder
) -> None:
    patch_youtube(monkeypatch, Transcript([Segment("  ", 0, 1)], "auto", "en"))
    with pytest.raises(IngestError, match="stage 'chunk'.*empty"):
        run(conn, settings, embedder)


def test_duration_is_guessed_when_metadata_lacks_it(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder
) -> None:
    patch_youtube(monkeypatch, auto_captions(100), meta=Metadata(title="T"))
    result = run(conn, settings, embedder)
    assert items.get(conn, result.item_id)["duration_sec"] == 300  # 100 snippets of 3 s


def test_article_happy_path(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder
) -> None:
    paragraphs = [Segment(" ".join(f"a{i}_{j}" for j in range(60))) for i in range(10)]
    doc = (Metadata(title="Post", author="Me", published_at="2026-03-04"), Transcript(paragraphs, "article"))
    monkeypatch.setattr(pipeline.article, "fetch", lambda url: doc)
    result = run(conn, settings, embedder, url=ARTICLE_URL)
    assert result.status == "embedded"
    item = items.get(conn, result.item_id)
    assert (item["source_type"], item["title"], item["transcript_source"]) == ("article", "Post", "article")
    chunks = items.load_chunks(conn, result.item_id)
    assert len(chunks) > 1
    assert chunks[0]["start_sec"] is None
    assert [t["stage"] for t in items.load_timings(conn, result.item_id)] == ["fetch_article", "chunk", "embed"]


def test_invalid_url_creates_no_item(
    conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder
) -> None:
    with pytest.raises(InvalidUrlError):
        run(conn, settings, embedder, url="https://www.youtube.com/playlist?list=PLabc")
    assert conn.execute("SELECT count(*) FROM items").fetchone()[0] == 0


def test_changing_embedding_model_is_refused(
    monkeypatch: pytest.MonkeyPatch, conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder
) -> None:
    patch_youtube(monkeypatch, auto_captions())
    run(conn, settings, embedder)
    other = FakeEmbedder()
    other.model_name = "another-model"
    with pytest.raises(IngestError, match="cannot be mixed"):
        run(conn, settings, other, url="https://youtu.be/aaaaaaaaaaa")
