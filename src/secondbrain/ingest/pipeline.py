"""Ingest pipeline: URL -> metadata -> transcript -> chunks -> embeddings.

Each stage runs only if its output is missing, so calling ingest again on a
failed or transcript-less item resumes from where it stopped.
"""

import logging
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from secondbrain import items, vectors
from secondbrain.config import Settings
from secondbrain.embed import Embedder, mean_vector
from secondbrain.errors import IngestError, SecondBrainError, TranscriptUnavailable
from secondbrain.ingest import article, youtube
from secondbrain.ingest.chunking import chunk_segments
from secondbrain.ingest.transcribe import transcribe_audio
from secondbrain.ingest.urls import parse_url
from secondbrain.models import Transcript

logger = logging.getLogger(__name__)

EmbedderFactory = Callable[[], Embedder]


@dataclass(frozen=True)
class IngestResult:
    item_id: int
    status: str
    created: bool
    note: str | None = None


def ingest_url(
    conn: sqlite3.Connection,
    settings: Settings,
    get_embedder: EmbedderFactory,
    url: str,
    *,
    transcribe: bool = False,
) -> IngestResult:
    """Save a URL and run every ingest stage that is still missing for it."""
    source_type, source_id, canonical = parse_url(url)
    item_id, created = items.get_or_create(conn, source_type, source_id, canonical)
    if not created:
        logger.info("Item %d already exists for %s; resuming if incomplete", item_id, canonical)
    status = process_item(conn, settings, get_embedder, item_id, transcribe=transcribe)
    return IngestResult(item_id, status, created, items.get(conn, item_id)["error"])


def process_item(
    conn: sqlite3.Connection,
    settings: Settings,
    get_embedder: EmbedderFactory,
    item_id: int,
    *,
    transcribe: bool = False,
) -> str:
    """Run missing stages for one item. Returns the final ingest_status.

    On failure the item is marked 'failed' with the reason, and IngestError is raised.
    """
    item = items.get(conn, item_id)
    stage = "fetch"
    try:
        if items.load_segments(conn, item_id) is None:
            _fetch_stage(conn, settings, item, transcribe)
        if items.load_segments(conn, item_id) is None:
            return "no_transcript"
        stage = "chunk"
        if not items.load_chunks(conn, item_id):
            _chunk_stage(conn, settings, item_id)
        stage = "embed"
        if not vectors.has_item_vector(conn, item_id):
            _embed_stage(conn, get_embedder(), item_id)
    except Exception as exc:
        conn.rollback()
        items.set_status(conn, item_id, "failed", error=f"[{stage}] {exc}")
        conn.commit()
        if isinstance(exc, SecondBrainError):
            logger.error("Item %d failed at stage '%s': %s", item_id, stage, exc)
        else:
            logger.exception("Item %d failed at stage '%s' with an unexpected error", item_id, stage)
        raise IngestError(item_id, stage, str(exc)) from exc
    return str(items.get(conn, item_id)["ingest_status"])


def _fetch_stage(
    conn: sqlite3.Connection, settings: Settings, item: sqlite3.Row, transcribe: bool
) -> None:
    item_id = int(item["id"])
    if item["source_type"] == "article":
        with _timed(conn, item_id, "fetch_article"):
            meta, transcript = article.fetch(item["url"])
        items.save_metadata(conn, item_id, meta)
    else:
        with _timed(conn, item_id, "metadata"):
            meta = youtube.fetch_metadata(item["source_id"])
        items.save_metadata(conn, item_id, meta)
        conn.commit()  # keep the title even if the transcript step fails
        found = _youtube_transcript(conn, settings, item, transcribe)
        if found is None:
            return
        transcript = found
        if meta.duration_sec is None:
            guess = youtube.duration_from_segments(transcript.segments)
            conn.execute(
                "UPDATE items SET duration_sec = COALESCE(duration_sec, ?) WHERE id = ?",
                (guess, item_id),
            )
    items.save_transcript(conn, item_id, transcript)
    items.set_status(conn, item_id, "fetched")
    conn.commit()


def _youtube_transcript(
    conn: sqlite3.Connection, settings: Settings, item: sqlite3.Row, transcribe: bool
) -> Transcript | None:
    """Captions if available; else Whisper if asked; else mark no_transcript and return None."""
    item_id, video_id = int(item["id"]), str(item["source_id"])
    try:
        with _timed(conn, item_id, "captions"):
            return youtube.fetch_captions(video_id, settings.caption_language_list)
    except TranscriptUnavailable as exc:
        if not transcribe:
            logger.warning("Item %d has no transcript: %s", item_id, exc)
            items.set_status(conn, item_id, "no_transcript", error=str(exc))
            conn.commit()
            return None
    with _timed(conn, item_id, "audio_download"):
        audio_path = youtube.download_audio(video_id, settings.audio_dir)
    with _timed(conn, item_id, "whisper"):
        transcript = transcribe_audio(audio_path, settings)
    audio_path.unlink(missing_ok=True)  # kept on failure so a retry skips the download
    return transcript


def _chunk_stage(conn: sqlite3.Connection, settings: Settings, item_id: int) -> None:
    segments = items.load_segments(conn, item_id) or []
    with _timed(conn, item_id, "chunk"):
        chunks = chunk_segments(segments, settings.chunk_words, settings.chunk_overlap_words)
    if not chunks:
        raise SecondBrainError("The transcript is empty, so there is nothing to chunk.")
    items.replace_chunks(conn, item_id, chunks)
    items.set_status(conn, item_id, "chunked")
    conn.commit()


def _embed_stage(conn: sqlite3.Connection, embedder: Embedder, item_id: int) -> None:
    vectors.ensure_vector_tables(conn, embedder.model_name, embedder.dim)
    chunk_rows = items.load_chunks(conn, item_id)
    with _timed(conn, item_id, "embed"):
        matrix = embedder.embed_documents([row["text"] for row in chunk_rows])
    if matrix.shape != (len(chunk_rows), embedder.dim):
        raise SecondBrainError(
            f"Embedder returned shape {matrix.shape}, expected {(len(chunk_rows), embedder.dim)}."
        )
    vectors.set_chunk_vectors(
        conn, [(int(row["id"]), vec.tolist()) for row, vec in zip(chunk_rows, matrix)]
    )
    vectors.set_item_vector(conn, item_id, mean_vector(matrix).tolist())
    items.set_status(conn, item_id, "embedded")
    conn.commit()


@contextmanager
def _timed(conn: sqlite3.Connection, item_id: int, stage: str) -> Iterator[None]:
    """Record how long a successful stage took (table ingest_timings + log line)."""
    start = time.perf_counter()
    yield
    seconds = time.perf_counter() - start
    items.add_timing(conn, item_id, stage, seconds)
    logger.info("Item %d: stage '%s' took %.2fs", item_id, stage, seconds)
