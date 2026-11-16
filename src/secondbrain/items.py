"""Read and write rows of the items table and its direct children."""

import json
import sqlite3
from dataclasses import asdict

from secondbrain.errors import StorageError
from secondbrain.models import Chunk, Metadata, Segment, Transcript


def get_or_create(
    conn: sqlite3.Connection, source_type: str, source_id: str, url: str
) -> tuple[int, bool]:
    """Return (item_id, created). The same source is never stored twice."""
    row = conn.execute(
        "SELECT id FROM items WHERE source_type = ? AND source_id = ?", (source_type, source_id)
    ).fetchone()
    if row:
        return int(row["id"]), False
    cursor = conn.execute(
        "INSERT INTO items (source_type, source_id, url) VALUES (?, ?, ?)",
        (source_type, source_id, url),
    )
    conn.commit()
    return int(cursor.lastrowid), True  # type: ignore[arg-type]


def get(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        raise StorageError(f"No item with id {item_id}. Run 'sb list' to see saved items.")
    return row


def list_all(conn: sqlite3.Connection, status: str | None = None) -> list[sqlite3.Row]:
    if status:
        return conn.execute(
            "SELECT * FROM items WHERE ingest_status = ? ORDER BY id", (status,)
        ).fetchall()
    return conn.execute("SELECT * FROM items ORDER BY id").fetchall()


def set_status(
    conn: sqlite3.Connection, item_id: int, status: str, error: str | None = None
) -> None:
    conn.execute(
        "UPDATE items SET ingest_status = ?, error = ? WHERE id = ?", (status, error, item_id)
    )


def save_metadata(conn: sqlite3.Connection, item_id: int, meta: Metadata) -> None:
    """Fill metadata columns; a None value never overwrites an existing value."""
    conn.execute(
        """
        UPDATE items SET
            title = COALESCE(?, title),
            author = COALESCE(?, author),
            duration_sec = COALESCE(?, duration_sec),
            language = COALESCE(?, language),
            published_at = COALESCE(?, published_at)
        WHERE id = ?
        """,
        (meta.title, meta.author, meta.duration_sec, meta.language, meta.published_at, item_id),
    )


def save_transcript(conn: sqlite3.Connection, item_id: int, transcript: Transcript) -> None:
    payload = json.dumps([asdict(s) for s in transcript.segments], ensure_ascii=False)
    conn.execute(
        "INSERT OR REPLACE INTO transcripts (item_id, segments_json) VALUES (?, ?)",
        (item_id, payload),
    )
    conn.execute(
        "UPDATE items SET transcript_source = ?, language = COALESCE(?, language) WHERE id = ?",
        (transcript.source, transcript.language, item_id),
    )


def load_segments(conn: sqlite3.Connection, item_id: int) -> list[Segment] | None:
    row = conn.execute(
        "SELECT segments_json FROM transcripts WHERE item_id = ?", (item_id,)
    ).fetchone()
    if row is None:
        return None
    return [Segment(**s) for s in json.loads(row["segments_json"])]


def replace_chunks(conn: sqlite3.Connection, item_id: int, chunks: list[Chunk]) -> None:
    conn.execute("DELETE FROM chunks WHERE item_id = ?", (item_id,))
    conn.executemany(
        "INSERT INTO chunks (item_id, idx, text, start_sec, end_sec) VALUES (?, ?, ?, ?, ?)",
        [(item_id, c.idx, c.text, c.start, c.end) for c in chunks],
    )


def load_chunks(conn: sqlite3.Connection, item_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE item_id = ? ORDER BY idx", (item_id,)
    ).fetchall()


def add_timing(conn: sqlite3.Connection, item_id: int, stage: str, seconds: float) -> None:
    conn.execute(
        "INSERT INTO ingest_timings (item_id, stage, seconds) VALUES (?, ?, ?)",
        (item_id, stage, seconds),
    )


def load_timings(conn: sqlite3.Connection, item_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT stage, seconds, created_at FROM ingest_timings WHERE item_id = ? ORDER BY id",
        (item_id,),
    ).fetchall()
