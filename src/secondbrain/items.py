"""Read and write rows of the items table and its direct children."""

import json
import sqlite3
from dataclasses import asdict

from secondbrain.errors import StorageError
from secondbrain.models import Chunk, Metadata, Segment, Transcript
from secondbrain.schemas import Analysis


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


def ids_without_analysis(conn: sqlite3.Connection) -> list[int]:
    """Items that have a transcript but no stored analysis yet."""
    rows = conn.execute(
        """
        SELECT t.item_id FROM transcripts t
        LEFT JOIN analyses a ON a.item_id = t.item_id
        WHERE a.item_id IS NULL ORDER BY t.item_id
        """
    ).fetchall()
    return [int(row[0]) for row in rows]


def save_analysis(
    conn: sqlite3.Connection,
    item_id: int,
    analysis: Analysis,
    provider: str,
    model: str,
    prompt_version: str,
) -> None:
    """Store the analysis and replace the item's topic and skill tags."""
    conn.execute(
        """
        INSERT OR REPLACE INTO analyses
            (item_id, summary, key_points_json, difficulty, provider, model, prompt_version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item_id,
            analysis.summary,
            json.dumps(analysis.key_points, ensure_ascii=False),
            analysis.difficulty,
            provider,
            model,
            prompt_version,
        ),
    )
    conn.execute("DELETE FROM item_topics WHERE item_id = ?", (item_id,))
    for kind, names in (("topic", analysis.topics), ("skill", analysis.skills)):
        for name in names:
            conn.execute("INSERT OR IGNORE INTO topics (name) VALUES (?)", (name,))
            conn.execute(
                "INSERT OR IGNORE INTO item_topics (item_id, topic_id, kind) "
                "SELECT ?, id, ? FROM topics WHERE name = ?",
                (item_id, kind, name),
            )


def load_analysis(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM analyses WHERE item_id = ?", (item_id,)).fetchone()


def load_tags(conn: sqlite3.Connection, item_id: int, kind: str) -> list[str]:
    rows = conn.execute(
        "SELECT t.name FROM item_topics it JOIN topics t ON t.id = it.topic_id "
        "WHERE it.item_id = ? AND it.kind = ? ORDER BY t.name",
        (item_id, kind),
    ).fetchall()
    return [str(row[0]) for row in rows]


def tag_vocabulary(conn: sqlite3.Connection, kind: str, limit: int) -> list[str]:
    """Most-used tag names of one kind, so the LLM can reuse them."""
    rows = conn.execute(
        "SELECT t.name FROM item_topics it JOIN topics t ON t.id = it.topic_id "
        "WHERE it.kind = ? GROUP BY t.id ORDER BY count(*) DESC, t.name LIMIT ?",
        (kind, limit),
    ).fetchall()
    return [str(row[0]) for row in rows]


def add_llm_call(
    conn: sqlite3.Connection,
    item_id: int,
    purpose: str,
    provider: str,
    model: str,
    attempt: int,
    valid: bool,
    input_tokens: int | None,
    output_tokens: int | None,
    seconds: float,
) -> None:
    conn.execute(
        """
        INSERT INTO llm_calls
            (item_id, purpose, provider, model, attempt, valid, input_tokens, output_tokens, seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (item_id, purpose, provider, model, attempt, int(valid), input_tokens, output_tokens, seconds),
    )


def llm_usage(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row:
    """Totals over all LLM calls made for one item."""
    return conn.execute(
        """
        SELECT count(*) AS calls, coalesce(sum(valid), 0) AS valid_calls,
               coalesce(sum(input_tokens), 0) AS input_tokens,
               coalesce(sum(output_tokens), 0) AS output_tokens
        FROM llm_calls WHERE item_id = ?
        """,
        (item_id,),
    ).fetchone()


def load_timings(conn: sqlite3.Connection, item_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT stage, seconds, created_at FROM ingest_timings WHERE item_id = ? ORDER BY id",
        (item_id,),
    ).fetchall()
