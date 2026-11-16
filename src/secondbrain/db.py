"""SQLite connection and schema. The schema version lives in PRAGMA user_version."""

import logging
import sqlite3
from pathlib import Path

import sqlite_vec

from secondbrain.errors import StorageError

logger = logging.getLogger(__name__)

# One entry per schema version. Later phases append; never edit an old entry.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE items (
        id INTEGER PRIMARY KEY,
        source_type TEXT NOT NULL CHECK (source_type IN ('youtube', 'article')),
        source_id TEXT NOT NULL,
        url TEXT NOT NULL,
        title TEXT,
        author TEXT,
        duration_sec INTEGER,
        language TEXT,
        published_at TEXT,
        transcript_source TEXT NOT NULL DEFAULT 'none'
            CHECK (transcript_source IN ('manual', 'auto', 'whisper', 'article', 'none')),
        ingest_status TEXT NOT NULL DEFAULT 'new'
            CHECK (ingest_status IN
                ('new', 'fetched', 'no_transcript', 'chunked', 'embedded', 'analyzed', 'failed')),
        error TEXT,
        progress_status TEXT NOT NULL DEFAULT 'to_watch'
            CHECK (progress_status IN ('to_watch', 'watching', 'done')),
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        done_at TEXT,
        UNIQUE (source_type, source_id)
    );
    CREATE TABLE transcripts (
        item_id INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
        segments_json TEXT NOT NULL
    );
    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL,
        text TEXT NOT NULL,
        start_sec REAL,
        end_sec REAL,
        UNIQUE (item_id, idx)
    );
    CREATE TABLE ingest_timings (
        id INTEGER PRIMARY KEY,
        item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        seconds REAL NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
    # 2 (Phase 2): LLM analysis, topic/skill tags, and a log of every LLM call.
    """
    CREATE TABLE analyses (
        item_id INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
        summary TEXT NOT NULL,
        key_points_json TEXT NOT NULL,
        difficulty TEXT NOT NULL CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE topics (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
    );
    CREATE TABLE item_topics (
        item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (kind IN ('topic', 'skill')),
        PRIMARY KEY (item_id, topic_id, kind)
    );
    CREATE TABLE llm_calls (
        id INTEGER PRIMARY KEY,
        item_id INTEGER REFERENCES items(id) ON DELETE SET NULL,
        purpose TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
        input_tokens INTEGER,
        output_tokens INTEGER,
        seconds REAL NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    # 3 (Phase 3): links between related items. Only pairs above the threshold are stored.
    """
    CREATE TABLE links (
        item_a INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        item_b INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        embed_sim REAL NOT NULL,
        tag_jaccard REAL NOT NULL,
        score REAL NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (item_a, item_b),
        CHECK (item_a < item_b)
    );
    """,
    # 4 (Phase 4): the learner profile. Progress itself lives in items.progress_status.
    """
    CREATE TABLE profile_skills (
        topic_id INTEGER PRIMARY KEY REFERENCES topics(id) ON DELETE CASCADE,
        level INTEGER NOT NULL CHECK (level BETWEEN 0 AND 3),
        source TEXT NOT NULL CHECK (source IN ('manual', 'derived')),
        evidence_count INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE profile_goals (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
]


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database, load sqlite-vec and bring the schema up to date."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _load_vector_extension(conn)
    _migrate(conn)
    return conn


def _load_vector_extension(conn: sqlite3.Connection) -> None:
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError) as exc:
        raise StorageError(
            "Could not load the sqlite-vec extension. This Python build's sqlite3 module "
            "may not allow loadable extensions. Use the uv-managed Python from the README "
            f"setup steps. Original error: {exc}"
        ) from exc


def _migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > len(MIGRATIONS):
        raise StorageError(
            f"Database schema version {current} is newer than this code supports "
            f"({len(MIGRATIONS)}). Update the code or use a different DATA_DIR."
        )
    for version in range(current, len(MIGRATIONS)):
        logger.info("Applying schema migration %d", version + 1)
        conn.executescript(MIGRATIONS[version])
        conn.execute(f"PRAGMA user_version = {version + 1}")
        conn.commit()
