"""Vector storage on sqlite-vec. The only module that contains sqlite-vec SQL.

Two virtual tables, both using cosine distance:
  vec_chunks  rowid = chunks.id
  vec_items   rowid = items.id
"""

import logging
import sqlite3
from collections.abc import Sequence

import numpy as np
import sqlite_vec

from secondbrain.errors import StorageError

logger = logging.getLogger(__name__)

_MODEL_KEY = "embedding_model"
_DIM_KEY = "embedding_dim"


def ensure_vector_tables(conn: sqlite3.Connection, model_name: str, dim: int) -> None:
    """Create the vector tables on first use; refuse to mix embedding models."""
    stored = dict(
        conn.execute(
            "SELECT key, value FROM meta WHERE key IN (?, ?)", (_MODEL_KEY, _DIM_KEY)
        ).fetchall()
    )
    if stored:
        if stored.get(_MODEL_KEY) != model_name or stored.get(_DIM_KEY) != str(dim):
            raise StorageError(
                f"The database holds vectors from '{stored.get(_MODEL_KEY)}' "
                f"(dim {stored.get(_DIM_KEY)}), but the configured model is '{model_name}' "
                f"(dim {dim}). Vectors from different models cannot be mixed. Set "
                "EMBEDDING_MODEL back, or start a fresh DATA_DIR and ingest again."
            )
        return
    logger.info("Creating vector tables for %s (dim %d)", model_name, dim)
    for table in ("vec_chunks", "vec_items"):
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
            f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
        )
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        [(_MODEL_KEY, model_name), (_DIM_KEY, str(dim))],
    )
    # Commit now: table creation is not undone by a later rollback, so the meta
    # rows must not be undone either, or the two would get out of step.
    conn.commit()


def tables_exist(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT 1 FROM meta WHERE key = ?", (_DIM_KEY,)).fetchone()
    return row is not None


def set_chunk_vectors(conn: sqlite3.Connection, rows: Sequence[tuple[int, Sequence[float]]]) -> None:
    """Store one vector per chunk id, replacing any existing vector."""
    _replace(conn, "vec_chunks", rows)


def set_item_vector(conn: sqlite3.Connection, item_id: int, vector: Sequence[float]) -> None:
    _replace(conn, "vec_items", [(item_id, vector)])


def has_item_vector(conn: sqlite3.Connection, item_id: int) -> bool:
    if not tables_exist(conn):
        return False
    row = conn.execute("SELECT 1 FROM vec_items WHERE rowid = ?", (item_id,)).fetchone()
    return row is not None


def delete_item(conn: sqlite3.Connection, item_id: int, chunk_ids: Sequence[int]) -> None:
    """Remove the vectors of one item and of its chunks."""
    if not tables_exist(conn):
        return
    conn.executemany("DELETE FROM vec_chunks WHERE rowid = ?", [(chunk_id,) for chunk_id in chunk_ids])
    conn.execute("DELETE FROM vec_items WHERE rowid = ?", (item_id,))


def all_item_vectors(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """Every item vector: (item ids, matrix with one row per id)."""
    if not tables_exist(conn):
        return [], np.zeros((0, 0), dtype=np.float32)
    rows = conn.execute("SELECT rowid, embedding FROM vec_items ORDER BY rowid").fetchall()
    ids = [int(row[0]) for row in rows]
    vectors = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
    return ids, np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)


def knn_chunks(
    conn: sqlite3.Connection, query: Sequence[float], k: int
) -> list[tuple[int, float]]:
    """Return (chunk_id, cosine_distance) for the k nearest chunks, nearest first."""
    rows = conn.execute(
        "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (sqlite_vec.serialize_float32(list(query)), k),
    ).fetchall()
    return [(int(row[0]), float(row[1])) for row in rows]


def _replace(
    conn: sqlite3.Connection, table: str, rows: Sequence[tuple[int, Sequence[float]]]
) -> None:
    conn.executemany(f"DELETE FROM {table} WHERE rowid = ?", [(row_id,) for row_id, _ in rows])
    conn.executemany(
        f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)",
        [(row_id, sqlite_vec.serialize_float32(list(vec))) for row_id, vec in rows],
    )
