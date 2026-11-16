"""Semantic search: query -> nearest chunks -> best chunk per item -> ranked items."""

import logging
import sqlite3
from dataclasses import dataclass

from secondbrain import vectors
from secondbrain.embed import Embedder
from secondbrain.errors import SecondBrainError

logger = logging.getLogger(__name__)

MAX_K = 4096  # sqlite-vec's upper limit for k in one KNN query


@dataclass(frozen=True)
class SearchHit:
    """One item in the results, represented by its best-matching passage."""

    item_id: int
    title: str
    url: str
    score: float  # cosine similarity of the best chunk, 1.0 = identical direction
    passage: str
    start_sec: float | None

    @property
    def link(self) -> str:
        """URL that opens the video at the matching passage (plain URL for articles)."""
        if self.start_sec is None or "youtube.com/watch" not in self.url:
            return self.url
        return f"{self.url}&t={int(self.start_sec)}s"


def search(
    conn: sqlite3.Connection, embedder: Embedder, query: str, *, limit: int = 5, chunk_k: int = 30
) -> list[SearchHit]:
    """Return up to `limit` items, best first. Each item appears once."""
    query = query.strip()
    if not query:
        raise SecondBrainError("The search query is empty.")
    if limit < 1:
        raise SecondBrainError("limit must be at least 1.")
    if not vectors.tables_exist(conn):
        return []  # nothing has been embedded yet
    vectors.ensure_vector_tables(conn, embedder.model_name, embedder.dim)  # refuses a changed model
    query_vector = embedder.embed_query(query).tolist()

    # Many of the nearest chunks can belong to one item, so widen k until
    # enough different items are found or every chunk has been looked at.
    k = min(max(chunk_k, limit), MAX_K)
    while True:
        nearest = vectors.knn_chunks(conn, query_vector, k)
        hits = _best_chunk_per_item(conn, nearest)
        if len(hits) >= limit or len(nearest) < k or k >= MAX_K:
            return hits[:limit]
        k = min(k * 2, MAX_K)


def _best_chunk_per_item(
    conn: sqlite3.Connection, nearest: list[tuple[int, float]]
) -> list[SearchHit]:
    """`nearest` is sorted nearest-first, so the first chunk seen per item is its best."""
    hits: dict[int, SearchHit] = {}
    for chunk_id, distance in nearest:
        row = conn.execute(
            """
            SELECT c.item_id, c.text, c.start_sec, i.title, i.url
            FROM chunks c JOIN items i ON i.id = c.item_id WHERE c.id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if row is None:  # a vector whose chunk was deleted; ignore it
            logger.warning("Vector %d has no chunk row; skipping", chunk_id)
            continue
        if row["item_id"] not in hits:
            hits[row["item_id"]] = SearchHit(
                item_id=int(row["item_id"]),
                title=row["title"] or row["url"],
                url=row["url"],
                score=1.0 - distance,
                passage=row["text"],
                start_sec=row["start_sec"],
            )
    return list(hits.values())
