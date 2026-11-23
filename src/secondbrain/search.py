"""Search over everything saved.

Three modes, chosen with SEARCH_MODE (or per call):
  dense   query embedding -> nearest chunks (sqlite-vec, cosine)
  bm25    query words     -> best keyword-matching chunks (SQLite FTS5, BM25)
  hybrid  both lists, merged with reciprocal rank fusion (RRF)

Every mode works on chunks and then keeps the best chunk per item, so each item appears once.
"""

import logging
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Literal

from secondbrain import vectors
from secondbrain.embed import Embedder
from secondbrain.errors import SecondBrainError

logger = logging.getLogger(__name__)

SearchMode = Literal["dense", "bm25", "hybrid"]
SEARCH_MODES: tuple[str, ...] = ("dense", "bm25", "hybrid")

MAX_K = 4096  # sqlite-vec's upper limit for k in one KNN query
DEFAULT_RRF_K = 60  # the constant from Cormack et al. (2009); larger = flatter weighting
DEFAULT_DEPTH = 50  # how many items each ranker contributes to the fusion


def _words(text: str) -> list[str]:
    """Split text into words the same way the FTS5 index does: runs of letters, digits and
    combining marks. Python's \\w leaves out combining marks, which would cut Telugu or Hindi
    words into single letters, so the character categories are checked directly."""
    words, current = [], []
    for char in text:
        if unicodedata.category(char)[0] in "LNM" or unicodedata.category(char) == "Co":
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


@dataclass(frozen=True)
class SearchHit:
    """One item in the results, represented by its best-matching passage.

    `score` depends on the mode: cosine similarity (dense), BM25 relevance with higher = better
    (bm25), or the RRF score (hybrid). Scores are only comparable within one result list.
    """

    item_id: int
    title: str
    url: str
    score: float
    passage: str
    start_sec: float | None
    dense_rank: int | None = None  # 1-based rank of this item in the dense list, if it was there
    bm25_rank: int | None = None  # 1-based rank of this item in the BM25 list, if it was there

    @property
    def link(self) -> str:
        """URL that opens the video at the matching passage (plain URL for articles)."""
        if self.start_sec is None or "youtube.com/watch" not in self.url:
            return self.url
        return f"{self.url}&t={int(self.start_sec)}s"


def search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 5,
    chunk_k: int = 30,
    mode: SearchMode = "hybrid",
    rrf_k: int = DEFAULT_RRF_K,
    depth: int = DEFAULT_DEPTH,
) -> list[SearchHit]:
    """Return up to `limit` items, best first. Each item appears once."""
    query = query.strip()
    if not query:
        raise SecondBrainError("The search query is empty.")
    if limit < 1:
        raise SecondBrainError("limit must be at least 1.")
    if mode not in SEARCH_MODES:
        raise SecondBrainError(f"Unknown search mode '{mode}'. Use one of: {', '.join(SEARCH_MODES)}.")
    if rrf_k < 1:
        raise SecondBrainError("RRF k must be at least 1.")

    if mode == "dense":
        return dense_items(conn, embedder, query, limit=limit, chunk_k=chunk_k)
    if mode == "bm25":
        return bm25_items(conn, query, limit=limit)

    pool = max(limit, depth)
    dense = dense_items(conn, embedder, query, limit=pool, chunk_k=chunk_k)
    keyword = bm25_items(conn, query, limit=pool)
    return fuse([dense, keyword], k=rrf_k)[:limit]


# ---------- dense ----------


def dense_items(
    conn: sqlite3.Connection, embedder: Embedder, query: str, *, limit: int, chunk_k: int = 30
) -> list[SearchHit]:
    """Items ranked by the cosine similarity of their best chunk."""
    if not vectors.tables_exist(conn):
        return []  # nothing has been embedded yet
    vectors.ensure_vector_tables(conn, embedder.model_name, embedder.dim)  # refuses a changed model
    query_vector = embedder.embed_query(query).tolist()

    # Many of the nearest chunks can belong to one item, so widen k until
    # enough different items are found or every chunk has been looked at.
    k = min(max(chunk_k, limit), MAX_K)
    while True:
        nearest = vectors.knn_chunks(conn, query_vector, k)
        hits = _best_chunk_per_item(conn, [(cid, 1.0 - dist) for cid, dist in nearest])
        if len(hits) >= limit or len(nearest) < k or k >= MAX_K:
            return [_with_rank(hit, dense_rank=i) for i, hit in enumerate(hits[:limit], start=1)]
        k = min(k * 2, MAX_K)


# ---------- BM25 ----------


def fts_query(query: str) -> str | None:
    """Turn free text into a safe FTS5 query: every word quoted, joined with OR.

    Quoting stops FTS5 from reading words like AND / NOT / NEAR, or characters like - " * :,
    as query syntax. OR (instead of FTS5's default AND) lets an item that matches only some
    of the words still be found; BM25 ranks the ones that match more, rarer words higher.
    """
    words = dict.fromkeys(word.lower() for word in _words(query))  # unique, in order
    if not words:
        return None
    return " OR ".join(f'"{word}"' for word in words)


def bm25_items(conn: sqlite3.Connection, query: str, *, limit: int, chunk_k: int = 30) -> list[SearchHit]:
    """Items ranked by the BM25 score of their best chunk."""
    match = fts_query(query)
    if match is None:
        return []
    k = max(chunk_k, limit)
    while True:
        # bm25() is lower-is-better (negative); negate it so a higher score means a better match.
        rows = conn.execute(
            "SELECT rowid, -bm25(chunks_fts) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, k),
        ).fetchall()
        hits = _best_chunk_per_item(conn, [(int(row[0]), float(row[1])) for row in rows])
        if len(hits) >= limit or len(rows) < k:
            return [_with_rank(hit, bm25_rank=i) for i, hit in enumerate(hits[:limit], start=1)]
        k *= 2


# ---------- fusion ----------


def fuse(rankings: list[list[SearchHit]], *, k: int = DEFAULT_RRF_K) -> list[SearchHit]:
    """Reciprocal rank fusion: score(item) = sum over lists of 1 / (k + rank in that list).

    Only ranks are used, never the raw scores, so cosine similarities and BM25 scores (which
    are on different scales) never have to be made comparable. The passage shown for an item
    comes from the list that ranked it highest (the first list wins a tie).
    """
    scores: dict[int, float] = {}
    best: dict[int, tuple[int, SearchHit]] = {}  # item id -> (best rank, hit from that list)
    dense_rank: dict[int, int] = {}
    bm25_rank: dict[int, int] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.item_id] = scores.get(hit.item_id, 0.0) + 1.0 / (k + rank)
            if hit.item_id not in best or rank < best[hit.item_id][0]:
                best[hit.item_id] = (rank, hit)
            if hit.dense_rank is not None:
                dense_rank[hit.item_id] = hit.dense_rank
            if hit.bm25_rank is not None:
                bm25_rank[hit.item_id] = hit.bm25_rank
    fused = [
        SearchHit(
            item_id=item_id,
            title=hit.title,
            url=hit.url,
            score=scores[item_id],
            passage=hit.passage,
            start_sec=hit.start_sec,
            dense_rank=dense_rank.get(item_id),
            bm25_rank=bm25_rank.get(item_id),
        )
        for item_id, (_, hit) in best.items()
    ]
    # Ties (same fused score) are broken by item id so results never depend on dict order.
    return sorted(fused, key=lambda h: (-h.score, h.item_id))


# ---------- display ----------


def describe(hit: SearchHit, mode: str) -> str:
    """Short label for a result: what its score means, and where each ranker placed it."""
    if mode == "dense":
        return f"similarity {hit.score:.2f}"
    if mode == "bm25":
        return f"BM25 {hit.score:.2f}"
    dense = f"#{hit.dense_rank}" if hit.dense_rank else "-"
    keyword = f"#{hit.bm25_rank}" if hit.bm25_rank else "-"
    return f"RRF {hit.score:.4f} (dense {dense}, BM25 {keyword})"


# ---------- shared ----------


def _with_rank(hit: SearchHit, *, dense_rank: int | None = None, bm25_rank: int | None = None) -> SearchHit:
    return SearchHit(
        hit.item_id, hit.title, hit.url, hit.score, hit.passage, hit.start_sec,
        dense_rank=dense_rank, bm25_rank=bm25_rank,
    )


def _best_chunk_per_item(
    conn: sqlite3.Connection, scored_chunks: list[tuple[int, float]]
) -> list[SearchHit]:
    """`scored_chunks` is sorted best-first, so the first chunk seen per item is its best."""
    hits: dict[int, SearchHit] = {}
    for chunk_id, score in scored_chunks:
        row = conn.execute(
            """
            SELECT c.item_id, c.text, c.start_sec, i.title, i.url
            FROM chunks c JOIN items i ON i.id = c.item_id WHERE c.id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if row is None:  # an index entry whose chunk was deleted; ignore it
            logger.warning("Chunk %d is in an index but has no chunk row; skipping", chunk_id)
            continue
        if row["item_id"] not in hits:
            hits[row["item_id"]] = SearchHit(
                item_id=int(row["item_id"]),
                title=row["title"] or row["url"],
                url=row["url"],
                score=score,
                passage=row["text"],
                start_sec=row["start_sec"],
            )
    return list(hits.values())
