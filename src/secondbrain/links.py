"""Item-to-item links and the graph built from them.

score = w * cosine(item vectors) + (1 - w) * jaccard(tags)

where tags are an item's topic and skill names together. A pair is stored as a
link when score >= LINK_THRESHOLD. Only pairs above the threshold are stored;
pair_score() computes any pair on demand (the eval needs low-scoring pairs too).
"""

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

from secondbrain import items, vectors
from secondbrain.config import Settings
from secondbrain.errors import SecondBrainError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairScore:
    embed_sim: float
    tag_jaccard: float
    score: float


def jaccard(a: set[int], b: set[int]) -> float:
    """Shared tags divided by all tags of the two items. 0 when either has none."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def combine(embed_sim: float, tag_jaccard: float, embed_weight: float) -> PairScore:
    sim = max(0.0, float(embed_sim))  # a negative cosine counts as "no similarity"
    score = embed_weight * sim + (1.0 - embed_weight) * tag_jaccard
    return PairScore(embed_sim=sim, tag_jaccard=tag_jaccard, score=score)


def all_pair_scores(conn: sqlite3.Connection, settings: Settings) -> list[tuple[int, int, PairScore]]:
    """Score every pair of items that have a vector. Returns (a, b, score) with a < b."""
    ids, matrix = vectors.all_item_vectors(conn)
    tags = items.all_item_tags(conn)
    sims = matrix @ matrix.T if ids else np.zeros((0, 0))
    pairs: list[tuple[int, int, PairScore]] = []
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            tag_sim = jaccard(tags.get(a, set()), tags.get(b, set()))
            pairs.append((a, b, combine(sims[i, j], tag_sim, settings.link_embed_weight)))
    return pairs


def pair_score(conn: sqlite3.Connection, settings: Settings, a: int, b: int) -> PairScore:
    """Score one pair, whether or not it is stored as a link."""
    ids, matrix = vectors.all_item_vectors(conn)
    missing = [item_id for item_id in (a, b) if item_id not in ids]
    if missing:
        raise SecondBrainError(f"Items {missing} have no embedding yet, so they cannot be scored.")
    tags = items.all_item_tags(conn)
    sim = float(matrix[ids.index(a)] @ matrix[ids.index(b)])
    return combine(sim, jaccard(tags.get(a, set()), tags.get(b, set())), settings.link_embed_weight)


def update_item(conn: sqlite3.Connection, settings: Settings, item_id: int) -> int:
    """Recompute the links of one item against all others. Returns how many were stored."""
    conn.execute("DELETE FROM links WHERE item_a = ? OR item_b = ?", (item_id, item_id))
    ids, matrix = vectors.all_item_vectors(conn)
    if item_id not in ids:
        return 0
    tags = items.all_item_tags(conn)
    sims = matrix @ matrix[ids.index(item_id)]
    stored = 0
    for other, sim in zip(ids, sims):
        if other == item_id:
            continue
        tag_sim = jaccard(tags.get(item_id, set()), tags.get(other, set()))
        stored += _store_if_linked(conn, settings, item_id, other, combine(sim, tag_sim, settings.link_embed_weight))
    logger.info("Item %d: %d links stored", item_id, stored)
    return stored


def rebuild(conn: sqlite3.Connection, settings: Settings) -> int:
    """Delete all links and recompute them. Use after changing the weight or threshold."""
    conn.execute("DELETE FROM links")
    stored = sum(
        _store_if_linked(conn, settings, a, b, score) for a, b, score in all_pair_scores(conn, settings)
    )
    conn.commit()
    logger.info("Rebuilt links: %d stored", stored)
    return stored


def related(conn: sqlite3.Connection, item_id: int, limit: int = 5) -> list[sqlite3.Row]:
    """The items most strongly linked to this one, best first."""
    items.get(conn, item_id)  # clear error for an unknown id
    return conn.execute(
        """
        SELECT i.id, i.title, i.url, l.score, l.embed_sim, l.tag_jaccard
        FROM links l
        JOIN items i ON i.id = CASE WHEN l.item_a = :id THEN l.item_b ELSE l.item_a END
        WHERE l.item_a = :id OR l.item_b = :id
        ORDER BY l.score DESC, i.id LIMIT :limit
        """,
        {"id": item_id, "limit": limit},
    ).fetchall()


def graph(conn: sqlite3.Connection, max_links_per_item: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Nodes and edges for a graph view. Each item contributes its strongest links only."""
    ids, _ = vectors.all_item_vectors(conn)
    nodes = [_node(conn, item_id) for item_id in ids]
    edges: dict[tuple[int, int], dict[str, Any]] = {}
    for item_id in ids:
        for row in related(conn, item_id, max_links_per_item):
            a, b = sorted((item_id, int(row["id"])))
            edges[(a, b)] = {
                "source": a,
                "target": b,
                "score": round(row["score"], 4),
                "embed_sim": round(row["embed_sim"], 4),
                "tag_jaccard": round(row["tag_jaccard"], 4),
            }
    return {"nodes": nodes, "edges": [edges[key] for key in sorted(edges)]}


def score_stats(conn: sqlite3.Connection, settings: Settings) -> dict[str, float]:
    """Distribution of scores over all pairs: the basis for choosing LINK_THRESHOLD."""
    scores = np.array([score.score for _, _, score in all_pair_scores(conn, settings)])
    if scores.size == 0:
        return {"pairs": 0}
    stats = {f"p{q}": float(np.percentile(scores, q)) for q in (10, 25, 50, 75, 90, 95, 99)}
    return {
        "pairs": int(scores.size),
        "min": float(scores.min()),
        **stats,
        "max": float(scores.max()),
        "at_or_above_threshold": int((scores >= settings.link_threshold).sum()),
    }


def _store_if_linked(
    conn: sqlite3.Connection, settings: Settings, a: int, b: int, score: PairScore
) -> int:
    if score.score < settings.link_threshold:
        return 0
    low, high = sorted((a, b))
    conn.execute(
        "INSERT OR REPLACE INTO links (item_a, item_b, embed_sim, tag_jaccard, score) VALUES (?, ?, ?, ?, ?)",
        (low, high, score.embed_sim, score.tag_jaccard, score.score),
    )
    return 1


def _node(conn: sqlite3.Connection, item_id: int) -> dict[str, Any]:
    item = items.get(conn, item_id)
    analysis = items.load_analysis(conn, item_id)
    return {
        "id": item_id,
        "title": item["title"] or item["url"],
        "url": item["url"],
        "source_type": item["source_type"],
        "duration_sec": item["duration_sec"],
        "progress_status": item["progress_status"],
        "difficulty": analysis["difficulty"] if analysis else None,
        "topics": items.load_tags(conn, item_id, "topic"),
    }
