"""Label templates for the evaluation set.

Templates refer to items by URL, not by database id, so the labelled files stay
valid if the database is rebuilt. Pair scores are deliberately left out of the
files: pairs must be labelled without knowing what the system thinks.
"""

import random
import sqlite3
from pathlib import Path

from secondbrain import items, links
from secondbrain.config import Settings
from secondbrain.errors import SecondBrainError

RETRIEVAL_FILE = "retrieval.yaml"
LINKS_FILE = "links.yaml"
SUMMARIES_FILE = "summaries.yaml"
DEFAULT_SEED = 7


def sample_pairs(
    scored: list[tuple[int, int, float]], per_band: int, seed: int
) -> list[tuple[int, int]]:
    """Pick pairs from across the score range: highest, around the middle, and random.

    Returns up to 3 * per_band pairs in shuffled order, without their scores.
    """
    rng = random.Random(seed)
    ranked = sorted(scored, key=lambda pair: pair[2], reverse=True)
    if len(ranked) <= 3 * per_band:
        chosen = ranked
    else:
        top = ranked[:per_band]
        rest = ranked[per_band:]
        middle_start = max(0, len(rest) // 2 - per_band // 2)
        middle = rest[middle_start : middle_start + per_band]
        remaining = rest[:middle_start] + rest[middle_start + per_band :]
        chosen = top + middle + rng.sample(remaining, per_band)
    pairs = [(a, b) for a, b, _ in chosen]
    rng.shuffle(pairs)
    return pairs


def write_templates(
    conn: sqlite3.Connection,
    settings: Settings,
    out_dir: Path,
    *,
    n_pairs: int = 30,
    n_summaries: int = 15,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> list[Path]:
    """Write the three template files. Existing files are never overwritten unless force."""
    targets = [out_dir / name for name in (RETRIEVAL_FILE, LINKS_FILE, SUMMARIES_FILE)]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not force:
        raise SecondBrainError(
            f"Not overwriting existing label files: {', '.join(existing)}. "
            "Move them away or pass --force (this would erase your labels)."
        )
    rows = {int(row["id"]): row for row in items.list_all(conn)}
    scored = [(a, b, score.score) for a, b, score in links.all_pair_scores(conn, settings)]
    if not scored:
        raise SecondBrainError("Need at least two embedded items before templates can be made.")
    analyzed = [item_id for item_id in rows if items.load_analysis(conn, item_id) is not None]

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = sample_pairs(scored, per_band=n_pairs // 3, seed=seed)
    summary_ids = sorted(random.Random(seed).sample(analyzed, min(n_summaries, len(analyzed))))
    embedded_ids = sorted({item_id for a, b, _ in scored for item_id in (a, b)})
    targets[0].write_text(_retrieval_template([rows[i] for i in embedded_ids]), encoding="utf-8")
    targets[1].write_text(_links_template([(rows[a], rows[b]) for a, b in pairs]), encoding="utf-8")
    targets[2].write_text(_summaries_template([rows[i] for i in summary_ids]), encoding="utf-8")
    return targets


def _title(row: sqlite3.Row) -> str:
    """Title made safe for a one-line YAML comment."""
    return " ".join(str(row["title"] or "(no title)").split())


def _retrieval_template(rows: list[sqlite3.Row]) -> str:
    listing = "\n".join(f"#   {row['url']}  |  {_title(row)}" for row in rows)
    return f"""# Retrieval eval. Goal: about 25 queries.
#
# For each query, write something you would really type to find a saved item,
# then list the URL of EVERY saved item that is a good answer (usually 1-3).
# Write the queries from memory of the content, not by copying titles.
#
# Example:
#   queries:
#     - query: "how does attention work in transformers"
#       relevant:
#         - https://www.youtube.com/watch?v=XXXXXXXXXXX
#
# Saved items you can refer to:
{listing}

queries: []
"""


def _links_template(pairs: list[tuple[sqlite3.Row, sqlite3.Row]]) -> str:
    entries = "\n".join(
        f"  - a: {a['url']}  # {_title(a)}\n    b: {b['url']}  # {_title(b)}\n    related: null"
        for a, b in pairs
    )
    return f"""# Linking eval. {len(pairs)} pairs, in random order.
#
# For each pair set `related:` to true or false.
#   true  = if you were studying one, the other would be a sensible thing to see
#           next to it (same subject, or one builds on the other).
#   false = no meaningful connection beyond both being learning content.
# Judge from what the two items are about. The system's own scores are left out
# on purpose so they cannot influence you.

pairs:
{entries}
"""


def _summaries_template(rows: list[sqlite3.Row]) -> str:
    entries = "\n".join(f"  - url: {row['url']}  # {_title(row)}\n    spot_check: false" for row in rows)
    return f"""# Summary faithfulness eval. {len(rows)} items picked at random.
#
# An LLM judge will check every summary sentence and key point against the
# transcript. Set `spot_check: true` on 5 items you are willing to check by
# hand as well; the eval will report how often you and the judge agree.

items:
{entries}
"""
