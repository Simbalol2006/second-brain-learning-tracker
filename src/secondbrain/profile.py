"""The learner profile: goals, skill levels, and progress on saved items.

Skill levels are 0-3 (none, beginner, intermediate, advanced). A level is either
set by hand ('manual') or worked out from finished items ('derived'):

    level 1: at least 1 finished item with that tag
    level 2: at least 2 finished items, one of them intermediate or advanced
    level 3: at least 3 finished items, one of them advanced

Derived levels are recomputed from scratch whenever progress changes, so undoing
a "done" also undoes its effect. Manual levels are never changed by this.
"""

import logging
import sqlite3

from secondbrain import items
from secondbrain.errors import SecondBrainError
from secondbrain.schemas import normalize_tag

logger = logging.getLogger(__name__)

PROGRESS_STATES = ("to_watch", "watching", "done")
LEVEL_NAMES = ("none", "beginner", "intermediate", "advanced")
DIFFICULTY_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3}


def derive_level(finished_items: int, hardest_rank: int) -> int:
    """Skill level implied by finished items (see the module docstring)."""
    if finished_items >= 3 and hardest_rank >= 3:
        return 3
    if finished_items >= 2 and hardest_rank >= 2:
        return 2
    return 1 if finished_items >= 1 else 0


def set_progress(conn: sqlite3.Connection, item_id: int, status: str) -> None:
    """Change an item's progress and bring the derived skills up to date."""
    if status not in PROGRESS_STATES:
        raise SecondBrainError(f"Unknown status '{status}'. Use one of: {', '.join(PROGRESS_STATES)}.")
    items.get(conn, item_id)  # clear error for an unknown id
    conn.execute(
        "UPDATE items SET progress_status = ?, "
        "done_at = CASE WHEN ? = 'done' THEN datetime('now') ELSE NULL END WHERE id = ?",
        (status, status, item_id),
    )
    refresh_derived(conn)
    conn.commit()
    logger.info("Item %d marked %s", item_id, status)


def refresh_derived(conn: sqlite3.Connection) -> None:
    """Recompute every derived skill level from the items marked done."""
    evidence = conn.execute(
        """
        SELECT it.topic_id, count(DISTINCT i.id) AS finished,
               max(CASE a.difficulty WHEN 'advanced' THEN 3 WHEN 'intermediate' THEN 2 ELSE 1 END) AS hardest
        FROM items i
        JOIN item_topics it ON it.item_id = i.id
        JOIN analyses a ON a.item_id = i.id
        WHERE i.progress_status = 'done'
        GROUP BY it.topic_id
        """
    ).fetchall()
    conn.execute("DELETE FROM profile_skills WHERE source = 'derived'")
    conn.execute("UPDATE profile_skills SET evidence_count = 0")
    for row in evidence:
        level = derive_level(row["finished"], row["hardest"])
        conn.execute(
            """
            INSERT INTO profile_skills (topic_id, level, source, evidence_count)
            VALUES (?, ?, 'derived', ?)
            ON CONFLICT (topic_id) DO UPDATE SET evidence_count = excluded.evidence_count
            """,
            (row["topic_id"], level, row["finished"]),
        )


def set_skill(conn: sqlite3.Connection, name: str, level: int) -> str:
    """Set a skill level by hand. Returns the normalized skill name."""
    if level not in range(len(LEVEL_NAMES)):
        raise SecondBrainError("Level must be 0, 1, 2 or 3 (none, beginner, intermediate, advanced).")
    tag = normalize_tag(name)
    if not tag:
        raise SecondBrainError("The skill name is empty.")
    conn.execute("INSERT OR IGNORE INTO topics (name) VALUES (?)", (tag,))
    conn.execute(
        """
        INSERT INTO profile_skills (topic_id, level, source)
        SELECT id, ?, 'manual' FROM topics WHERE name = ?
        ON CONFLICT (topic_id) DO UPDATE
            SET level = excluded.level, source = 'manual', updated_at = datetime('now')
        """,
        (level, tag),
    )
    refresh_derived(conn)  # fills in evidence_count for the new row
    conn.commit()
    return tag


def remove_skill(conn: sqlite3.Connection, name: str) -> None:
    """Drop a manual level. The skill falls back to its derived level, if any."""
    tag = normalize_tag(name)
    cursor = conn.execute(
        "DELETE FROM profile_skills WHERE source = 'manual' "
        "AND topic_id = (SELECT id FROM topics WHERE name = ?)",
        (tag,),
    )
    if cursor.rowcount == 0:
        raise SecondBrainError(f"No manually set skill named '{tag}'.")
    refresh_derived(conn)
    conn.commit()


def skills(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All skills with a level above 0 or set by hand, strongest first."""
    return conn.execute(
        """
        SELECT t.name, p.level, p.source, p.evidence_count
        FROM profile_skills p JOIN topics t ON t.id = p.topic_id
        ORDER BY p.level DESC, p.evidence_count DESC, t.name
        """
    ).fetchall()


def skill_levels(conn: sqlite3.Connection) -> dict[int, int]:
    """topic_id -> level, for scoring."""
    return {int(r[0]): int(r[1]) for r in conn.execute("SELECT topic_id, level FROM profile_skills")}


def add_goal(conn: sqlite3.Connection, text: str) -> int:
    text = " ".join(text.split())
    if not text:
        raise SecondBrainError("The goal text is empty.")
    cursor = conn.execute("INSERT INTO profile_goals (text) VALUES (?)", (text,))
    conn.commit()
    return int(cursor.lastrowid)  # type: ignore[arg-type]


def remove_goal(conn: sqlite3.Connection, goal_id: int) -> None:
    cursor = conn.execute("DELETE FROM profile_goals WHERE id = ?", (goal_id,))
    if cursor.rowcount == 0:
        raise SecondBrainError(f"No goal with id {goal_id}. See: sb profile show")
    conn.commit()


def goals(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT id, text FROM profile_goals ORDER BY id").fetchall()


def finished_titles(conn: sqlite3.Connection, limit: int = 15) -> list[str]:
    """Titles of the most recently finished items."""
    rows = conn.execute(
        "SELECT coalesce(title, url) FROM items WHERE progress_status = 'done' "
        "ORDER BY done_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [str(row[0]) for row in rows]
