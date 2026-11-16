"""Profile tests: progress, derived skill levels, manual skills, goals."""

import sqlite3

import pytest

from secondbrain import analyze, items, profile
from secondbrain.config import Settings
from secondbrain.errors import SecondBrainError, StorageError
from secondbrain.models import Segment, Transcript
from secondbrain.schemas import Analysis
from conftest import FakeProvider


def add_analyzed(conn: sqlite3.Connection, n: int, topics: list[str], difficulty: str, skills: list[str] | None = None) -> int:
    item_id, _ = items.get_or_create(conn, "youtube", f"{n:011d}", f"https://www.youtube.com/watch?v={n:011d}")
    conn.execute("UPDATE items SET title = ? WHERE id = ?", (f"Item {n}", item_id))
    analysis = Analysis(summary="s", key_points=["a", "b"], topics=topics, skills=skills or [], difficulty=difficulty)
    items.save_analysis(conn, item_id, analysis, "fake", "fake-model", "v1")
    conn.commit()
    return item_id


def levels(conn: sqlite3.Connection) -> dict[str, tuple[int, str, int]]:
    return {r["name"]: (r["level"], r["source"], r["evidence_count"]) for r in profile.skills(conn)}


@pytest.mark.parametrize(
    ("finished", "hardest", "expected"),
    [(0, 0, 0), (1, 1, 1), (1, 3, 1), (2, 1, 1), (2, 2, 2), (2, 3, 2), (3, 2, 2), (3, 3, 3), (9, 1, 1)],
)
def test_derive_level(finished: int, hardest: int, expected: int) -> None:
    assert profile.derive_level(finished, hardest) == expected


def test_marking_done_creates_derived_skills(conn: sqlite3.Connection) -> None:
    item_id = add_analyzed(conn, 1, ["python"], "beginner", skills=["scripting"])
    assert levels(conn) == {}
    profile.set_progress(conn, item_id, "done")
    assert levels(conn) == {"python": (1, "derived", 1), "scripting": (1, "derived", 1)}
    row = items.get(conn, item_id)
    assert row["progress_status"] == "done" and row["done_at"] is not None


def test_levels_grow_with_more_and_harder_finished_items(conn: sqlite3.Connection) -> None:
    ids = [
        add_analyzed(conn, 1, ["python"], "beginner"),
        add_analyzed(conn, 2, ["python"], "intermediate"),
        add_analyzed(conn, 3, ["python"], "advanced"),
    ]
    profile.set_progress(conn, ids[0], "done")
    assert levels(conn)["python"] == (1, "derived", 1)
    profile.set_progress(conn, ids[1], "done")
    assert levels(conn)["python"] == (2, "derived", 2)
    profile.set_progress(conn, ids[2], "done")
    assert levels(conn)["python"] == (3, "derived", 3)


def test_watching_does_not_count_and_undoing_done_removes_the_skill(conn: sqlite3.Connection) -> None:
    item_id = add_analyzed(conn, 1, ["python"], "beginner")
    profile.set_progress(conn, item_id, "watching")
    assert levels(conn) == {}
    profile.set_progress(conn, item_id, "done")
    profile.set_progress(conn, item_id, "to_watch")
    assert levels(conn) == {}
    assert items.get(conn, item_id)["done_at"] is None


def test_manual_level_is_never_overwritten(conn: sqlite3.Connection) -> None:
    assert profile.set_skill(conn, "  Python ", 3) == "python"
    item_id = add_analyzed(conn, 1, ["python"], "beginner")
    profile.set_progress(conn, item_id, "done")
    assert levels(conn)["python"] == (3, "manual", 1)  # level kept, evidence still counted
    profile.set_progress(conn, item_id, "to_watch")
    assert levels(conn)["python"] == (3, "manual", 0)


def test_removing_a_manual_level_falls_back_to_the_derived_one(conn: sqlite3.Connection) -> None:
    item_id = add_analyzed(conn, 1, ["python"], "beginner")
    profile.set_progress(conn, item_id, "done")
    profile.set_skill(conn, "python", 3)
    profile.remove_skill(conn, "Python")
    assert levels(conn)["python"] == (1, "derived", 1)
    with pytest.raises(SecondBrainError, match="No manually set skill named 'python'"):
        profile.remove_skill(conn, "python")


def test_manual_skill_can_be_new_and_can_be_zero(conn: sqlite3.Connection) -> None:
    profile.set_skill(conn, "rust", 0)
    assert levels(conn)["rust"] == (0, "manual", 0)
    assert profile.skill_levels(conn) == {1: 0}


@pytest.mark.parametrize("level", [-1, 4])
def test_invalid_level_is_rejected(conn: sqlite3.Connection, level: int) -> None:
    with pytest.raises(SecondBrainError, match="Level must be 0, 1, 2 or 3"):
        profile.set_skill(conn, "python", level)


def test_empty_skill_name_is_rejected(conn: sqlite3.Connection) -> None:
    with pytest.raises(SecondBrainError, match="skill name is empty"):
        profile.set_skill(conn, " .. ", 1)


def test_invalid_status_and_unknown_item(conn: sqlite3.Connection) -> None:
    item_id = add_analyzed(conn, 1, ["python"], "beginner")
    with pytest.raises(SecondBrainError, match="Unknown status 'finished'"):
        profile.set_progress(conn, item_id, "finished")
    with pytest.raises(StorageError, match="No item with id 99"):
        profile.set_progress(conn, 99, "done")


def test_goals(conn: sqlite3.Connection) -> None:
    first = profile.add_goal(conn, "  become an   ML engineer ")
    second = profile.add_goal(conn, "learn SQL")
    assert [(g["id"], g["text"]) for g in profile.goals(conn)] == [(first, "become an ML engineer"), (second, "learn SQL")]
    profile.remove_goal(conn, first)
    assert [g["text"] for g in profile.goals(conn)] == ["learn SQL"]
    with pytest.raises(SecondBrainError, match="No goal with id"):
        profile.remove_goal(conn, first)
    with pytest.raises(SecondBrainError, match="goal text is empty"):
        profile.add_goal(conn, "   ")


def test_finished_titles_newest_first(conn: sqlite3.Connection) -> None:
    first = add_analyzed(conn, 1, ["a"], "beginner")
    second = add_analyzed(conn, 2, ["b"], "beginner")
    profile.set_progress(conn, first, "done")
    profile.set_progress(conn, second, "done")
    conn.execute("UPDATE items SET done_at = '2026-01-01 00:00:00' WHERE id = ?", (first,))
    assert profile.finished_titles(conn) == ["Item 2", "Item 1"]
    assert profile.finished_titles(conn, limit=1) == ["Item 2"]


def test_reanalysis_of_a_finished_item_updates_skills(conn: sqlite3.Connection, settings: Settings) -> None:
    item_id = add_analyzed(conn, 1, ["python"], "beginner")
    items.save_transcript(conn, item_id, Transcript([Segment("some text")], "auto", "en"))
    profile.set_progress(conn, item_id, "done")
    answer = '{"summary": "s", "key_points": ["a", "b"], "topics": ["rust"], "skills": [], "difficulty": "beginner"}'
    analyze.analyze_item(conn, settings, FakeProvider([answer]), item_id, force=True)
    assert set(levels(conn)) == {"rust"}
