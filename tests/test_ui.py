"""Tests for the Streamlit-free UI helpers."""

import sqlite3

import pytest

from secondbrain import items, ui
from secondbrain.schemas import Analysis

GRAPH = {
    "nodes": [
        {"id": 1, "title": 'Intro to "RAG" systems and retrieval pipelines for beginners', "url": "https://a", "progress_status": "done", "difficulty": "beginner"},
        {"id": 2, "title": "Vector databases", "url": "https://b", "progress_status": "watching", "difficulty": "advanced"},
        {"id": 3, "title": "Sourdough", "url": "https://c", "progress_status": "to_watch", "difficulty": None},
        {"id": 4, "title": "Rye bread", "url": "https://d", "progress_status": "to_watch", "difficulty": "beginner"},
    ],
    "edges": [
        {"source": 1, "target": 2, "score": 0.8, "embed_sim": 0.9, "tag_jaccard": 0.5},
        {"source": 3, "target": 4, "score": 0.6, "embed_sim": 0.8, "tag_jaccard": 0.1},
    ],
}


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, "-"), (10, "1 min"), (630, "10 min"), (3540, "59 min"), (3600, "1 h 00 min"), (5400, "1 h 30 min")],
)
def test_format_minutes(seconds: int | None, expected: str) -> None:
    assert ui.format_minutes(seconds) == expected


def test_item_label() -> None:
    assert ui.item_label({"id": 3, "title": "A talk", "url": "u"}) == "3 · A talk"
    assert ui.item_label({"id": 3, "title": None, "url": "https://x"}) == "3 · https://x"


def test_library_rows(conn: sqlite3.Connection) -> None:
    first, _ = items.get_or_create(conn, "youtube", "aaaaaaaaaaa", "https://www.youtube.com/watch?v=aaaaaaaaaaa")
    conn.execute("UPDATE items SET title = 'A talk', duration_sec = 630 WHERE id = ?", (first,))
    analysis = Analysis(summary="s", key_points=["a", "b"], topics=["rag", "search"], skills=[], difficulty="advanced")
    items.save_analysis(conn, first, analysis, "fake", "m", "v1")
    items.get_or_create(conn, "article", "x", "https://example.com/x")
    rows = ui.library_rows(conn)
    assert rows[0] == {
        "id": first, "title": "A talk", "progress": "to_watch", "status": "new",
        "difficulty": "advanced", "time": "10 min", "topics": "rag, search", "source": "youtube",
    }
    assert rows[1]["title"] == "https://example.com/x" and rows[1]["difficulty"] == "" and rows[1]["time"] == "-"


ROWS = [
    {"id": 1, "title": "Intro to RAG", "progress": "done", "topics": "rag, search"},
    {"id": 2, "title": "Vector DBs", "progress": "to_watch", "topics": "search"},
    {"id": 3, "title": "Sourdough", "progress": "to_watch", "topics": ""},
]


@pytest.mark.parametrize(
    ("progress", "topic", "text", "expected"),
    [
        ([], None, "", [1, 2, 3]),
        (["to_watch"], None, "", [2, 3]),
        ([], "search", "", [1, 2]),
        ([], "rag", "", [1]),
        ([], None, "VECTOR", [2]),
        ([], None, "rag", [1]),
        (["done"], "search", "intro", [1]),
        (["watching"], None, "", []),
    ],
)
def test_filter_rows(progress: list[str], topic: str | None, text: str, expected: list[int]) -> None:
    assert [r["id"] for r in ui.filter_rows(ROWS, progress, topic, text)] == expected


def test_all_topics() -> None:
    assert ui.all_topics(ROWS) == ["rag", "search"]


def test_neighbourhood_keeps_the_item_and_its_direct_neighbours() -> None:
    near = ui.neighbourhood(GRAPH, 1)
    assert [n["id"] for n in near["nodes"]] == [1, 2]
    assert len(near["edges"]) == 1
    alone = ui.neighbourhood({"nodes": GRAPH["nodes"], "edges": []}, 3)
    assert [n["id"] for n in alone["nodes"]] == [3] and alone["edges"] == []


def test_graph_dot_structure() -> None:
    dot = ui.graph_dot(GRAPH)
    assert dot.startswith("graph items {") and dot.endswith("}")
    assert dot.count(" -- ") == 2
    assert "n1 -- n2 [penwidth=3.4" in dot and 'tooltip="score 0.80"' in dot
    for node_id in (1, 2, 3, 4):
        assert f"  n{node_id} [label=" in dot


def test_graph_dot_escapes_quotes_and_wraps_titles() -> None:
    dot = ui.graph_dot(GRAPH)
    assert '\\"RAG\\"' in dot  # the quote inside the title cannot end the DOT string
    first_node = next(line for line in dot.splitlines() if line.startswith("  n1 "))
    assert first_node.count("\\n") == 2  # wrapped to three lines
    assert 'tooltip="Intro to \\"RAG\\" systems and retrieval pipelines for beginners"' in first_node


def test_graph_dot_marks_and_colours() -> None:
    by_progress = ui.graph_dot(GRAPH, "progress")
    assert 'label="✓ Intro' in by_progress and 'label="▶ Vector databases"' in by_progress
    assert ui.PROGRESS_COLORS["done"] in by_progress and ui.PROGRESS_COLORS["watching"] in by_progress
    by_difficulty = ui.graph_dot(GRAPH, "difficulty")
    assert ui.DIFFICULTY_COLORS["advanced"] in by_difficulty
    assert ui.UNKNOWN_COLOR in by_difficulty  # the item with no analysis


def test_graph_dot_highlights_the_focus() -> None:
    assert "penwidth=3, color" not in ui.graph_dot(GRAPH)
    focused = ui.graph_dot(GRAPH, focus=2)
    assert [line for line in focused.splitlines() if "penwidth=3, color" in line][0].startswith("  n2 ")
