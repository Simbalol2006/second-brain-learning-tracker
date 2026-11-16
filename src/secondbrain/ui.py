"""Helpers for the Streamlit app that contain no Streamlit code, so they can be unit-tested."""

import sqlite3
import textwrap
from typing import Any

from secondbrain import items
from secondbrain.config import Settings
from secondbrain.embed import Embedder, FastEmbedEmbedder

PROGRESS_MARKS = {"to_watch": "", "watching": "▶ ", "done": "✓ "}
PROGRESS_COLORS = {"to_watch": "#E8EEF7", "watching": "#FFE8A3", "done": "#BFE5C8"}
DIFFICULTY_COLORS = {"beginner": "#D6ECFF", "intermediate": "#A9D0F5", "advanced": "#6FA8DC"}
UNKNOWN_COLOR = "#EEEEEE"


def load_embedder(settings: Settings) -> Embedder:
    """Load the embedding model. The app caches the result for the whole session."""
    return FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)


def format_minutes(duration_sec: int | None) -> str:
    """'-' for unknown, '12 min', or '1 h 05 min'."""
    if duration_sec is None:
        return "-"
    minutes = max(1, round(duration_sec / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def item_label(row: Any) -> str:
    """Text for an item in a drop-down."""
    return f"{row['id']} · {row['title'] or row['url']}"


def library_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """One plain dict per saved item, ready for a table."""
    rows = []
    for item in items.list_all(conn):
        item_id = int(item["id"])
        analysis = items.load_analysis(conn, item_id)
        rows.append(
            {
                "id": item_id,
                "title": item["title"] or item["url"],
                "progress": item["progress_status"],
                "status": item["ingest_status"],
                "difficulty": analysis["difficulty"] if analysis else "",
                "time": format_minutes(item["duration_sec"]),
                "topics": ", ".join(items.load_tags(conn, item_id, "topic")),
                "source": item["source_type"],
            }
        )
    return rows


def filter_rows(
    rows: list[dict[str, Any]], progress: list[str], topic: str | None, text: str
) -> list[dict[str, Any]]:
    """Keep rows matching all given filters. Empty filters match everything."""
    needle = text.strip().lower()
    kept = []
    for row in rows:
        if progress and row["progress"] not in progress:
            continue
        if topic and topic not in row["topics"].split(", "):
            continue
        if needle and needle not in row["title"].lower() and needle not in row["topics"].lower():
            continue
        kept.append(row)
    return kept


def all_topics(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({topic for row in rows for topic in row["topics"].split(", ") if topic})


def neighbourhood(graph: dict[str, list[dict[str, Any]]], item_id: int) -> dict[str, list[dict[str, Any]]]:
    """The part of the graph around one item: the item, its direct neighbours, and links among them."""
    near = {item_id}
    for edge in graph["edges"]:
        if item_id in (edge["source"], edge["target"]):
            near.update((edge["source"], edge["target"]))
    return {
        "nodes": [node for node in graph["nodes"] if node["id"] in near],
        "edges": [e for e in graph["edges"] if e["source"] in near and e["target"] in near],
    }


def graph_dot(graph: dict[str, list[dict[str, Any]]], color_by: str = "progress", focus: int | None = None) -> str:
    """Graphviz DOT text for the item graph. color_by is 'progress' or 'difficulty'."""
    lines = [
        "graph items {",
        "  overlap=false; splines=true; bgcolor=transparent;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#666666"];',
        '  edge [color="#888888"];',
    ]
    for node in graph["nodes"]:
        if color_by == "progress":
            fill = PROGRESS_COLORS.get(node["progress_status"], UNKNOWN_COLOR)
        else:
            fill = DIFFICULTY_COLORS.get(node["difficulty"] or "", UNKNOWN_COLOR)
        label = PROGRESS_MARKS.get(node["progress_status"], "") + node["title"]
        wrapped = "\\n".join(_escape(line) for line in textwrap.wrap(label, width=24)[:3])
        border = ', penwidth=3, color="#222222"' if node["id"] == focus else ""
        lines.append(
            f'  n{node["id"]} [label="{wrapped}", tooltip="{_escape(node["title"])}", '
            f'fillcolor="{fill}", URL="{_escape(node["url"])}", target="_blank"{border}];'
        )
    for edge in graph["edges"]:
        width = 1.0 + 3.0 * float(edge["score"])
        lines.append(
            f'  n{edge["source"]} -- n{edge["target"]} '
            f'[penwidth={width:.1f}, tooltip="score {edge["score"]:.2f}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def _escape(text: str) -> str:
    """Make text safe inside a double-quoted DOT string."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
