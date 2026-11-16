"""Deleting an item removes everything derived from it."""

import sqlite3

import pytest
from conftest import FakeEmbedder, FakeProvider, add_text_item

from secondbrain import analyze, items, library, links, profile, search, vectors
from secondbrain.config import Settings
from secondbrain.errors import StorageError

ANSWER = '{"summary": "s", "key_points": ["a", "b"], "topics": ["shared topic"], "skills": [], "difficulty": "beginner"}'


def test_delete_removes_rows_vectors_links_and_derived_skills(conn: sqlite3.Connection, embedder: FakeEmbedder) -> None:
    settings = Settings(link_threshold=0.0, _env_file=None)
    text = "gradient descent trains neural network weights"
    first = add_text_item(conn, settings, embedder, "a" * 11, "First", text)
    second = add_text_item(conn, settings, embedder, "b" * 11, "Second", text + " faster")
    for item_id in (first, second):
        analyze.analyze_item(conn, settings, FakeProvider([ANSWER]), item_id)
    profile.set_progress(conn, first, "done")
    chunks_of_second = conn.execute("SELECT count(*) FROM chunks WHERE item_id = ?", (second,)).fetchone()[0]
    assert links.related(conn, second) and profile.skills(conn)

    assert library.delete_item(conn, first) == "First"

    assert [int(r["id"]) for r in items.list_all(conn)] == [second]
    for table in ("transcripts", "chunks", "analyses", "item_topics", "ingest_timings"):
        assert conn.execute(f"SELECT count(*) FROM {table} WHERE item_id = ?", (first,)).fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM links").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == chunks_of_second
    assert vectors.all_item_vectors(conn)[0] == [second]
    assert profile.skills(conn) == []  # the skill came only from the deleted item
    assert conn.execute("SELECT count(*) FROM llm_calls WHERE item_id IS NULL").fetchone()[0] == 1  # log kept
    assert [hit.item_id for hit in search.search(conn, embedder, "gradient descent")] == [second]


def test_delete_unknown_item(conn: sqlite3.Connection) -> None:
    with pytest.raises(StorageError, match="No item with id 4"):
        library.delete_item(conn, 4)


def test_delete_item_that_never_got_a_transcript(conn: sqlite3.Connection) -> None:
    item_id, _ = items.get_or_create(conn, "youtube", "c" * 11, "https://www.youtube.com/watch?v=" + "c" * 11)
    library.delete_item(conn, item_id)
    assert items.list_all(conn) == []
