"""Link tests with hand-made vectors and tags, so every score is known exactly."""

import sqlite3

import pytest
from conftest import FakeEmbedder, FakeProvider, add_text_item

from secondbrain import analyze, items, links, vectors
from secondbrain.config import Settings
from secondbrain.errors import SecondBrainError, StorageError
from secondbrain.schemas import Analysis

VECTORS = {"A": [1.0, 0.0, 0.0, 0.0], "B": [0.8, 0.6, 0.0, 0.0], "C": [0.0, 0.0, 1.0, 0.0]}
TAGS = {"A": (["x", "y"], []), "B": (["y"], ["z"]), "C": (["q"], [])}


def make_item(conn: sqlite3.Connection, name: str) -> int:
    item_id, _ = items.get_or_create(conn, "youtube", f"{name * 11}", f"https://www.youtube.com/watch?v={name * 11}")
    conn.execute("UPDATE items SET title = ? WHERE id = ?", (f"Item {name}", item_id))
    vectors.ensure_vector_tables(conn, "m", 4)
    vectors.set_item_vector(conn, item_id, VECTORS[name])
    topics, skills = TAGS[name]
    analysis = Analysis(summary="s", key_points=["a", "b"], topics=topics, skills=skills, difficulty="beginner")
    items.save_analysis(conn, item_id, analysis, "fake", "fake-model", "v1")
    conn.commit()
    return item_id


@pytest.fixture
def abc(conn: sqlite3.Connection) -> dict[str, int]:
    return {name: make_item(conn, name) for name in "ABC"}


def test_jaccard() -> None:
    assert links.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert links.jaccard({1}, {1}) == 1.0
    assert links.jaccard(set(), {1}) == 0.0
    assert links.jaccard(set(), set()) == 0.0


def test_combine_uses_the_weight_and_ignores_negative_similarity() -> None:
    assert links.combine(0.8, 0.5, 0.7).score == pytest.approx(0.7 * 0.8 + 0.3 * 0.5)
    assert links.combine(-0.4, 0.5, 0.7).score == pytest.approx(0.15)
    assert links.combine(1.0, 1.0, 0.7).score == pytest.approx(1.0)


def test_pair_score_matches_the_formula(conn: sqlite3.Connection, settings: Settings, abc: dict[str, int]) -> None:
    score = links.pair_score(conn, settings, abc["A"], abc["B"])
    assert score.embed_sim == pytest.approx(0.8)
    assert score.tag_jaccard == pytest.approx(1 / 3)  # tags {x,y} and {y,z}: skills count too
    assert score.score == pytest.approx(0.7 * 0.8 + 0.3 / 3)
    assert links.pair_score(conn, settings, abc["B"], abc["A"]) == score
    assert links.pair_score(conn, settings, abc["A"], abc["C"]).score == pytest.approx(0.0)


def test_pair_score_needs_embedded_items(conn: sqlite3.Connection, settings: Settings, abc: dict[str, int]) -> None:
    with pytest.raises(SecondBrainError, match=r"Items \[99\] have no embedding"):
        links.pair_score(conn, settings, abc["A"], 99)


def test_rebuild_stores_only_pairs_above_threshold(conn: sqlite3.Connection, settings: Settings, abc: dict[str, int]) -> None:
    assert links.rebuild(conn, settings) == 1
    row = conn.execute("SELECT * FROM links").fetchone()
    assert (row["item_a"], row["item_b"]) == (abc["A"], abc["B"])
    assert row["score"] == pytest.approx(0.66)


def test_threshold_and_weight_come_from_settings(conn: sqlite3.Connection, abc: dict[str, int]) -> None:
    strict = Settings(link_threshold=0.9, _env_file=None)
    assert links.rebuild(conn, strict) == 0
    loose = Settings(link_threshold=0.0, _env_file=None)
    assert links.rebuild(conn, loose) == 3
    tags_only = Settings(link_embed_weight=0.0, link_threshold=0.3, _env_file=None)
    assert links.rebuild(conn, tags_only) == 1
    assert conn.execute("SELECT score FROM links").fetchone()[0] == pytest.approx(1 / 3)


def test_related_is_symmetric_and_sorted(conn: sqlite3.Connection, abc: dict[str, int]) -> None:
    links.rebuild(conn, Settings(link_threshold=0.0, _env_file=None))
    from_a = links.related(conn, abc["A"])
    assert [row["id"] for row in from_a] == [abc["B"], abc["C"]]
    assert from_a[0]["title"] == "Item B"
    assert [row["id"] for row in links.related(conn, abc["B"])][0] == abc["A"]
    assert len(links.related(conn, abc["A"], limit=1)) == 1


def test_related_unknown_item(conn: sqlite3.Connection) -> None:
    with pytest.raises(StorageError, match="No item with id 5"):
        links.related(conn, 5)


def test_update_item_replaces_only_that_items_links(conn: sqlite3.Connection, abc: dict[str, int]) -> None:
    loose = Settings(link_threshold=0.0, _env_file=None)
    links.rebuild(conn, loose)
    vectors.set_item_vector(conn, abc["C"], [1.0, 0.0, 0.0, 0.0])  # C now points the same way as A
    assert links.update_item(conn, loose, abc["C"]) == 2
    assert links.pair_score(conn, loose, abc["A"], abc["C"]).embed_sim == pytest.approx(1.0)
    stored = {(r["item_a"], r["item_b"]): r["embed_sim"] for r in conn.execute("SELECT * FROM links")}
    assert stored[(abc["A"], abc["C"])] == pytest.approx(1.0)
    assert stored[(abc["A"], abc["B"])] == pytest.approx(0.8)  # untouched


def test_score_stats(conn: sqlite3.Connection, settings: Settings, abc: dict[str, int]) -> None:
    stats = links.score_stats(conn, settings)
    assert stats["pairs"] == 3
    assert stats["max"] == pytest.approx(0.66)
    assert stats["min"] == pytest.approx(0.0)
    assert stats["at_or_above_threshold"] == 1
    assert "p50" in stats and "p95" in stats


def test_score_stats_on_empty_database(conn: sqlite3.Connection, settings: Settings) -> None:
    assert links.score_stats(conn, settings) == {"pairs": 0}


def test_graph_has_nodes_and_deduplicated_edges(conn: sqlite3.Connection, abc: dict[str, int]) -> None:
    links.rebuild(conn, Settings(link_threshold=0.0, _env_file=None))
    graph = links.graph(conn)
    assert [n["id"] for n in graph["nodes"]] == sorted(abc.values())
    node = graph["nodes"][0]
    assert node["title"] == "Item A"
    assert node["topics"] == ["x", "y"]
    assert node["difficulty"] == "beginner"
    assert node["progress_status"] == "to_watch"
    assert len(graph["edges"]) == 3  # each pair once, not once per direction
    assert {(e["source"], e["target"]) for e in graph["edges"]} == {
        (abc["A"], abc["B"]), (abc["A"], abc["C"]), (abc["B"], abc["C"])
    }
    assert all(e["source"] < e["target"] for e in graph["edges"])


def test_graph_keeps_only_the_strongest_links_per_item(conn: sqlite3.Connection, abc: dict[str, int]) -> None:
    links.rebuild(conn, Settings(link_threshold=0.0, _env_file=None))
    graph = links.graph(conn, max_links_per_item=1)
    assert (abc["A"], abc["B"]) in {(e["source"], e["target"]) for e in graph["edges"]}
    assert len(graph["edges"]) < 3


def test_graph_of_empty_database(conn: sqlite3.Connection) -> None:
    assert links.graph(conn) == {"nodes": [], "edges": []}


def test_ingest_links_similar_items(conn: sqlite3.Connection, embedder: FakeEmbedder) -> None:
    settings = Settings(link_threshold=0.6, _env_file=None)
    text = "gradient descent updates neural network weights to reduce training loss"
    first = add_text_item(conn, settings, embedder, "aaaaaaaaaaa", "Part 1", text)
    second = add_text_item(conn, settings, embedder, "bbbbbbbbbbb", "Part 2", text + " with momentum")
    other = add_text_item(conn, settings, embedder, "ccccccccccc", "Bread", "sourdough bread needs flour water and salt")
    assert [row["id"] for row in links.related(conn, second)] == [first]
    assert links.related(conn, other) == []
    assert "link" in [t["stage"] for t in items.load_timings(conn, second)]


def test_analysis_updates_links_through_shared_tags(conn: sqlite3.Connection, embedder: FakeEmbedder) -> None:
    strict = Settings(link_threshold=0.99, _env_file=None)
    first = add_text_item(conn, strict, embedder, "aaaaaaaaaaa", "One", "alpha beta gamma delta epsilon zeta eta theta")
    second = add_text_item(conn, strict, embedder, "bbbbbbbbbbb", "Two", "alpha beta gamma delta one two three four five six")
    before = links.pair_score(conn, strict, first, second)
    assert before.tag_jaccard == 0.0
    assert links.related(conn, first) == []

    # A threshold the pair misses on text alone but reaches once the shared tag adds 0.3.
    settings = Settings(link_threshold=before.score + 0.15, _env_file=None)
    answer = '{"summary": "s", "key_points": ["a", "b"], "topics": ["greek letters"], "skills": [], "difficulty": "beginner"}'
    analyze.analyze_item(conn, settings, FakeProvider([answer]), first)
    assert links.related(conn, first) == []  # only one of the two has the tag so far
    analyze.analyze_item(conn, settings, FakeProvider([answer]), second)
    after = links.pair_score(conn, settings, first, second)
    assert after.tag_jaccard == 1.0
    assert after.score == pytest.approx(before.score + 0.3)
    assert [row["id"] for row in links.related(conn, first)] == [second]
