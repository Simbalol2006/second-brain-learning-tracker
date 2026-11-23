"""Phase 7: the BM25 index (FTS5), reciprocal rank fusion and the three search modes."""

import sqlite3

import pytest
from conftest import FakeEmbedder, add_text_item

from secondbrain import db, items, library
from secondbrain.config import Settings
from secondbrain.errors import SecondBrainError
from secondbrain.models import Chunk
from secondbrain.search import SearchHit, bm25_items, describe, fts_query, fuse, search

ML = "gradient descent updates neural network weights to reduce training loss"
BREAD = "sourdough bread needs flour water salt and a long slow fermentation"
SQL = "relational database tables are combined with joins in sql queries"


@pytest.fixture
def library_ids(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder) -> dict[str, int]:
    return {
        "ml": add_text_item(conn, settings, embedder, "aaaaaaaaaaa", "Training networks", ML),
        "bread": add_text_item(conn, settings, embedder, "bbbbbbbbbbb", "Baking bread", BREAD),
        "sql": add_text_item(conn, settings, embedder, "ccccccccccc", "SQL joins", SQL),
    }


def fts_count(conn: sqlite3.Connection, word: str) -> int:
    return conn.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (f'"{word}"',)).fetchone()[0]


def integrity_check(conn: sqlite3.Connection) -> None:
    """FTS5 raises if the index and the `chunks` table disagree."""
    conn.execute("INSERT INTO chunks_fts (chunks_fts, rank) VALUES ('integrity-check', 1)")


# ---------- keeping the index in step with `chunks` ----------


def test_saved_chunks_are_indexed(conn: sqlite3.Connection, library_ids: dict[str, int]) -> None:
    chunk_count = len(items.load_chunks(conn, library_ids["bread"]))
    assert fts_count(conn, "sourdough") == chunk_count > 0
    integrity_check(conn)


def test_replaced_chunks_replace_the_index_entries(conn: sqlite3.Connection, library_ids: dict[str, int]) -> None:
    items.replace_chunks(conn, library_ids["bread"], [Chunk(0, "rye crackers baked crisp", 0.0, 5.0)])
    conn.commit()
    assert fts_count(conn, "sourdough") == 0
    assert fts_count(conn, "crackers") == 1
    integrity_check(conn)


def test_updated_chunk_text_is_reindexed(conn: sqlite3.Connection, library_ids: dict[str, int]) -> None:
    chunk_id = items.load_chunks(conn, library_ids["sql"])[0]["id"]
    conn.execute("UPDATE chunks SET text = 'window functions over partitions' WHERE id = ?", (chunk_id,))
    assert fts_count(conn, "partitions") == 1
    integrity_check(conn)


def test_deleting_an_item_removes_its_chunks_from_the_index(conn: sqlite3.Connection, library_ids: dict[str, int]) -> None:
    library.delete_item(conn, library_ids["bread"])  # chunks go through ON DELETE CASCADE
    assert fts_count(conn, "sourdough") == 0
    assert fts_count(conn, "gradient") > 0
    integrity_check(conn)


def test_existing_chunks_are_indexed_when_the_database_is_upgraded(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    all_migrations = db.MIGRATIONS
    monkeypatch.setattr(db, "MIGRATIONS", all_migrations[:4])  # a Phase 4-6 database
    old = db.connect(settings.db_path)
    item_id, _ = items.get_or_create(old, "youtube", "aaaaaaaaaaa", "https://www.youtube.com/watch?v=aaaaaaaaaaa")
    items.replace_chunks(old, item_id, [Chunk(0, "sourdough starter feeding schedule", 0.0, 5.0)])
    old.commit()
    old.close()

    monkeypatch.setattr(db, "MIGRATIONS", all_migrations)
    new = db.connect(settings.db_path)
    assert fts_count(new, "sourdough") == 1
    assert [hit.item_id for hit in bm25_items(new, "sourdough starter", limit=5)] == [item_id]
    new.close()


# ---------- turning a query into FTS5 syntax ----------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("sql joins", '"sql" OR "joins"'),
        ("SQL sql Joins", '"sql" OR "joins"'),  # lower-cased, duplicates dropped
        ('C++ "quoted" -minus col:on NOT AND', '"c" OR "quoted" OR "minus" OR "col" OR "on" OR "not" OR "and"'),
        ("తెలుగు text", '"తెలుగు" OR "text"'),  # unicode words are kept
        ("?!- ***", None),
    ],
)
def test_fts_query(query: str, expected: str | None) -> None:
    assert fts_query(query) == expected


def test_telugu_words_are_indexed_whole(conn: sqlite3.Connection) -> None:
    item_id, _ = items.get_or_create(conn, "article", "te", "https://example.com/te")
    items.replace_chunks(conn, item_id, [Chunk(0, "తెలుగు భాష చరిత్ర", None, None)])
    other, _ = items.get_or_create(conn, "article", "te2", "https://example.com/te2")
    items.replace_chunks(conn, other, [Chunk(0, "తల గల", None, None)])
    assert [h.item_id for h in bm25_items(conn, "తెలుగు", limit=5)] == [item_id]
    assert bm25_items(conn, "త", limit=5) == []  # a single letter is not a word of either text


def test_query_syntax_characters_do_not_break_search(conn: sqlite3.Connection, library_ids: dict[str, int]) -> None:
    hits = bm25_items(conn, 'sourdough AND "bread -NOT* NEAR(', limit=5)
    assert hits[0].item_id == library_ids["bread"]
    assert bm25_items(conn, "?!- ***", limit=5) == []


# ---------- BM25 mode ----------


def test_bm25_ranks_the_keyword_match_first(conn: sqlite3.Connection, embedder: FakeEmbedder, library_ids: dict[str, int]) -> None:
    hits = search(conn, embedder, "fermentation", mode="bm25")
    assert [h.item_id for h in hits] == [library_ids["bread"]]  # no other item has the word
    assert hits[0].bm25_rank == 1 and hits[0].dense_rank is None
    assert hits[0].score > 0  # higher = better, unlike raw bm25()
    assert "fermentation" in hits[0].passage


def test_bm25_stems_words(conn: sqlite3.Connection, embedder: FakeEmbedder, library_ids: dict[str, int]) -> None:
    hits = search(conn, embedder, "join", mode="bm25")  # the text says "joins"
    assert hits[0].item_id == library_ids["sql"]


def test_bm25_each_item_once_and_widens(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, library_ids: dict[str, int]) -> None:
    add_text_item(conn, settings, embedder, "ddddddddddd", "Long ML course", ML, repeat=400)
    hits = bm25_items(conn, "gradient descent", limit=2, chunk_k=3)
    assert len(hits) == 2 and len({h.item_id for h in hits}) == 2


# ---------- fusion ----------


def hit(item_id: int, *, passage: str = "", dense: int | None = None, bm25: int | None = None) -> SearchHit:
    return SearchHit(item_id, f"Item {item_id}", f"https://example.com/{item_id}", 0.0, passage, None, dense, bm25)


def test_rrf_scores_and_order() -> None:
    dense = [hit(1, dense=1), hit(2, dense=2), hit(3, dense=3)]
    keyword = [hit(3, bm25=1), hit(1, bm25=2), hit(4, bm25=3)]
    fused = fuse([dense, keyword], k=60)
    assert [h.item_id for h in fused] == [1, 3, 2, 4]  # found by both lists beats found by one
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].score == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    assert (fused[0].dense_rank, fused[0].bm25_rank) == (1, 2)
    assert (fused[3].dense_rank, fused[3].bm25_rank) == (None, 3)  # keyword-only items are kept


def test_rrf_passage_comes_from_the_list_that_ranked_the_item_higher() -> None:
    dense = [hit(9, dense=1), hit(1, passage="dense passage", dense=2)]
    keyword = [hit(1, passage="keyword passage", bm25=1)]
    assert next(h for h in fuse([dense, keyword]) if h.item_id == 1).passage == "keyword passage"
    tied = fuse([[hit(1, passage="first", dense=1)], [hit(1, passage="second", bm25=1)]])
    assert tied[0].passage == "first"  # a tie goes to the first list


def test_rrf_ties_are_broken_by_item_id() -> None:
    fused = fuse([[hit(7, dense=1)], [hit(3, bm25=1)]])
    assert [h.item_id for h in fused] == [3, 7]


def test_smaller_k_rewards_the_top_rank_more() -> None:
    ranking = [[hit(1, dense=1), hit(2, dense=2)]]
    flat, steep = fuse(ranking, k=60), fuse(ranking, k=1)
    assert flat[0].score / flat[1].score == pytest.approx(62 / 61)  # ranks 1 and 2 nearly equal
    assert steep[0].score / steep[1].score == pytest.approx(3 / 2)


# ---------- hybrid mode and the search() entry point ----------


def test_hybrid_puts_the_item_both_rankers_agree_on_first(conn: sqlite3.Connection, embedder: FakeEmbedder, library_ids: dict[str, int]) -> None:
    hits = search(conn, embedder, "sourdough bread fermentation", mode="hybrid")
    assert hits[0].item_id == library_ids["bread"]
    assert (hits[0].dense_rank, hits[0].bm25_rank) == (1, 1)
    assert hits[0].score == pytest.approx(2 / 61)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len({h.item_id for h in hits}) == len(hits)


def test_hybrid_respects_limit(conn: sqlite3.Connection, embedder: FakeEmbedder, library_ids: dict[str, int]) -> None:
    assert len(search(conn, embedder, "gradient descent", mode="hybrid", limit=2)) == 2


def test_hybrid_with_no_keyword_match_falls_back_to_the_dense_order(conn: sqlite3.Connection, embedder: FakeEmbedder, library_ids: dict[str, int]) -> None:
    query = "zzzz qqqq"  # no chunk contains these words
    assert bm25_items(conn, query, limit=5) == []
    dense = [h.item_id for h in search(conn, embedder, query, mode="dense", limit=3)]
    assert [h.item_id for h in search(conn, embedder, query, mode="hybrid", limit=3)] == dense


def test_default_mode_is_hybrid(conn: sqlite3.Connection, embedder: FakeEmbedder, library_ids: dict[str, int]) -> None:
    hit_ = search(conn, embedder, "sourdough bread")[0]
    assert hit_.dense_rank is not None and hit_.bm25_rank is not None
    assert Settings(_env_file=None).search_mode == "hybrid"


def test_empty_library_gives_no_results_in_every_mode(conn: sqlite3.Connection, embedder: FakeEmbedder) -> None:
    for mode in ("dense", "bm25", "hybrid"):
        assert search(conn, embedder, "anything", mode=mode) == []  # type: ignore[arg-type]


def test_bad_mode_and_k_are_rejected(conn: sqlite3.Connection, embedder: FakeEmbedder) -> None:
    with pytest.raises(SecondBrainError, match="Unknown search mode 'fuzzy'"):
        search(conn, embedder, "q", mode="fuzzy")  # type: ignore[arg-type]
    with pytest.raises(SecondBrainError, match="RRF k must be at least 1"):
        search(conn, embedder, "q", rrf_k=0)


def test_describe() -> None:
    h = SearchHit(1, "t", "u", 0.0328, "p", None, dense_rank=1, bm25_rank=None)
    assert describe(h, "hybrid") == "RRF 0.0328 (dense #1, BM25 -)"
    assert describe(SearchHit(1, "t", "u", 0.81, "p", None), "dense") == "similarity 0.81"
    assert describe(SearchHit(1, "t", "u", 4.2, "p", None), "bm25") == "BM25 4.20"
