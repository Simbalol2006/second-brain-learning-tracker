"""Search tests with the fake (bag-of-words) embedder."""

import sqlite3

import pytest
from conftest import FakeEmbedder, add_text_item

from secondbrain import items
from secondbrain.config import Settings
from secondbrain.errors import SecondBrainError, StorageError
from secondbrain.search import SearchHit, search

ML = "gradient descent updates neural network weights to reduce training loss"
BREAD = "sourdough bread needs flour water salt and a long slow fermentation"
SQL = "relational database tables are combined with joins in sql queries"


@pytest.fixture
def library(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder) -> dict[str, int]:
    return {
        "ml": add_text_item(conn, settings, embedder, "aaaaaaaaaaa", "Training networks", ML),
        "bread": add_text_item(conn, settings, embedder, "bbbbbbbbbbb", "Baking bread", BREAD),
        "sql": add_text_item(conn, settings, embedder, "ccccccccccc", "SQL joins", SQL),
    }


def test_best_match_comes_first(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]) -> None:
    hits = search(conn, embedder, "sourdough bread fermentation")
    assert hits[0].item_id == library["bread"]
    assert hits[0].title == "Baking bread"
    assert "sourdough" in hits[0].passage
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert 0.0 < hits[0].score <= 1.0


def test_each_item_appears_once(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]) -> None:
    hits = search(conn, embedder, "sql joins database", limit=10)
    assert len(items.load_chunks(conn, library["sql"])) > 1  # several chunks could match
    assert sorted(h.item_id for h in hits) == sorted(library.values())


def test_limit_is_respected(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]) -> None:
    assert len(search(conn, embedder, "gradient descent", limit=2)) == 2
    assert len(search(conn, embedder, "gradient descent", limit=1)) == 1


def test_search_widens_when_one_item_has_all_the_nearest_chunks(
    conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, library: dict[str, int]
) -> None:
    add_text_item(conn, settings, embedder, "ddddddddddd", "Long ML course", ML, repeat=400)
    hits = search(conn, embedder, "gradient descent neural network", limit=4, chunk_k=3)
    assert len(hits) == 4  # found all four items although the first 3 chunks are from one


def test_link_jumps_to_the_passage(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]) -> None:
    hit = search(conn, embedder, "sourdough bread")[0]
    assert hit.start_sec is not None
    assert hit.link == f"https://www.youtube.com/watch?v=bbbbbbbbbbb&t={int(hit.start_sec)}s"


def test_article_link_is_the_plain_url() -> None:
    hit = SearchHit(1, "Post", "https://example.com/post", 0.9, "text", None)
    assert hit.link == "https://example.com/post"


def test_empty_library_gives_no_results(conn: sqlite3.Connection, embedder: FakeEmbedder) -> None:
    assert search(conn, embedder, "anything") == []


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(conn: sqlite3.Connection, embedder: FakeEmbedder, query: str) -> None:
    with pytest.raises(SecondBrainError, match="query is empty"):
        search(conn, embedder, query)


def test_searching_with_a_different_model_is_refused(
    conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]
) -> None:
    other = FakeEmbedder()
    other.model_name = "another-model"
    with pytest.raises(StorageError, match="cannot be mixed"):
        search(conn, other, "gradient descent")
