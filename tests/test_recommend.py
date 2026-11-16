"""Recommendation tests: scoring, ordering, LLM step and its fallback."""

import json
import sqlite3

import pytest
from conftest import FakeEmbedder, FakeProvider, add_text_item

from secondbrain import items, profile, recommend
from secondbrain.config import Settings
from secondbrain.errors import LLMError
from secondbrain.schemas import Analysis

ML = "gradient descent trains neural network weights by reducing loss"
BREAD = "sourdough bread needs flour water salt and slow fermentation"
SQL = "relational database tables are combined with joins in sql queries"


def add(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, key: str, title: str, text: str,
        topics: list[str], difficulty: str) -> int:
    item_id = add_text_item(conn, settings, embedder, key * 11, title, text)
    analysis = Analysis(summary="s", key_points=["a", "b"], topics=topics, skills=[], difficulty=difficulty)
    items.save_analysis(conn, item_id, analysis, "fake", "fake-model", "v1")
    conn.commit()
    return item_id


@pytest.fixture
def library(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder) -> dict[str, int]:
    return {
        "ml": add(conn, settings, embedder, "a", "Neural nets", ML, ["machine learning"], "intermediate"),
        "bread": add(conn, settings, embedder, "b", "Baking", BREAD, ["baking"], "beginner"),
        "sql": add(conn, settings, embedder, "c", "SQL joins", SQL, ["sql"], "advanced"),
    }


def answer(*pairs: tuple[int, str]) -> str:
    return json.dumps({"recommendations": [{"item_id": i, "reason": r} for i, r in pairs]})


@pytest.mark.parametrize(
    ("rank", "level", "expected"),
    [(1, 0.0, 1.0), (2, 0.0, 2 / 3), (3, 0.0, 1 / 3), (3, 2.0, 1.0), (1, 2.0, 1 / 3), (1, 3.0, 0.0), (2, 1.0, 1.0)],
)
def test_readiness(rank: int, level: float, expected: float) -> None:
    assert recommend.readiness(rank, level) == pytest.approx(expected)


def test_without_goals_the_easiest_new_item_comes_first(
    conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]
) -> None:
    ranked = recommend.candidates(conn, lambda: embedder)
    assert [c.item_id for c in ranked] == [library["bread"], library["ml"], library["sql"]]
    first = ranked[0]
    assert (first.goal_fit, first.best_goal, first.novelty, first.readiness) == (0.0, None, 1.0, 1.0)
    assert first.score == pytest.approx(1.0)  # weights are rescaled when there are no goals
    assert first.new_tags == ["baking"]
    assert embedder.calls == 3  # no goal was embedded, only the three items at ingest


def test_a_goal_pulls_the_matching_item_up(
    conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]
) -> None:
    profile.add_goal(conn, "learn sql queries and database joins")
    profile.add_goal(conn, "run a marathon")
    ranked = recommend.candidates(conn, lambda: embedder)
    assert ranked[0].item_id == library["sql"]
    assert ranked[0].best_goal == "learn sql queries and database joins"
    assert ranked[0].goal_fit == pytest.approx(1.0)  # rescaled: closest candidate is 1 ...
    assert min(c.goal_fit for c in ranked) == pytest.approx(0.0)  # ... and the farthest is 0
    assert ranked[0].score == pytest.approx(0.5 * 1.0 + 0.2 * 1.0 + 0.3 * (1 / 3))


def test_known_skills_change_novelty_and_readiness(
    conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]
) -> None:
    profile.set_skill(conn, "sql", 2)
    by_id = {c.item_id: c for c in recommend.candidates(conn, lambda: embedder)}
    sql = by_id[library["sql"]]
    assert (sql.novelty, sql.new_tags) == (0.0, [])
    assert sql.readiness == pytest.approx(1.0)  # advanced is one step above intermediate


def test_done_and_unanalyzed_items_are_not_candidates(
    conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, library: dict[str, int]
) -> None:
    add_text_item(conn, settings, embedder, "d" * 11, "Not analyzed", "some other text entirely")
    profile.set_progress(conn, library["bread"], "done")
    assert sorted(c.item_id for c in recommend.candidates(conn, lambda: embedder)) == [library["ml"], library["sql"]]


def test_finishing_an_item_changes_what_comes_next(
    conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, library: dict[str, int]
) -> None:
    harder = add(conn, settings, embedder, "e", "Advanced baking", BREAD + " and lamination", ["baking"], "intermediate")
    before = {c.item_id: c for c in recommend.candidates(conn, lambda: embedder)}
    assert before[harder].readiness == pytest.approx(2 / 3)
    profile.set_progress(conn, library["bread"], "done")
    after = {c.item_id: c for c in recommend.candidates(conn, lambda: embedder)}
    assert after[harder].readiness == pytest.approx(1.0)  # now exactly one step up
    assert after[harder].novelty == 0.0


def test_rule_based_recommendations(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]) -> None:
    profile.add_goal(conn, "learn sql queries and database joins")
    result = recommend.recommend(conn, lambda: embedder, None, limit=2)
    assert (result.by_llm, result.warning) == (False, None)
    assert len(result.suggestions) == 2
    top = result.suggestions[0]
    assert top.candidate.item_id == library["sql"]
    assert top.reason == (
        'Closest to your goal "learn sql queries and database joins"; new for you: sql; '
        "advanced level is a stretch from where you are."
    )


def test_llm_order_and_reasons_are_used(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]) -> None:
    profile.add_goal(conn, "become a data engineer")
    profile.set_skill(conn, "python", 2)
    profile.set_progress(conn, library["bread"], "done")
    provider = FakeProvider([answer((library["sql"], "Core for data work."), (library["ml"], "Builds on it."))])
    result = recommend.recommend(conn, lambda: embedder, provider, limit=2)

    assert result.by_llm and result.warning is None
    assert [(s.candidate.item_id, s.reason) for s in result.suggestions] == [
        (library["sql"], "Core for data work."), (library["ml"], "Builds on it."),
    ]
    prompt = provider.calls[0]["user"]
    assert "Pick the 2 best items" in prompt
    assert "- become a data engineer" in prompt
    assert "- python (intermediate)" in prompt
    assert "RECENTLY FINISHED:\n- Baking" in prompt
    assert f"- id {library['sql']}: SQL joins | advanced, 10 min | topics: sql | skills: -" in prompt
    assert "Baking |" not in prompt  # finished items are not offered again
    row = conn.execute("SELECT purpose, item_id, valid FROM llm_calls").fetchone()
    assert tuple(row) == ("recommend", None, 1)


def test_unknown_id_gets_a_retry_then_succeeds(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]) -> None:
    provider = FakeProvider([answer((999, "Made up.")), answer((library["ml"], "Good next step."))])
    result = recommend.recommend(conn, lambda: embedder, provider, limit=1)
    assert result.by_llm and result.suggestions[0].candidate.item_id == library["ml"]
    assert "item_id not in the list of items: [999]" in provider.calls[1]["user"]


@pytest.mark.parametrize(
    "bad",
    [
        json.dumps({"recommendations": []}),
        "not json",
        json.dumps({"recommendations": [{"item_id": 1, "reason": "a"}, {"item_id": 1, "reason": "b"}]}),
        json.dumps({"recommendations": [{"item_id": i, "reason": "r"} for i in (1, 2, 3)]}),  # over the limit of 2
    ],
)
def test_bad_llm_answers_fall_back_with_a_warning(
    conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int], bad: str
) -> None:
    result = recommend.recommend(conn, lambda: embedder, FakeProvider([bad, bad]), limit=2)
    assert not result.by_llm
    assert result.warning is not None and "invalid output 2 times" in result.warning
    assert [s.candidate.item_id for s in result.suggestions] == [library["bread"], library["ml"]]
    assert all(s.reason.endswith(".") for s in result.suggestions)


def test_llm_outage_falls_back_with_a_warning(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]) -> None:
    provider = FakeProvider([LLMError("Ollama is not reachable")])
    result = recommend.recommend(conn, lambda: embedder, provider)
    assert not result.by_llm and "Ollama is not reachable" in (result.warning or "")
    assert len(result.suggestions) == 3


def test_nothing_to_recommend(conn: sqlite3.Connection, embedder: FakeEmbedder) -> None:
    result = recommend.recommend(conn, lambda: embedder, FakeProvider([]))
    assert result == recommend.Recommendations([], by_llm=False)


def test_goal_fit_of_a_single_candidate_is_one(
    conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int]
) -> None:
    profile.add_goal(conn, "anything at all")
    profile.set_progress(conn, library["ml"], "done")
    profile.set_progress(conn, library["sql"], "done")
    only = recommend.candidates(conn, lambda: embedder)
    assert [c.item_id for c in only] == [library["bread"]]
    assert only[0].goal_fit == 1.0
