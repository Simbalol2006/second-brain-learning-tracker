"""Eval label templates: sampling and file contents."""

import re
import sqlite3
from pathlib import Path

import pytest
import yaml
from conftest import FakeEmbedder, FakeProvider, add_text_item

from secondbrain import analyze, evalset
from secondbrain.config import Settings
from secondbrain.errors import SecondBrainError

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
ANSWER = '{"summary": "s", "key_points": ["a", "b"], "topics": ["letters"], "skills": [], "difficulty": "beginner"}'


def scored_pairs(n: int) -> list[tuple[int, int, float]]:
    """n pairs with distinct scores: pair (i, i+1000) has score i / n."""
    return [(i, i + 1000, i / n) for i in range(n)]


def test_sample_takes_top_middle_and_random() -> None:
    pairs = evalset.sample_pairs(scored_pairs(100), per_band=10, seed=1)
    firsts = {a for a, _ in pairs}
    assert len(pairs) == 30 and len(set(pairs)) == 30
    assert set(range(90, 100)) <= firsts  # the 10 highest scores
    assert len(firsts & set(range(40, 50))) == 10  # the middle of the rest
    assert len(firsts - set(range(90, 100)) - set(range(40, 50))) == 10  # random others


def test_sample_is_repeatable_and_shuffled() -> None:
    first = evalset.sample_pairs(scored_pairs(100), per_band=10, seed=1)
    assert first == evalset.sample_pairs(scored_pairs(100), per_band=10, seed=1)
    assert first != evalset.sample_pairs(scored_pairs(100), per_band=10, seed=2)
    assert [a for a, _ in first] != sorted((a for a, _ in first), reverse=True)  # order hides the score


def test_sample_takes_everything_when_there_are_few_pairs() -> None:
    assert len(evalset.sample_pairs(scored_pairs(12), per_band=10, seed=1)) == 12


@pytest.fixture
def library(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder) -> list[int]:
    ids = []
    for i in range(6):
        text = " ".join(WORDS[i : i + 3])
        ids.append(add_text_item(conn, settings, embedder, f"{chr(97 + i) * 11}", f"Talk: number {i} # extra", text))
    for item_id in ids[:4]:
        analyze.analyze_item(conn, settings, FakeProvider([ANSWER]), item_id)
    return ids


def test_templates_are_valid_yaml_with_urls(
    conn: sqlite3.Connection, settings: Settings, library: list[int], tmp_path: Path
) -> None:
    written = evalset.write_templates(conn, settings, tmp_path / "eval")
    assert [p.name for p in written] == ["retrieval.yaml", "links.yaml", "summaries.yaml"]

    retrieval = yaml.safe_load(written[0].read_text())
    assert retrieval == {"queries": []}
    assert written[0].read_text().count("https://www.youtube.com/watch?v=") == 6 + 1  # 6 items + the example

    pairs = yaml.safe_load(written[1].read_text())["pairs"]
    assert len(pairs) == 15  # all C(6,2) pairs, fewer than the 30 asked for
    assert all(p["related"] is None for p in pairs)
    assert all(p["a"].startswith("https://www.youtube.com/watch?v=") and p["a"] != p["b"] for p in pairs)

    summaries = yaml.safe_load(written[2].read_text())["items"]
    assert len(summaries) == 4  # only analyzed items can be judged
    assert all(s["spot_check"] is False for s in summaries)


def test_link_template_does_not_reveal_scores(
    conn: sqlite3.Connection, settings: Settings, library: list[int], tmp_path: Path
) -> None:
    written = evalset.write_templates(conn, settings, tmp_path)
    body = written[1].read_text().split("pairs:")[1]
    assert "score" not in body
    assert not re.search(r"\b0\.\d+", body)


def test_existing_labels_are_never_overwritten_silently(
    conn: sqlite3.Connection, settings: Settings, library: list[int], tmp_path: Path
) -> None:
    written = evalset.write_templates(conn, settings, tmp_path)
    written[1].write_text("pairs: my labels", encoding="utf-8")
    with pytest.raises(SecondBrainError, match="Not overwriting existing label files"):
        evalset.write_templates(conn, settings, tmp_path)
    assert written[1].read_text() == "pairs: my labels"
    evalset.write_templates(conn, settings, tmp_path, force=True)
    assert written[1].read_text() != "pairs: my labels"


def test_needs_at_least_two_items(conn: sqlite3.Connection, settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(SecondBrainError, match="at least two embedded items"):
        evalset.write_templates(conn, settings, tmp_path)
