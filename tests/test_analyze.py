"""Analysis tests with a scripted fake LLM."""

import json
import sqlite3

import pytest
from conftest import FakeProvider

from secondbrain import analyze, items, prompts
from secondbrain.config import Settings
from secondbrain.errors import AnalysisError, LLMError
from secondbrain.models import Metadata, Segment, Transcript
from secondbrain.schemas import Analysis

FINAL = json.dumps(
    {
        "summary": "It explains vector search.",
        "key_points": ["Embeddings map text to vectors.", "Cosine similarity compares them."],
        "topics": ["Vector Search", "embeddings"],
        "skills": ["semantic search"],
        "difficulty": "intermediate",
    }
)
NOTES = json.dumps({"summary": "Part summary.", "key_points": ["A point."]})


def add_item(conn: sqlite3.Connection, n_segments: int = 10, source_id: str = "vid00000001") -> int:
    """Insert an item that has finished Phase 1 (transcript stored, status embedded)."""
    item_id, _ = items.get_or_create(conn, "youtube", source_id, f"https://youtu.be/{source_id}")
    items.save_metadata(conn, item_id, Metadata(title="A talk", author="A channel"))
    segments = [Segment(f"sentence number {i} about vectors and search.", i, i + 1) for i in range(n_segments)]
    items.save_transcript(conn, item_id, Transcript(segments, "auto", "en"))
    items.set_status(conn, item_id, "embedded")
    conn.commit()
    return item_id


def llm_calls(conn: sqlite3.Connection) -> list[tuple[str, int, int]]:
    rows = conn.execute("SELECT purpose, attempt, valid FROM llm_calls ORDER BY id").fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def test_short_item_needs_one_call(conn: sqlite3.Connection, settings: Settings) -> None:
    item_id = add_item(conn)
    provider = FakeProvider([FINAL])
    result = analyze.analyze_item(conn, settings, provider, item_id)

    assert result.topics == ["vector search", "embeddings"]
    assert len(provider.calls) == 1
    prompt = provider.calls[0]["user"]
    assert "Title: A talk" in prompt and "sentence number 9" in prompt
    assert provider.calls[0]["system"] == prompts.SYSTEM
    assert provider.calls[0]["schema"]["required"]

    item = items.get(conn, item_id)
    assert (item["ingest_status"], item["error"]) == ("analyzed", None)
    row = items.load_analysis(conn, item_id)
    assert (row["provider"], row["model"], row["prompt_version"]) == ("fake", "fake-model", prompts.PROMPT_VERSION)
    assert items.load_tags(conn, item_id, "topic") == ["embeddings", "vector search"]
    assert items.load_tags(conn, item_id, "skill") == ["semantic search"]
    assert llm_calls(conn) == [("final", 1, 1)]
    assert [t["stage"] for t in items.load_timings(conn, item_id)] == ["analyze"]
    usage = items.llm_usage(conn, item_id)
    assert (usage["calls"], usage["input_tokens"], usage["output_tokens"]) == (1, 100, 20)


def test_stored_analysis_round_trips(conn: sqlite3.Connection, settings: Settings) -> None:
    item_id = add_item(conn)
    result = analyze.analyze_item(conn, settings, FakeProvider([FINAL]), item_id)
    stored = analyze.stored_analysis(conn, item_id)
    assert stored is not None
    assert stored.summary == result.summary
    assert stored.key_points == result.key_points
    assert set(stored.topics) == set(result.topics)
    assert analyze.stored_analysis(conn, 999) is None


def test_existing_analysis_is_reused_unless_forced(conn: sqlite3.Connection, settings: Settings) -> None:
    item_id = add_item(conn)
    analyze.analyze_item(conn, settings, FakeProvider([FINAL]), item_id)

    untouched = FakeProvider([])
    assert analyze.analyze_item(conn, settings, untouched, item_id).difficulty == "intermediate"
    assert untouched.calls == []

    changed = FINAL.replace("intermediate", "advanced").replace("Vector Search", "ann indexes")
    forced = analyze.analyze_item(conn, settings, FakeProvider([changed]), item_id, force=True)
    assert forced.difficulty == "advanced"
    assert items.load_tags(conn, item_id, "topic") == ["ann indexes", "embeddings"]  # old tag link removed


def test_invalid_answer_gets_one_retry_with_the_errors(conn: sqlite3.Connection, settings: Settings) -> None:
    item_id = add_item(conn)
    bad = FINAL.replace("intermediate", "expert")
    provider = FakeProvider([bad, FINAL])
    analyze.analyze_item(conn, settings, provider, item_id)

    assert llm_calls(conn) == [("final", 1, 0), ("final", 2, 1)]
    retry_prompt = provider.calls[1]["user"]
    assert retry_prompt.startswith(provider.calls[0]["user"])
    assert "previous answer was rejected" in retry_prompt
    assert "difficulty" in retry_prompt and "expert" in retry_prompt


def test_two_invalid_answers_fail_visibly(conn: sqlite3.Connection, settings: Settings) -> None:
    item_id = add_item(conn)
    with pytest.raises(AnalysisError, match="invalid output 2 times.*Invalid JSON"):
        analyze.analyze_item(conn, settings, FakeProvider(["not json", "still not json"]), item_id)
    item = items.get(conn, item_id)
    assert item["ingest_status"] == "failed"
    assert item["error"].startswith("[analyze] The model returned invalid output")
    assert items.load_analysis(conn, item_id) is None
    assert llm_calls(conn) == [("final", 1, 0), ("final", 2, 0)]  # the log survives the failure


def test_failed_item_can_be_analyzed_again(conn: sqlite3.Connection, settings: Settings) -> None:
    item_id = add_item(conn)
    with pytest.raises(AnalysisError):
        analyze.analyze_item(conn, settings, FakeProvider([LLMError("Ollama is not reachable")]), item_id)
    assert "not reachable" in items.get(conn, item_id)["error"]
    assert items.ids_without_analysis(conn) == [item_id]

    analyze.analyze_item(conn, settings, FakeProvider([FINAL]), item_id)
    assert items.get(conn, item_id)["ingest_status"] == "analyzed"
    assert items.ids_without_analysis(conn) == []


def test_item_without_transcript_is_refused(conn: sqlite3.Connection, settings: Settings) -> None:
    item_id, _ = items.get_or_create(conn, "youtube", "vid00000002", "u")
    with pytest.raises(AnalysisError, match="no transcript.*sb transcribe"):
        analyze.analyze_item(conn, settings, FakeProvider([]), item_id)
    assert items.get(conn, item_id)["ingest_status"] == "new"  # nothing was changed


def test_long_item_uses_map_reduce(conn: sqlite3.Connection, tmp_path: object) -> None:
    small = Settings(llm_max_input_tokens=1500, _env_file=None)  # about 4000 chars per window
    item_id = add_item(conn, n_segments=300)  # about 14000 chars
    provider = FakeProvider([NOTES, NOTES, NOTES, NOTES, FINAL])
    analyze.analyze_item(conn, small, provider, item_id)

    assert llm_calls(conn) == [("map", 1, 1)] * 4 + [("final", 1, 1)]
    first, last, final = provider.calls[0]["user"], provider.calls[3]["user"], provider.calls[4]["user"]
    assert "part 1 of 4" in first and "sentence number 0 " in first
    assert "part 4 of 4" in last and "sentence number 299 " in last
    assert "sentence number" not in final  # the final call sees notes, not the transcript
    assert "Part 1 summary: Part summary." in final and "Part 4 key points:" in final
    budget_chars = 1500 * 4
    assert all(len(c["system"]) + len(c["user"]) <= budget_chars for c in provider.calls)


def test_every_segment_lands_in_exactly_one_window(conn: sqlite3.Connection) -> None:
    small = Settings(llm_max_input_tokens=1500, _env_file=None)
    item_id = add_item(conn, n_segments=300)
    provider = FakeProvider([NOTES] * 4 + [FINAL])
    analyze.analyze_item(conn, small, provider, item_id)
    map_text = " ".join(c["user"] for c in provider.calls[:4])
    assert all(map_text.count(f"sentence number {i} ") == 1 for i in range(300))


def test_notes_are_collapsed_when_they_do_not_fit(conn: sqlite3.Connection) -> None:
    small = Settings(llm_max_input_tokens=1500, _env_file=None)
    item_id = add_item(conn, n_segments=300)
    big_notes = json.dumps({"summary": "s " * 900, "key_points": ["A point."]})  # ~1800 chars each
    provider = FakeProvider([big_notes] * 4 + [NOTES, NOTES, FINAL])
    analyze.analyze_item(conn, small, provider, item_id)
    purposes = [p for p, _, _ in llm_calls(conn)]
    assert purposes == ["map"] * 4 + ["collapse"] * 2 + ["final"]
    assert "Merge them" in provider.calls[4]["user"]


def test_too_small_budget_is_a_clear_error(conn: sqlite3.Connection) -> None:
    tiny = Settings(llm_max_input_tokens=1000, _env_file=None)
    item_id = add_item(conn)
    for i in range(40):  # a large tag vocabulary eats the remaining budget
        conn.execute("INSERT INTO topics (name) VALUES (?)", (f"a rather long topic name number {i}",))
        conn.execute("INSERT INTO item_topics VALUES (?, ?, 'topic')", (item_id, i + 1))
    conn.commit()
    with pytest.raises(AnalysisError, match="LLM_MAX_INPUT_TOKENS=1000 leaves no room"):
        analyze.analyze_item(conn, tiny, FakeProvider([]), item_id, force=True)


def test_existing_tags_are_offered_for_reuse(conn: sqlite3.Connection, settings: Settings) -> None:
    first = add_item(conn, source_id="vid00000001")
    second = add_item(conn, source_id="vid00000002")
    provider = FakeProvider([FINAL, FINAL])
    analyze.analyze_item(conn, settings, provider, first)
    analyze.analyze_item(conn, settings, provider, second)

    assert "Existing topics" not in provider.calls[0]["user"]  # nothing to reuse yet
    assert "Existing topics: embeddings, vector search" in provider.calls[1]["user"]
    assert "Existing skills: semantic search" in provider.calls[1]["user"]
    assert conn.execute("SELECT count(*) FROM topics").fetchone()[0] == 3  # shared, not duplicated


def test_pack_groups_and_splits() -> None:
    assert analyze.pack(["aaa", "bbb", "ccc"], budget_chars=8) == [["aaa", "bbb"], ["ccc"]]
    assert analyze.pack([], budget_chars=8) == []
    groups = analyze.pack(["one two three four five six"], budget_chars=10)  # one oversized text
    assert len(groups) > 1
    assert all(len(" ".join(g)) <= 10 for g in groups)
    assert " ".join(" ".join(g) for g in groups) == "one two three four five six"


def test_analysis_model_is_what_gets_validated() -> None:
    assert Analysis.model_validate_json(FINAL).difficulty == "intermediate"
