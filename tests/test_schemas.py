import json

import pytest
from pydantic import ValidationError

from secondbrain.schemas import Analysis, WindowNotes, json_schema_for, normalize_tag

VALID = {
    "summary": "It explains vector search.",
    "key_points": ["Embeddings map text to vectors.", "Cosine similarity compares them."],
    "topics": ["Vector Search", "embeddings"],
    "skills": ["semantic search"],
    "difficulty": "beginner",
}


def test_valid_analysis_is_accepted_and_tags_normalized() -> None:
    analysis = Analysis.model_validate_json(json.dumps(VALID))
    assert analysis.topics == ["vector search", "embeddings"]
    assert analysis.difficulty == "beginner"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Machine   Learning ", "machine learning"),
        ("RAG.", "rag"),
        ("prompt_design", "prompt design"),
        ('"Fine-Tuning"', "fine-tuning"),
        ("#python", "python"),
    ],
)
def test_normalize_tag(raw: str, expected: str) -> None:
    assert normalize_tag(raw) == expected


def test_duplicate_and_empty_entries_are_dropped() -> None:
    data = {**VALID, "topics": ["RAG", "rag ", "", "Rag."], "key_points": ["a", " a ", "b", ""]}
    analysis = Analysis.model_validate(data)
    assert analysis.topics == ["rag"]
    assert analysis.key_points == ["a", "b"]


def test_skills_may_be_empty() -> None:
    assert Analysis.model_validate({**VALID, "skills": []}).skills == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"summary": "   "}, "summary must not be empty"),
        ({"summary": "x" * 2001}, "summary is too long"),
        ({"key_points": ["only one"]}, "need 2-10 key points"),
        ({"key_points": [f"p{i}" for i in range(11)]}, "need 2-10 key points"),
        ({"topics": []}, "at least one topic"),
        ({"topics": [f"t{i}" for i in range(9)]}, "at most 8 tags"),
        ({"topics": ["x" * 61]}, "short names"),
        ({"difficulty": "expert"}, "beginner"),
        ({"extra_field": 1}, "Extra inputs are not permitted"),
    ],
)
def test_invalid_analysis_is_rejected(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Analysis.model_validate({**VALID, **change})


def test_missing_field_is_rejected() -> None:
    data = {k: v for k, v in VALID.items() if k != "difficulty"}
    with pytest.raises(ValidationError, match="difficulty"):
        Analysis.model_validate(data)


def test_not_json_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Invalid JSON"):
        Analysis.model_validate_json("Sure! Here is the summary...")


def test_window_notes_need_summary_and_one_point() -> None:
    assert WindowNotes.model_validate({"summary": "s", "key_points": ["p"]}).key_points == ["p"]
    with pytest.raises(ValidationError):
        WindowNotes.model_validate({"summary": "s", "key_points": []})


def test_json_schema_is_plain_and_strict() -> None:
    schema = json_schema_for(Analysis)
    text = json.dumps(schema)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"summary", "key_points", "topics", "skills", "difficulty"}
    assert schema["properties"]["difficulty"]["enum"] == ["beginner", "intermediate", "advanced"]
    assert '"title"' not in text
    for keyword in ("minItems", "maxItems", "minLength", "maxLength", "$ref"):
        assert keyword not in text
