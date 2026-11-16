import json

import pytest
from pydantic import ValidationError

from secondbrain.schemas import Analysis, RecommendationList, WindowNotes, json_schema_for, normalize_tag

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


RECS = {"recommendations": [{"item_id": 3, "reason": "  Builds on\nwhat you finished. "}, {"item_id": 5, "reason": "Next."}]}


def test_recommendations_are_cleaned_and_checked_against_context() -> None:
    parsed = RecommendationList.model_validate(RECS, context={"allowed_ids": [3, 5, 8], "limit": 2})
    assert parsed.recommendations[0].reason == "Builds on what you finished."
    assert RecommendationList.model_validate(RECS).recommendations[1].item_id == 5  # no context: ids unchecked


@pytest.mark.parametrize(
    ("data", "context", "message"),
    [
        (RECS, {"allowed_ids": [3]}, r"item_id not in the list of items: \[5\]"),
        (RECS, {"limit": 1}, "at most 1 recommendations"),
        ({"recommendations": []}, {}, "at least one recommendation"),
        ({"recommendations": [RECS["recommendations"][0]] * 2}, {}, "more than once"),
        ({"recommendations": [{"item_id": 3, "reason": " "}]}, {}, "reason must not be empty"),
        ({"recommendations": [{"item_id": 3, "reason": "x" * 301}]}, {}, "reason is too long"),
        ({"recommendations": [{"item_id": "three", "reason": "r"}]}, {}, "valid integer"),
    ],
)
def test_invalid_recommendations_are_rejected(data: dict, context: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RecommendationList.model_validate(data, context=context)


def test_nested_schema_is_written_inline() -> None:
    schema = json_schema_for(RecommendationList)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text and '"title"' not in text
    item = schema["properties"]["recommendations"]["items"]
    assert item["properties"]["item_id"]["type"] == "integer"
    assert item["additionalProperties"] is False and set(item["required"]) == {"item_id", "reason"}
