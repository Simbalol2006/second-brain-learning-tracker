"""Schemas for LLM output. The LLM must return JSON that validates against these.

Limits (list sizes, text length) are checked in validators instead of JSON Schema
keywords like minItems/maxLength, because structured-output modes support only a
subset of JSON Schema. The schema sent to the model stays plain: types, enums,
required fields and descriptions.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

Difficulty = Literal["beginner", "intermediate", "advanced"]

MAX_SUMMARY_CHARS = 2000
MAX_KEY_POINTS = 10
MAX_TAGS = 8
MAX_TAG_CHARS = 60


def _clean_lines(values: list[str]) -> list[str]:
    """Trim whitespace, drop empties and duplicates, keep order."""
    seen: dict[str, None] = {}
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


def normalize_tag(name: str) -> str:
    """Canonical tag form: lower case, single spaces, no surrounding punctuation."""
    name = re.sub(r"[_\s]+", " ", str(name).lower())
    return name.strip(" .,;:!?\"'#")


def _clean_tags(values: list[str]) -> list[str]:
    tags = _clean_lines([normalize_tag(v) for v in values])
    too_long = [t for t in tags if len(t) > MAX_TAG_CHARS]
    if too_long:
        raise ValueError(f"tags must be short names (max {MAX_TAG_CHARS} chars): {too_long}")
    if len(tags) > MAX_TAGS:
        raise ValueError(f"at most {MAX_TAGS} tags allowed, got {len(tags)}")
    return tags


class WindowNotes(BaseModel):
    """Notes on one part of a long transcript (the 'map' step)."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="3-5 sentence summary of this part only.")
    key_points: list[str] = Field(description="3-6 key points from this part, one sentence each.")

    @field_validator("summary")
    @classmethod
    def _summary(cls, value: str) -> str:
        return _check_summary(value)

    @field_validator("key_points")
    @classmethod
    def _key_points(cls, value: list[str]) -> list[str]:
        return _check_key_points(value, minimum=1)


class Analysis(BaseModel):
    """The stored understanding of one item."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="3-5 sentence summary of what the content teaches.")
    key_points: list[str] = Field(
        description="3-7 most important takeaways, one sentence each, most important first."
    )
    topics: list[str] = Field(
        description="2-5 subject areas covered, as short lower-case names (1-3 words each)."
    )
    skills: list[str] = Field(
        description="0-5 practical abilities the content helps build, as short lower-case names."
    )
    difficulty: Difficulty = Field(
        description="beginner: no background needed. intermediate: assumes the basics of the "
        "field. advanced: assumes solid working experience or dense theory."
    )

    @field_validator("summary")
    @classmethod
    def _summary(cls, value: str) -> str:
        return _check_summary(value)

    @field_validator("key_points")
    @classmethod
    def _key_points(cls, value: list[str]) -> list[str]:
        return _check_key_points(value, minimum=2)

    @field_validator("topics")
    @classmethod
    def _topics(cls, value: list[str]) -> list[str]:
        tags = _clean_tags(value)
        if not tags:
            raise ValueError("at least one topic is required")
        return tags

    @field_validator("skills")
    @classmethod
    def _skills(cls, value: list[str]) -> list[str]:
        return _clean_tags(value)


def _check_summary(value: str) -> str:
    value = " ".join(value.split())
    if not value:
        raise ValueError("summary must not be empty")
    if len(value) > MAX_SUMMARY_CHARS:
        raise ValueError(f"summary is too long ({len(value)} chars, max {MAX_SUMMARY_CHARS})")
    return value


def _check_key_points(value: list[str], minimum: int) -> list[str]:
    points = _clean_lines(value)
    if not minimum <= len(points) <= MAX_KEY_POINTS:
        raise ValueError(f"need {minimum}-{MAX_KEY_POINTS} key points, got {len(points)}")
    return points


MAX_REASON_CHARS = 300


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: int = Field(description="The id of one of the listed items.")
    reason: str = Field(
        description="One sentence, at most 25 words: why this item, and why at this point."
    )

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("reason must not be empty")
        if len(value) > MAX_REASON_CHARS:
            raise ValueError(f"reason is too long ({len(value)} chars, max {MAX_REASON_CHARS})")
        return value


class RecommendationList(BaseModel):
    """Ordered study suggestions. Validation context may carry 'allowed_ids' and 'limit'."""

    model_config = ConfigDict(extra="forbid")

    recommendations: list[Recommendation] = Field(
        description="The chosen items, in the order they should be studied."
    )

    @model_validator(mode="after")
    def _check_ids(self, info: ValidationInfo) -> "RecommendationList":
        ids = [r.item_id for r in self.recommendations]
        context = info.context or {}
        if not ids:
            raise ValueError("at least one recommendation is required")
        if len(set(ids)) != len(ids):
            raise ValueError("an item_id appears more than once")
        allowed = context.get("allowed_ids")
        if allowed is not None and (unknown := sorted(set(ids) - set(allowed))):
            raise ValueError(f"item_id not in the list of items: {unknown}. Allowed ids: {allowed}")
        limit = context.get("limit")
        if limit is not None and len(ids) > limit:
            raise ValueError(f"at most {limit} recommendations allowed, got {len(ids)}")
        return self


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Plain JSON Schema for a model: no 'title' annotations, nested models written inline."""
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    return _strip_titles(_inline_refs(schema, definitions))


def _inline_refs(node: Any, definitions: dict[str, Any]) -> Any:
    """Replace {"$ref": "#/$defs/X"} with the definition of X."""
    if isinstance(node, dict):
        if "$ref" in node:
            name = str(node["$ref"]).rsplit("/", 1)[-1]
            return _inline_refs(definitions[name], definitions)
        return {key: _inline_refs(value, definitions) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline_refs(value, definitions) for value in node]
    return node


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: _strip_titles(value)
            for key, value in node.items()
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(node, list):
        return [_strip_titles(value) for value in node]
    return node
