"""Ask an LLM for JSON, validate it, and retry once with the errors. Logs every call."""

import logging
import sqlite3
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from secondbrain import items
from secondbrain.errors import LLMOutputError
from secondbrain.llm.base import LLMProvider
from secondbrain.schemas import json_schema_for

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2

RETRY_SUFFIX = """

Your previous answer was rejected.
Previous answer: {previous}
Problems: {problems}
Answer again with corrected JSON only."""

ModelT = TypeVar("ModelT", bound=BaseModel)


def call_validated(
    conn: sqlite3.Connection,
    provider: LLMProvider,
    purpose: str,
    system: str,
    user: str,
    model: type[ModelT],
    *,
    item_id: int | None = None,
    context: dict[str, Any] | None = None,
) -> ModelT:
    """Return a validated `model` instance, or raise LLMOutputError after two bad answers.

    `context` is passed to Pydantic validation, for checks that need outside
    facts (for example, which item ids the model was allowed to pick).
    """
    schema = json_schema_for(model)
    prompt = user
    problems = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        response = provider.generate_json(system, prompt, schema)
        seconds = time.perf_counter() - start
        try:
            parsed: ModelT | None = model.model_validate_json(response.text, context=context)
        except ValidationError as exc:
            parsed, problems = None, describe(exc)
        items.add_llm_call(
            conn, item_id, purpose, provider.name, provider.model, attempt,
            parsed is not None, response.input_tokens, response.output_tokens, seconds,
        )
        conn.commit()  # the call log survives even if the caller rolls back later
        if parsed is not None:
            return parsed
        logger.warning("LLM step '%s' attempt %d returned invalid output: %s", purpose, attempt, problems)
        prompt = user + RETRY_SUFFIX.format(previous=response.text[:2000], problems=problems)
    raise LLMOutputError(
        f"The model returned invalid output {MAX_ATTEMPTS} times for step '{purpose}': {problems}"
    )


def describe(exc: ValidationError) -> str:
    """Short, model-readable list of validation problems."""
    parts = [f"{'.'.join(str(p) for p in err['loc']) or 'document'}: {err['msg']}" for err in exc.errors()]
    return "; ".join(parts)[:500]
