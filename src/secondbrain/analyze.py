"""Understand an item: summary, key points, topics, skills, difficulty.

Short content needs one LLM call. Content longer than the input budget is
handled map-reduce style: notes per window ("map"), notes merged if they are
still too long ("collapse"), then one final call that produces the analysis.
Every answer is validated against a Pydantic schema; an invalid answer gets
one retry that shows the model its mistakes.
"""

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from secondbrain import items, prompts
from secondbrain.config import Settings
from secondbrain.errors import AnalysisError, SecondBrainError
from secondbrain.llm.base import LLMProvider
from secondbrain.schemas import Analysis, WindowNotes, json_schema_for

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # rough rule for English; there is no tokenizer in the app
BUDGET_SAFETY = 0.9  # keep 10% spare because the rule above is rough
HEADER_RESERVE_CHARS = 500  # title, author and section labels
MIN_WINDOW_CHARS = 1000
VOCABULARY_SIZE = 40
MAX_ATTEMPTS = 2

ModelT = TypeVar("ModelT", bound=BaseModel)
Caller = Callable[[str, str, type[ModelT]], ModelT]


def analyze_item(
    conn: sqlite3.Connection,
    settings: Settings,
    provider: LLMProvider,
    item_id: int,
    *,
    force: bool = False,
) -> Analysis:
    """Analyze one item and store the result. Returns the stored analysis.

    An item that already has an analysis is not sent to the LLM again unless
    force=True. On failure the item is marked 'failed' and AnalysisError is raised.
    """
    item = items.get(conn, item_id)
    segments = items.load_segments(conn, item_id)
    if segments is None:
        raise AnalysisError(
            f"Item {item_id} has no transcript to analyze. Try: sb transcribe {item_id}"
        )
    if not force:
        existing = stored_analysis(conn, item_id)
        if existing is not None:
            logger.info("Item %d is already analyzed; use force to redo it", item_id)
            return existing

    start = time.perf_counter()
    try:
        analysis = _run(conn, settings, provider, item, [s.text for s in segments])
        items.save_analysis(conn, item_id, analysis, provider.name, provider.model, prompts.PROMPT_VERSION)
        items.set_status(conn, item_id, "analyzed")
        items.add_timing(conn, item_id, "analyze", time.perf_counter() - start)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        items.set_status(conn, item_id, "failed", error=f"[analyze] {exc}")
        conn.commit()
        if isinstance(exc, SecondBrainError):
            logger.error("Item %d: analysis failed: %s", item_id, exc)
        else:
            logger.exception("Item %d: analysis failed with an unexpected error", item_id)
        raise AnalysisError(f"item {item_id} failed at stage 'analyze': {exc}") from exc
    logger.info("Item %d analyzed in %.1fs", item_id, time.perf_counter() - start)
    return analysis


def stored_analysis(conn: sqlite3.Connection, item_id: int) -> Analysis | None:
    row = items.load_analysis(conn, item_id)
    if row is None:
        return None
    return Analysis(
        summary=row["summary"],
        key_points=json.loads(row["key_points_json"]),
        topics=items.load_tags(conn, item_id, "topic"),
        skills=items.load_tags(conn, item_id, "skill"),
        difficulty=row["difficulty"],
    )


def pack(texts: list[str], budget_chars: int) -> list[list[str]]:
    """Group consecutive texts so each group fits the budget. Oversized texts are split."""
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in (piece for t in texts for piece in _split_oversized(t, budget_chars)):
        if current and size + len(text) + 1 > budget_chars:
            groups.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text) + 1
    if current:
        groups.append(current)
    return groups


def _split_oversized(text: str, budget_chars: int) -> list[str]:
    if len(text) <= budget_chars:
        return [text]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in text.split():
        if current and size + len(word) + 1 > budget_chars:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(word[:budget_chars])
        size += len(word) + 1
    pieces.append(" ".join(current))
    return pieces


def _run(
    conn: sqlite3.Connection,
    settings: Settings,
    provider: LLMProvider,
    item: sqlite3.Row,
    texts: list[str],
) -> Analysis:
    title, author = item["title"], item["author"]
    vocab = prompts.vocabulary(
        items.tag_vocabulary(conn, "topic", VOCABULARY_SIZE),
        items.tag_vocabulary(conn, "skill", VOCABULARY_SIZE),
    )
    budget = _window_budget_chars(settings, vocab)

    def call(purpose: str, user: str, model: type[ModelT]) -> ModelT:
        return _call_validated(conn, provider, int(item["id"]), purpose, user, model)

    windows = [" ".join(group) for group in pack(texts, budget)]
    if len(windows) == 1:
        label, content = "CONTENT", windows[0]
    else:
        logger.info("Item %d is long: %d windows (map-reduce)", item["id"], len(windows))
        notes = [
            _format_notes(i, call("map", prompts.map_prompt(i, len(windows), title, author, w), WindowNotes))
            for i, w in enumerate(windows, start=1)
        ]
        notes = _collapse(call, notes, budget)
        label, content = "NOTES ON CONSECUTIVE PARTS OF THE CONTENT", "\n\n".join(notes)
    return call("final", prompts.final_prompt(title, author, vocab, label, content), Analysis)


def _window_budget_chars(settings: Settings, vocab: str) -> int:
    """How many characters of content fit in one prompt, after the fixed prompt text."""
    total = int(settings.llm_max_input_tokens * CHARS_PER_TOKEN * BUDGET_SAFETY)
    fixed = len(prompts.SYSTEM) + len(prompts.FINAL_TASK) + len(vocab) + HEADER_RESERVE_CHARS
    budget = total - fixed
    if budget < MIN_WINDOW_CHARS:
        raise AnalysisError(
            f"LLM_MAX_INPUT_TOKENS={settings.llm_max_input_tokens} leaves no room for content "
            f"after the fixed prompt text ({fixed} chars). Raise it."
        )
    return budget


def _collapse(call: Caller, notes: list[str], budget_chars: int) -> list[str]:
    """Merge notes in rounds until all of them fit into one final prompt."""
    while sum(len(n) + 2 for n in notes) > budget_chars:
        groups = pack(notes, budget_chars)
        if len(groups) >= len(notes):
            raise AnalysisError(
                "The notes for this item do not fit the input budget even after merging. "
                "Raise LLM_MAX_INPUT_TOKENS."
            )
        logger.info("Collapsing %d notes into %d", len(notes), len(groups))
        notes = [
            group[0]
            if len(group) == 1
            else _format_notes(i, call("collapse", prompts.collapse_prompt(group), WindowNotes))
            for i, group in enumerate(groups, start=1)
        ]
    return notes


def _format_notes(part: int, notes: WindowNotes) -> str:
    points = "\n".join(f"- {point}" for point in notes.key_points)
    return f"Part {part} summary: {notes.summary}\nPart {part} key points:\n{points}"


def _call_validated(
    conn: sqlite3.Connection,
    provider: LLMProvider,
    item_id: int,
    purpose: str,
    user: str,
    model: type[ModelT],
) -> ModelT:
    """Ask the LLM, validate the JSON, retry once with the errors. Logs every call."""
    schema = json_schema_for(model)
    prompt = user
    problems = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        response = provider.generate_json(prompts.SYSTEM, prompt, schema)
        seconds = time.perf_counter() - start
        try:
            parsed: ModelT | None = model.model_validate_json(response.text)
        except ValidationError as exc:
            parsed, problems = None, _describe(exc)
        items.add_llm_call(
            conn, item_id, purpose, provider.name, provider.model, attempt,
            parsed is not None, response.input_tokens, response.output_tokens, seconds,
        )
        conn.commit()  # the call log survives even if the analysis fails later
        if parsed is not None:
            return parsed
        logger.warning("Item %d: '%s' attempt %d returned invalid output: %s", item_id, purpose, attempt, problems)
        prompt = user + prompts.retry_suffix(response.text, problems)
    raise AnalysisError(
        f"The model returned invalid output {MAX_ATTEMPTS} times for step '{purpose}': {problems}"
    )


def _describe(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in err['loc']) or 'document'}: {err['msg']}" for err in exc.errors()]
    return "; ".join(parts)[:500]
