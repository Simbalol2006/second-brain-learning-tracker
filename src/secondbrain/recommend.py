"""What to learn next.

Step 1 (always, no LLM): score every unfinished, analyzed item on three things,
each between 0 and 1:

    goal fit   - how close the item is to your goals (embedding similarity,
                 rescaled so the closest candidate is 1 and the farthest is 0)
    novelty    - share of the item's tags you have no skill level in yet
    readiness  - how well its difficulty matches one step above your level

Step 2 (LLM, optional): the top candidates and the profile go to the LLM, which
picks an order and writes a one-line reason for each. If the LLM is unavailable
or answers badly, the step-1 order is used with rule-based reasons, and a
warning says so.
"""

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from secondbrain import items, profile, prompts, vectors
from secondbrain.embed import Embedder
from secondbrain.errors import LLMError
from secondbrain.llm.base import LLMProvider
from secondbrain.llm.validated import call_validated
from secondbrain.schemas import RecommendationList

logger = logging.getLogger(__name__)

# Starting values, not tuned on data. With no goals set, the other two are rescaled to sum to 1.
WEIGHT_GOAL = 0.5
WEIGHT_NOVELTY = 0.2
WEIGHT_READINESS = 0.3
NEUTRAL = 0.5  # used for novelty when an item has no tags


@dataclass(frozen=True)
class Candidate:
    item_id: int
    title: str
    url: str
    difficulty: str
    duration_sec: int | None
    progress_status: str
    topics: list[str]
    skills: list[str]
    goal_fit: float
    best_goal: str | None
    novelty: float
    new_tags: list[str]
    readiness: float
    score: float


@dataclass(frozen=True)
class Suggestion:
    candidate: Candidate
    reason: str


@dataclass(frozen=True)
class Recommendations:
    suggestions: list[Suggestion]
    by_llm: bool
    warning: str | None = None


def readiness(difficulty_rank: int, user_level: float) -> float:
    """1.0 when the item is one step above the user's level; lower the further away."""
    return max(0.0, 1.0 - abs(difficulty_rank - (user_level + 1.0)) / 3.0)


def candidates(conn: sqlite3.Connection, get_embedder: Callable[[], Embedder]) -> list[Candidate]:
    """Every unfinished, analyzed item with its scores, best first."""
    rows = conn.execute(
        """
        SELECT i.id, i.title, i.url, i.duration_sec, i.progress_status, a.difficulty
        FROM items i JOIN analyses a ON a.item_id = i.id
        WHERE i.progress_status != 'done' ORDER BY i.id
        """
    ).fetchall()
    if not rows:
        return []
    goal_rows = profile.goals(conn)
    goal_texts = [str(g["text"]) for g in goal_rows]
    goal_sims = _rescale(
        _goal_similarities(conn, get_embedder, goal_texts), [int(row["id"]) for row in rows]
    )
    levels = profile.skill_levels(conn)

    found = []
    for row in rows:
        item_id = int(row["id"])
        topics = items.load_tags(conn, item_id, "topic")
        skills = items.load_tags(conn, item_id, "skill")
        tag_levels = {name: levels.get(tag_id, 0) for name, tag_id in _item_tags(conn, item_id).items()}
        my_levels = list(tag_levels.values())
        names = sorted(tag_levels)
        new_tags = [name for name in names if tag_levels[name] == 0]
        fit, best_goal = goal_sims.get(item_id, (0.0, None))
        novelty = len(new_tags) / len(names) if names else NEUTRAL
        ready = readiness(profile.DIFFICULTY_RANK[row["difficulty"]], float(np.mean(my_levels)) if my_levels else 0.0)
        found.append(
            Candidate(
                item_id=item_id,
                title=row["title"] or row["url"],
                url=row["url"],
                difficulty=row["difficulty"],
                duration_sec=row["duration_sec"],
                progress_status=row["progress_status"],
                topics=topics,
                skills=skills,
                goal_fit=fit,
                best_goal=best_goal,
                novelty=novelty,
                new_tags=new_tags,
                readiness=ready,
                score=_combine(fit, novelty, ready, has_goals=bool(goal_texts)),
            )
        )
    return sorted(found, key=lambda c: (-c.score, c.item_id))


def recommend(
    conn: sqlite3.Connection,
    get_embedder: Callable[[], Embedder],
    provider: LLMProvider | None,
    *,
    limit: int = 5,
    pool: int = 10,
) -> Recommendations:
    """Suggest up to `limit` items. provider=None skips the LLM step on purpose."""
    ranked = candidates(conn, get_embedder)
    if not ranked:
        return Recommendations([], by_llm=False)
    fallback = [Suggestion(c, rule_based_reason(c)) for c in ranked[:limit]]
    if provider is None:
        return Recommendations(fallback, by_llm=False)

    shortlist = ranked[: max(pool, limit)]
    by_id = {c.item_id: c for c in shortlist}
    user = prompts.recommend_prompt(
        goals=[str(g["text"]) for g in profile.goals(conn)],
        skills=[f"{s['name']} ({profile.LEVEL_NAMES[s['level']]})" for s in profile.skills(conn)],
        finished=profile.finished_titles(conn),
        cards=[_card(c) for c in shortlist],
        limit=min(limit, len(shortlist)),
    )
    try:
        answer = call_validated(
            conn, provider, "recommend", prompts.RECOMMEND_SYSTEM, user, RecommendationList,
            context={"allowed_ids": sorted(by_id), "limit": limit},
        )
    except LLMError as exc:
        logger.warning("Recommendation LLM step failed, using rule-based order: %s", exc)
        warning = f"The LLM step failed ({exc}). Showing the rule-based order and reasons instead."
        return Recommendations(fallback, by_llm=False, warning=warning)
    picked = [Suggestion(by_id[r.item_id], r.reason) for r in answer.recommendations]
    return Recommendations(picked, by_llm=True)


def rule_based_reason(candidate: Candidate) -> str:
    """One-line reason built from the scores, used when no LLM reason is available."""
    parts = []
    if candidate.best_goal and candidate.goal_fit >= 0.5:
        parts.append(f"closest to your goal \"{candidate.best_goal}\"")
    if candidate.new_tags:
        parts.append(f"new for you: {', '.join(candidate.new_tags[:3])}")
    step = "a good next step" if candidate.readiness >= 0.99 else "within reach" if candidate.readiness >= 0.6 else "a stretch"
    parts.append(f"{candidate.difficulty} level is {step} from where you are")
    return "; ".join(parts).capitalize() + "."


def _combine(goal_fit: float, novelty: float, ready: float, has_goals: bool) -> float:
    if has_goals:
        return WEIGHT_GOAL * goal_fit + WEIGHT_NOVELTY * novelty + WEIGHT_READINESS * ready
    return (WEIGHT_NOVELTY * novelty + WEIGHT_READINESS * ready) / (WEIGHT_NOVELTY + WEIGHT_READINESS)


def _goal_similarities(
    conn: sqlite3.Connection, get_embedder: Callable[[], Embedder], goal_texts: list[str]
) -> dict[int, tuple[float, str]]:
    """item_id -> (similarity to its closest goal, that goal's text). Empty without goals."""
    ids, matrix = vectors.all_item_vectors(conn)
    if not goal_texts or not ids:
        return {}
    embedder = get_embedder()
    vectors.ensure_vector_tables(conn, embedder.model_name, embedder.dim)  # refuses a changed model
    goal_matrix = np.vstack([embedder.embed_query(text) for text in goal_texts])
    sims = np.clip(matrix @ goal_matrix.T, 0.0, 1.0)
    best = sims.argmax(axis=1)
    return {item_id: (float(sims[row, best[row]]), goal_texts[best[row]]) for row, item_id in enumerate(ids)}


def _rescale(
    sims: dict[int, tuple[float, str]], candidate_ids: list[int]
) -> dict[int, tuple[float, str]]:
    """Min-max rescale goal similarity over the candidates.

    Raw embedding similarities sit in a narrow band (unrelated texts still score
    well above 0), so on their own they would barely move the ranking.
    """
    present = {item_id: sims[item_id] for item_id in candidate_ids if item_id in sims}
    if not present:
        return {}
    values = [fit for fit, _ in present.values()]
    low, spread = min(values), max(values) - min(values)
    if spread == 0:
        return {item_id: (1.0, goal) for item_id, (_, goal) in present.items()}
    return {item_id: ((fit - low) / spread, goal) for item_id, (fit, goal) in present.items()}


def _item_tags(conn: sqlite3.Connection, item_id: int) -> dict[str, int]:
    """Tag name -> tag id for one item (topics and skills together)."""
    rows = conn.execute(
        "SELECT DISTINCT t.name, t.id FROM item_topics it JOIN topics t ON t.id = it.topic_id "
        "WHERE it.item_id = ?",
        (item_id,),
    ).fetchall()
    return {str(name): int(tag_id) for name, tag_id in rows}


def _card(candidate: Candidate) -> str:
    minutes = "?" if candidate.duration_sec is None else str(round(candidate.duration_sec / 60))
    started = ", already started" if candidate.progress_status == "watching" else ""
    return (
        f"id {candidate.item_id}: {candidate.title} | {candidate.difficulty}, {minutes} min{started} | "
        f"topics: {', '.join(candidate.topics) or '-'} | skills: {', '.join(candidate.skills) or '-'}"
    )

