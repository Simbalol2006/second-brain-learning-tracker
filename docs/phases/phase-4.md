# Phase 4 — Profile, progress tracking, recommendations

Date: 2026-11-16 · Built and tested on CPython 3.12.3 (uv-managed), Linux, no GPU, no LLM access.

## What was built

- **Progress tracking** (`sb status <id> to_watch|watching|done`), shown in `sb list`.
- **Profile**: goals (`sb profile add-goal / remove-goal`), skill levels set by hand
  (`sb profile set-skill / remove-skill`), and skill levels derived from finished items.
  `sb profile show` prints all of it.
- **Marking an item done updates the profile**: its topic and skill tags gain evidence and their
  levels are recomputed. Undoing "done" undoes the effect.
- **Recommendations** (`sb recommend`): rule-based scoring of every unfinished, analyzed item,
  then an LLM step that orders the top ten and writes a one-line reason each. `--no-llm` skips
  the LLM. A missing or failing LLM gives the rule-based result plus a visible warning.
- **Database migration 4**: `profile_skills`, `profile_goals`. Existing databases upgrade in place.
- Refactor: the validate-retry-log LLM helper is now shared (`llm/validated.py`).

## Files

| File | Change |
|------|--------|
| `src/secondbrain/profile.py` | new — progress, goals, manual and derived skills |
| `src/secondbrain/recommend.py` | new — scoring, LLM step, fallback |
| `src/secondbrain/llm/validated.py` | new — `call_validated` (moved out of `analyze.py`) |
| `src/secondbrain/schemas.py` | `Recommendation`, `RecommendationList`; nested schemas written inline |
| `src/secondbrain/prompts.py` | recommendation prompt; retry text moved out |
| `src/secondbrain/analyze.py` | uses `call_validated`; refreshes skills after analysis |
| `src/secondbrain/db.py` | migration 4 |
| `src/secondbrain/errors.py` | `LLMOutputError` |
| `src/secondbrain/cli.py` | `status`, `recommend`, `profile ...`; progress column in `list` |
| `tests/test_profile.py`, `test_recommend.py` | new |
| `tests/test_schemas.py`, `test_cli.py`, `test_db.py` | extended |

## How to run

```bash
uv sync
uv run pytest
uv run sb profile add-goal "what you want to learn"
uv run sb profile set-skill python 2
uv run sb status 1 done
uv run sb profile show
uv run sb recommend            # needs analyzed items; uses the LLM from .env
uv run sb recommend --no-llm
```

## Test result

`uv run pytest` → **253 passed, 3 deselected** (the 3 deselected are the live tests).

## Not verified in the build environment — please run on the laptop

The scoring, the profile rules, the id validation and the fallback are covered by tests. What is
**not** verified: any real LLM recommendation, and whether the ranking is *useful* on a real
library with the real embedding model.

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 1 | Rule-based ranking on your items | `uv run sb recommend --no-llm` | goal-related, level-appropriate items near the top |
| 2 | LLM ordering and reasons | `uv run sb recommend` | reasons mention your goals/skills; no warning line |
| 3 | Profile update | `uv run sb status <id> done` then `uv run sb profile show` | the item's tags appear as skills; it no longer shows in `sb recommend` |
| 4 | Local model *(if you use Ollama)* | same as 2 with the Ollama `.env` | works, or falls back with a warning |

## How it connects to earlier phases

- Uses `items.progress_status` (column created in Phase 1).
- Uses Phase 2's tags and difficulty for skill levels, novelty and readiness; needs items to be
  analyzed (`sb analyze --all`).
- Uses Phase 1's item vectors and Phase 3's `embed_query` for goal fit.
- Uses the Phase 2 provider interface; every recommendation call is logged in `llm_calls`.
- Differences from DESIGN.md: goal fit is rescaled across candidates (D41); goals have no
  `active` flag (D44).

What Phase 5 will use: `profile.set_progress`, `profile.skills/goals`, `recommend.recommend`
(the UI calls the same functions as the CLI).

## Known limitations

- **Recommendation quality is unmeasured and not part of the eval plan.** The weights are
  starting values (D41). If you want a number for it, we would need to add a small labelled set
  in Phase 6 — tell me.
- Skill levels reflect what you finished, not what you can do (D37).
- A hand-set skill only matches items whose tags use the same name (D39).
- Only the ten best-scored items are shown to the LLM (D40).
- With goals set, the closest candidate always gets goal fit 1, even if nothing really matches.
- Items that are not analyzed are never recommended.
- No history of status changes beyond the "done" timestamp.
