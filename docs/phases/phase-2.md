# Phase 2 — Understand (summary, tags, difficulty)

Date: 2026-11-16 · Built and tested on CPython 3.12.3 (uv-managed), Linux, no GPU, no LLM access.

## What was built

For every saved item with a transcript, an LLM produces a **summary, key points, topic tags,
skill tags and a difficulty level** as JSON, which is validated against a schema and stored.

- **Provider interface** (`LLMProvider`) with two backends: **Anthropic API** and **Ollama**.
  The backend is chosen by `LLM_PROVIDER` in `.env`; nothing else in the app calls an LLM.
- **Schemas** (`Analysis`, `WindowNotes`) in Pydantic, with tag normalization and size limits.
- **Prompts** in one file, versioned.
- **Analysis pipeline**: one call for short content; map → collapse → final for long content;
  one retry on invalid output; failures recorded on the item.
- **Database migration 2**: `analyses`, `topics`, `item_topics`, `llm_calls`. An existing
  Phase 1 database is upgraded in place the next time any command runs.
- **CLI**: `sb analyze <id>`, `sb analyze --all`, `--force`, `sb ingest --analyze`; `sb show`
  now prints the analysis.
- Articles now get an estimated reading time (shown in the `min` column of `sb list`).

## Files

| File | Change |
|------|--------|
| `src/secondbrain/llm/base.py` | new — `LLMProvider` protocol, `LLMResponse` |
| `src/secondbrain/llm/anthropic_provider.py` | new — Anthropic backend |
| `src/secondbrain/llm/ollama_provider.py` | new — Ollama backend |
| `src/secondbrain/llm/__init__.py` | new — `build_provider(settings)` |
| `src/secondbrain/schemas.py` | new — LLM output schemas, tag normalization |
| `src/secondbrain/prompts.py` | new — all prompt text, `PROMPT_VERSION` |
| `src/secondbrain/analyze.py` | new — analysis pipeline |
| `src/secondbrain/db.py` | migration 2 |
| `src/secondbrain/items.py` | SQL for analyses, tags, LLM call log |
| `src/secondbrain/config.py` | LLM settings |
| `src/secondbrain/errors.py` | `LLMError`, `AnalysisError` |
| `src/secondbrain/cli.py` | `analyze` command, `--analyze` flag, analysis in `show` |
| `src/secondbrain/ingest/article.py` | reading-time estimate |
| `pyproject.toml`, `uv.lock` | added `anthropic`, `httpx` |
| `.env.example`, `README.md` | LLM configuration |
| `tests/test_schemas.py`, `test_analyze.py`, `test_llm_providers.py` | new |
| `tests/test_cli.py`, `test_db.py`, `test_config.py`, `test_article.py`, `test_live.py`, `conftest.py` | extended |

## How to run

```bash
# .env: set ANTHROPIC_API_KEY=...   (or switch to Ollama, see README "The LLM")
uv sync
uv run pytest                 # offline tests
uv run sb analyze --all       # analyze everything saved in Phase 1
uv run sb show 1              # see summary, key points, tags, difficulty
```

## Test result

`uv run pytest` → **156 passed, 3 deselected** (the 3 deselected are the live tests).

## Not verified in the build environment — please run on the laptop

There is no API key, no Ollama and no internet for these in the build environment. Both providers
are tested only against fakes that mimic the documented request and response shapes. **No real
LLM call has been made yet.**

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 1 | Real LLM call with your `.env` provider | `uv run pytest -m live -s` | LLM test passes and prints the JSON it got (one small paid call on Anthropic) |
| 2 | Analyze a real saved video | `uv run sb analyze 1` then `uv run sb show 1` | status `analyzed`; summary matches the video |
| 3 | Long video (over ~1 hour) | `uv run sb ingest "<url>" --analyze` | log line `is long: N windows (map-reduce)`; still `analyzed` |
| 4 | Ollama *(if you want local)* | set the four `.env` lines from the README, then `uv run sb analyze 1 --force` | `analyzed`; note the time and `nvidia-smi` VRAM |

If check 1 fails on Anthropic with an HTTP 400, send me the error text: the request shape
(`output_config.format`, no sampling parameters) comes from the documentation and is the most
likely thing to need a fix.

What *was* observed here with the real CLI: with no key it prints
`ANTHROPIC_API_KEY is not set...` and exits 1; with `LLM_PROVIDER=ollama` and no server it prints
`Ollama is not reachable at http://localhost:11434. Start it with: ollama serve`, and the item
shows as `failed` in `sb list`.

## How it connects to earlier phases

- Reads the raw transcript stored by Phase 1 (`items.load_segments`), not the overlapping chunks.
- Moves `ingest_status` from `embedded` to `analyzed`, and writes an `analyze` row into
  `ingest_timings` next to the Phase 1 stage timings.
- Implements DESIGN.md §5.1 step 6 and the "Extra" eval metrics of §11 (`llm_calls`).
- Differences from the design: analysis is a separate command rather than automatic in ingest
  (D25); `analyses` also stores the provider name; `llm_calls` is a new table (D24).

What later phases will use: `item_topics` (Phase 3 linking by shared topics; Phase 4 profile
skills), `analyses.difficulty` (Phase 4 ordering), `build_provider` (Phase 4 reasons, Phase 6
judge), `llm_calls` (Phase 6 metrics).

## Known limitations

- **No real LLM call verified** — see the table above.
- Output quality is unmeasured until the Phase 6 faithfulness eval.
- Token budgets use a 4-characters-per-token estimate, not a tokenizer.
- Very long content is summarized from notes, so fine detail is lost.
- Tag reuse depends on the model; near-duplicate tags can appear (D23).
- The Ollama model in the README is an untested example; thinking-style local models are not
  handled specially.
- Because an analysis failure sets status `failed`, a failed item that is already embedded still
  has its chunks and vectors; later phases select by stored data, not by status.
- Calls are sequential; `sb analyze --all` on many items takes a while.
