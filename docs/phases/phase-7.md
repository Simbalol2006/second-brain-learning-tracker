# Phase 7 — Hybrid search

Date: 2026-11-23 · Built and tested on CPython 3.12 (uv-managed), Linux, no GPU, no LLM access,
and **without the real embedding model** (its download was blocked in the build environment;
all tests use the offline fake embedder).

## What was built

- **BM25 keyword search** over chunk text, using SQLite's built-in FTS5 (schema migration 5).
  The index is kept in step with `chunks` by triggers, and existing databases are indexed when
  they are upgraded. No re-ingest is needed.
- **Hybrid search**: the dense and BM25 rankings are merged with reciprocal rank fusion (RRF).
  Each result records where each ranker placed it.
- **Three modes**: `dense`, `bm25`, `hybrid` (default). Choose with `SEARCH_MODE` in `.env`,
  `sb search --mode`, or the selector on the Search page.
- **Evaluation**: the retrieval section now compares all three modes on the same labelled
  queries (recall@5, MRR, misses, and the first-relevant rank per mode for every query).
- **Indic-script fix**: words keep their combining marks in both the index and the query, so
  Telugu or Hindi words are not cut into single letters (D63).

Decisions D61–D66 in [DECISIONS.md](../DECISIONS.md).

## Files

| File | Change |
|------|--------|
| `src/secondbrain/db.py` | migration 5: `chunks_fts` + triggers + rebuild |
| `src/secondbrain/search.py` | modes, `bm25_items`, `fts_query`, `fuse` (RRF), `describe` |
| `src/secondbrain/config.py` | `SEARCH_MODE`, `RRF_K`, `SEARCH_DEPTH` |
| `src/secondbrain/cli.py` | `sb search --mode` |
| `app/streamlit_app.py` | mode selector on the Search page |
| `src/secondbrain/evaluation.py` | three-mode comparison; search settings in the run table |
| `tests/test_hybrid_search.py` | new: index sync, upgrade, query escaping, BM25, RRF, modes |
| `tests/test_cli.py`, `tests/test_app.py`, `tests/test_evaluation.py` | mode tests |
| `.env.example`, `README.md` | new settings and usage |

## Tests

`uv run pytest`: **365 passed** (offline), 3 live tests deselected.

## Not verified here — check on your machine

1. `uv run sb search "<a query>" --mode dense`, `--mode bm25` and with no flag, on your real
   library: the three lists should differ, and an exact term (a library name, an acronym) should
   rank higher in bm25 and hybrid than in dense.
2. Open the Search page and switch modes.
3. After labelling, run `scripts/eval.py` and read the "Search modes compared" table. Keep
   hybrid only if it is at least as good as dense; otherwise set `SEARCH_MODE=dense`.
4. Upgrading: the first command after pulling this phase applies migration 5 and indexes every
   existing chunk. On a few hundred items this should be quick, but it has not been timed.
