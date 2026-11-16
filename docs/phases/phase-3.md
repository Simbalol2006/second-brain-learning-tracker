# Phase 3 — Search, links, graph data

Date: 2026-11-16 · Built and tested on CPython 3.12.3 (uv-managed), Linux, no GPU.

## What was built

- **Semantic search** (`sb search "<query>"`): query → embedding → nearest chunks → one result
  per item, with the best-matching passage and a link that opens the video at that moment.
- **Related-item links**: a score from text similarity and shared tags; pairs above a threshold
  are stored. `sb related <id>` lists them with both parts of the score.
- **Automatic link updates** at the end of ingest and after analysis; `sb links --rebuild` for a
  full recompute and `sb links --stats` to see the score distribution.
- **Graph data** (`sb graph`): nodes and edges as JSON, ready for the Phase 5 graph view.
- **Eval label templates** (`sb eval-templates`) plus `eval/README.md` with labelling
  instructions, delivered now so labelling can happen while Phases 4-5 are built.
- **Database migration 3**: `links`. Existing databases upgrade in place.

## Files

| File | Change |
|------|--------|
| `src/secondbrain/search.py` | new — search |
| `src/secondbrain/links.py` | new — scoring, link storage, related items, graph, stats |
| `src/secondbrain/evalset.py` | new — pair sampling and label templates |
| `eval/README.md` | new — how to label |
| `src/secondbrain/vectors.py` | `all_item_vectors()` |
| `src/secondbrain/embed.py` | `embed_query()` |
| `src/secondbrain/items.py` | `all_item_tags()` |
| `src/secondbrain/db.py` | migration 3 |
| `src/secondbrain/config.py` | `LINK_EMBED_WEIGHT`, `LINK_THRESHOLD` |
| `src/secondbrain/ingest/pipeline.py`, `analyze.py` | call `links.update_item()` |
| `src/secondbrain/cli.py` | `search`, `related`, `links`, `graph`, `eval-templates` |
| `pyproject.toml`, `uv.lock` | added `pyyaml` |
| `tests/test_search.py`, `test_links.py`, `test_evalset.py` | new |
| `tests/test_cli.py`, `test_pipeline.py`, `test_db.py`, `test_live.py`, `conftest.py` | extended |

## How to run

```bash
uv sync
uv run pytest
uv run sb links --rebuild            # once, if you saved items before this phase
uv run sb search "your question"
uv run sb related 1
uv run sb links --stats
uv run sb graph --out graph.json
```

## Test result

`uv run pytest` → **195 passed, 3 deselected** (the 3 deselected are the live tests).

## Not verified in the build environment — please run on the laptop

Search and link *logic* is tested with a fake embedder and hand-made vectors, so the formulas,
ranking, grouping and storage are verified. What is **not** verified is how the real embedding
model behaves: whether search results are good and what link scores look like. No search-quality
or link-quality number exists yet; those come from the Phase 6 eval.

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 1 | Real query embedding | `uv run pytest -m live -s` | embedding test passes |
| 2 | Search on your items | `uv run sb search "<something one of your videos covers>"` | that video near the top; the link opens at the right moment |
| 3 | Score distribution | `uv run sb links --stats` | send me this output (needs 20+ items to be useful) |
| 4 | Related items | `uv run sb related <id>` for 3-4 items | sensible neighbours; if empty or noisy, the threshold needs changing |

**Check 3 matters most.** The link threshold is still a guess (D31). With the stats from your real
library I can propose a value, which is then frozen before you label pairs.

## What I need from you before Phase 6

1. Save and analyze ~30-40 real items.
2. Run check 3 and settle the threshold.
3. `uv run sb eval-templates`, then fill the three files as described in `eval/README.md`.

None of this blocks Phases 4 and 5.

## How it connects to earlier phases

- Uses Phase 1's chunk vectors (`vec_chunks`) for search and item vectors (`vec_items`) for links.
- Uses Phase 2's tags (`item_topics`) for the tag-overlap part of the link score.
- Adds a `link` stage to the ingest timings.
- Selects items by stored data, not by `ingest_status`, so an item whose analysis failed is still
  searchable and linkable.
- Differences from DESIGN.md: tag overlap uses topics + skills and the column is `tag_jaccard`
  (D32); the weight and threshold were not tuned on seed data (D31).

What later phases will use: `links.related()` and `links.graph()` (Phase 5 UI), `search()`
(Phase 5 UI, Phase 6 retrieval eval), `links.pair_score()` and `eval/*.yaml` (Phase 6).

## Known limitations

- **Link weight and threshold are unvalidated** (D31).
- Search quality is unmeasured until Phase 6.
- Embedding-only search: exact names and acronyms may rank poorly (D30).
- Each `sb search` call loads the embedding model first, which adds start-up time to every CLI
  call; the UI will keep it loaded.
- Link updates and `--stats` load all item vectors into memory; fine for about a thousand items,
  not designed for far more.
- Still no command to delete an item; vectors and links of a deleted item would need cleaning.
- `sb eval-templates` picks summary items only from analyzed items; if fewer than 15 are
  analyzed, the file is shorter.
