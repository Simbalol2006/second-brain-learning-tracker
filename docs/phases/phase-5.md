# Phase 5 — Streamlit UI

Date: 2026-11-16 · Built and tested on CPython 3.12.3 (uv-managed), Linux, **no browser**.

## What was built

A web UI over everything from Phases 1–4. Seven pages, chosen in the sidebar:

| Page | What it does |
|------|--------------|
| **Add** | Paste one or more URLs; saves each (fetch, chunk, embed) and optionally analyzes it, with a progress box per URL and the error text if something fails |
| **Library** | Table of all items; filter by progress, topic or text; open an item |
| **Item** | Title, source link, metadata, progress switch, summary, key points, tags, difficulty, "related to this" with scores; buttons to analyze / re-analyze, transcribe with Whisper, retry a failed save |
| **Search** | Query box; results with the matching passage, a timestamp link and an "Open" button |
| **Graph** | Item graph, coloured by progress or difficulty, with a links-per-item slider and a "focus on one item" option |
| **Recommend** | "What should I learn next?" with an LLM on/off switch; ordered suggestions with reasons; warning when it falls back to rules |
| **Profile** | Progress counts, goals (add/remove), skills table, set or remove a hand-set level |

## Files

| File | Change |
|------|--------|
| `app/streamlit_app.py` | new — the whole UI |
| `src/secondbrain/ui.py` | new — Streamlit-free helpers (table rows, filters, time format, graph DOT text) |
| `.streamlit/config.toml` | new — usage statistics off |
| `pyproject.toml`, `uv.lock` | added `streamlit`, `graphviz` |
| `tests/test_app.py` | new — headless app tests |
| `tests/test_ui.py` | new — helper tests |
| `README.md` | "Run the app" section |

No core module changed in this phase.

## How to run

```bash
uv sync
uv run pytest
uv run streamlit run app/streamlit_app.py      # http://localhost:8501
```

## Test result

`uv run pytest` → **305 passed, 3 deselected** (the 3 deselected are the live tests).
52 of these are new: 22 helper tests and 30 headless app tests covering every page with an empty
and a seeded database plus the main actions. The real server was also started once here and
answered its health check.

## Not verified in the build environment — please run on the laptop

The tests run the app script without a browser. They show that every page executes without
errors and puts the right text on the page. **Nobody has looked at this UI yet.** Layout, the
graph drawing, and anything visual are unverified.

| # | Check | What to look for |
|---|-------|------------------|
| 1 | `uv run streamlit run app/streamlit_app.py` | the page opens; sidebar shows the 7 pages and your item count |
| 2 | **Add** a real URL with "Summarize and tag" ticked | progress box ends green with the title; the item appears in Library |
| 3 | **Graph** with 5+ items | a readable network: boxes with titles, lines between related items, colours changing with the "Colour by" switch. **Most likely page to need a fix** — tell me what you see, or send a screenshot |
| 4 | **Item** → change progress to done → **Profile** | the item's tags appear under Skills |
| 5 | **Search**, then "Open" on a result | lands on that item; the title link opens YouTube at the right time |
| 6 | **Recommend** with and without the LLM switch | suggestions appear; no warning when the LLM is configured |
| 7 | Dark theme (Settings → Theme) | graph text still readable |

## How it connects to earlier phases

The UI calls the same functions as the CLI, so both always behave the same:
`pipeline.ingest_url` / `process_item` (Phase 1), `analyze.analyze_item` (Phase 2),
`search.search`, `links.related`, `links.graph` (Phase 3), `profile.*`, `recommend.recommend`
(Phase 4). The CLI keeps working and shares the same database.

Differences from DESIGN.md: the graph is drawn with Graphviz through Streamlit instead of pyvis
(D48); navigation is a sidebar radio instead of separate page files (D46).

## Known limitations

- **Visual result unverified** — see the table above.
- Saving, analyzing and Whisper block the page until they finish; there is no background queue.
  The Add page handles several URLs one after another, which can take minutes.
- No per-page URLs: reloading the browser returns to the Add page.
- Graph nodes cannot be dragged; clicking one opens the source, not the item page. Above roughly
  50–100 items the whole-library graph will get crowded; use "Focus on".
- Recommendations stay on screen until you press the button again, even after marking items done.
- A change to `.env` needs an app restart (settings and the LLM provider are cached).
- No delete button for items (there is still no delete in the core either).
- Single user, local only: do not expose the port to a network; there is no login.
