# Decision log

One entry per meaningful decision. Newest phase at the bottom.
Format: **What / Why / Alternatives / Trade-off.**

---

## Phase 1 — Ingest + storage

### D1. uv with Python pinned to 3.12
- **What:** `.python-version` = 3.12; `uv sync` installs that interpreter and a locked dependency set (`uv.lock`).
- **Why:** Kubuntu 26.04's system Python is newer than most native wheels are tested on. Built and tested here on CPython 3.12.3.
- **Alternatives:** system Python + `venv` + `requirements.txt`; conda.
- **Trade-off:** one extra tool to install (uv), in exchange for identical versions on every machine.

### D2. Embeddings with fastembed, not sentence-transformers
- **What:** `BAAI/bge-small-en-v1.5` (384-dim) through fastembed's ONNX runtime on CPU.
- **Why:** no PyTorch install, small download, fast start. GPU stays free for Whisper / Ollama.
- **Alternatives:** sentence-transformers (more widely known, pulls in torch); Ollama embeddings.
- **Trade-off:** fastembed ships a *quantized* ONNX build of the model (`Qdrant/bge-small-en-v1.5-onnx-Q`), so vectors can differ slightly from the original weights. The Phase 6 retrieval eval measures the result.

### D3. sqlite-vec with exact cosine search
- **What:** two `vec0` tables (`vec_chunks`, `vec_items`) with `distance_metric=cosine`, in the same SQLite file as the rest. No ANN index.
- **Why:** one file to back up, no server, and exact search is fast at the expected size (tens of thousands of vectors).
- **Alternatives:** Chroma, LanceDB, FAISS, plain NumPy over BLOBs.
- **Trade-off:** sqlite-vec is pre-1.0 (pinned at 0.1.9 in `uv.lock`). All its SQL is in `vectors.py` so a swap touches one file.

### D4. Vector tables are created on first embed and tied to one model
- **What:** `ensure_vector_tables()` creates the tables with the embedder's dimension and stores model name + dimension in `meta`. A different model later is refused with an explanation.
- **Why:** vectors from two models are not comparable; mixing them silently would corrupt search.
- **Alternatives:** fixed 384 in the schema; a model column per vector.
- **Trade-off:** changing the embedding model means a fresh `DATA_DIR` and re-ingest (no re-embed command yet).
- **Note:** table creation commits immediately. SQLite DDL is not undone by a later rollback, and a test caught the tables and the `meta` rows getting out of step.

### D5. Schema version in `PRAGMA user_version` + a list of migrations
- **What:** `db.MIGRATIONS` is a list of SQL scripts; `connect()` applies the ones newer than `user_version`.
- **Why:** later phases add tables; this is the smallest mechanism that upgrades an existing database safely.
- **Alternatives:** `CREATE TABLE IF NOT EXISTS` everywhere (cannot change existing tables); Alembic.
- **Trade-off:** no downgrade path. (DESIGN.md said the version would live in `meta`; `user_version` is built in and simpler.)

### D6. Stages resume by "is the output missing?", not by status
- **What:** `process_item()` runs fetch if no transcript is stored, chunk if no chunks, embed if no item vector. `ingest_status` is for display.
- **Why:** a failed item can be retried with the same command and finished stages are never repeated. Status alone cannot say where a `failed` item stopped.
- **Alternatives:** a state machine driven by status; a `failed_stage` column.
- **Trade-off:** three small existence queries per ingest call.

### D7. Chunking by word count on segment boundaries (220 words, 40 overlap)
- **What:** caption snippets are packed into ~220-word chunks; the last ≤40 words' worth of snippets repeat in the next chunk. Chunks keep start/end times.
- **Why:** auto-captions have no punctuation, so sentence splitting is unreliable. ~220 words stays inside the model's 512-token limit. Cutting between snippets keeps timestamps exact.
- **Alternatives:** token-based splitting with the model tokenizer; sentence splitting; fixed time windows.
- **Trade-off:** word count only approximates tokens. A chunk can reach 260 words; unusually token-dense text could be truncated by the model. Sizes are settings, and the eval will show if they need tuning.

### D8. Item vector = normalized mean of its chunk vectors
- **What:** one vector per item in `vec_items`, computed at embed time.
- **Why:** available in Phase 1 without an LLM call; used for linking in Phase 3.
- **Alternatives:** embed the LLM summary (needs Phase 2); embed title + description.
- **Trade-off:** a mean blurs videos that cover several unrelated topics. The linking eval will show whether that matters.

### D9. Metadata from yt-dlp with an oEmbed fallback
- **What:** yt-dlp gives title, channel, duration, date. If it fails, YouTube's oEmbed endpoint gives title + channel, and duration is taken from the last caption timestamp.
- **Why:** yt-dlp breaks when YouTube changes or when the JavaScript runtime is missing; captions often still work, so ingest should not die on metadata alone.
- **Alternatives:** fail hard; YouTube Data API (needs a key and quota).
- **Trade-off:** fallback items have no publish date and an approximate duration.

### D10. "No transcript" is a status, not a failure; Whisper is opt-in and defaults to CPU
- **What:** videos without captions are saved as `no_transcript` with the reason. Whisper runs only on `--transcribe` / `sb transcribe`. Defaults: `small`, `int8`, `WHISPER_DEVICE=cpu`.
- **Why:** transcription takes minutes, so it must be a choice. CPU works on a fresh machine with no GPU libraries.
- **Alternatives:** always transcribe; default device `auto` (DESIGN.md's plan).
- **Trade-off:** slower by default. `auto` was dropped because, with the NVIDIA driver present but cuBLAS/cuDNN missing, it would pick the GPU and fail. GPU use is a documented opt-in.

### D11. Stage timings stored in a table
- **What:** `ingest_timings(item_id, stage, seconds)` gets a row for each successful stage.
- **Why:** Phase 6 must report ingest time per video from real runs; a table is easier to aggregate than log files.
- **Alternatives:** parse log output; time only inside the eval script.
- **Trade-off:** one small extra table that was not in DESIGN.md's data model.

### D12. One exception family; the pipeline records the failure, then raises
- **What:** all expected errors derive from `SecondBrainError`. `process_item()` catches any exception, writes `failed` + reason on the item, and raises `IngestError`. The CLI prints it and exits with code 1.
- **Why:** failures are visible in three places (terminal, log, database) and never swallowed.
- **Alternatives:** return error objects; let raw library exceptions reach the user.
- **Trade-off:** unexpected bugs are also wrapped in `IngestError`; the full traceback goes to the log.

### D13. Offline tests with fakes; live checks are separate
- **What:** the default test run uses a fake embedder, fake YouTube responses and a local HTML fixture. Tests marked `live` need internet and are skipped unless asked for.
- **Why:** the build environment cannot reach YouTube or Hugging Face, and tests should be free and repeatable.
- **Alternatives:** recorded HTTP cassettes (could not be recorded from the build environment).
- **Trade-off:** fakes prove our logic, not that YouTube's real responses still match. The live checks in `docs/phases/phase-1.md` cover that and must be run on the laptop.

### D14. Dataclasses for internal data; Pydantic only at the edges
- **What:** `Segment`, `Chunk`, `Metadata`, `Transcript` are frozen dataclasses. Pydantic is used for settings now and for LLM output from Phase 2.
- **Why:** internal data is created by our own code and does not need validation.
- **Alternatives:** Pydantic models everywhere.
- **Trade-off:** two styles of data class in the codebase.

### D15. Audio is deleted after a successful transcription
- **What:** the downloaded audio file is removed once Whisper succeeds, and kept if it fails.
- **Why:** the transcript is what the app needs; audio files are large. Keeping it on failure lets a retry skip the download.
- **Alternatives:** keep everything; a `KEEP_AUDIO` setting.
- **Trade-off:** re-transcribing with a different Whisper model later needs a new download.

---

## Phase 2 — Understand (summary, tags, difficulty)

### D16. One provider interface with one method
- **What:** `LLMProvider.generate_json(system, user, schema)` returns the raw answer text plus token counts. Two implementations: Anthropic and Ollama. `build_provider(settings)` is the only place that picks one.
- **Why:** every LLM call in the app (now analysis, later recommendations and the eval judge) goes through the same door, so switching provider is one line in `.env`.
- **Alternatives:** LiteLLM or LangChain (large dependencies for two backends); provider-specific calls spread through the code.
- **Trade-off:** provider-specific features (streaming, tool use, prompt caching) are not reachable through the interface.

### D17. Providers return text; the caller validates
- **What:** providers ask the backend to enforce the schema, but `analyze.py` always validates the returned text with Pydantic.
- **Why:** enforcement differs by backend and model. One validation path means the same guarantees, and the same measured JSON-validity rate, for both.
- **Alternatives:** trust the backend's structured-output mode; let each provider return parsed objects.
- **Trade-off:** with a strict backend the validation is mostly redundant work (microseconds).

### D18. Anthropic: structured outputs via `output_config.format`, default `claude-haiku-5-5`
- **What:** the schema is sent as `output_config={"format": {"type": "json_schema", ...}}`. No `temperature`/`top_p`. `max_tokens` defaults to 4096. Refusals and `max_tokens` stops raise `LLMError`.
- **Why:** this is the documented way to get schema-shaped JSON; the docs for Haiku 5.5 say to omit sampling parameters and that thinking counts toward `max_tokens`, hence the generous limit.
- **Alternatives:** a forced tool call (also accepted, skips thinking); plain "answer in JSON" prompting.
- **Trade-off:** built from the documentation read on 2026-11-16 and **not yet run against the real API** (no key in the build environment). See the live checks in `docs/phases/phase-2.md`.

### D19. Ollama: plain HTTP to `/api/chat` with explicit `num_ctx`
- **What:** `httpx` POST with the schema in `format`, `stream: false`, `temperature: 0`, `num_predict` = output limit and `num_ctx` = `LLM_MAX_INPUT_TOKENS + LLM_MAX_OUTPUT_TOKENS`. The schema is also pasted into the system prompt.
- **Why:** Ollama's default context window is small and silently drops what does not fit, which would produce summaries of half a transcript with no error. Setting it ties the window to the same budget the chunking uses.
- **Alternatives:** the `ollama` Python package (thin wrapper, one more dependency); Ollama's OpenAI-compatible endpoint (context size is awkward to set there).
- **Trade-off:** a bigger `num_ctx` needs more VRAM, so local use needs smaller budgets (README suggests 4000 / 1024). Not run against a real Ollama server yet.

### D20. Size limits live in validators, not in the JSON Schema
- **What:** the schema sent to the model has only types, enums, required fields and descriptions. "2–10 key points", "at most 8 tags", "summary ≤ 2000 chars" are checked in Pydantic validators.
- **Why:** structured-output modes support only a subset of JSON Schema; keywords like `minItems`/`maxLength` can be rejected or ignored. The descriptions tell the model the intended sizes.
- **Alternatives:** full JSON Schema constraints; no limits at all.
- **Trade-off:** the backend cannot enforce the limits, so a model can break them; that costs a retry.

### D21. One retry that shows the model its mistakes, then fail
- **What:** if validation fails, the same prompt is sent again with the previous answer and the validation errors appended. A second failure raises `AnalysisError` and marks the item `failed`.
- **Why:** small models often fix a schema slip when shown it. More retries rarely help and cost time and money.
- **Alternatives:** no retry; many retries; "repair" the JSON in code.
- **Trade-off:** up to 2× the calls for a bad answer. Silent repair was rejected because it would hide how reliable a model really is.

### D22. Long content: map → collapse → final, with a character budget
- **What:** content is packed into windows of `LLM_MAX_INPUT_TOKENS × 4 chars × 0.9` minus the fixed prompt text. One window → one call. More → notes per window, notes merged in rounds if they still do not fit, then the final call on the notes.
- **Why:** local models have small context windows. Windows are cut between transcript segments from the raw transcript (not the overlapping search chunks), so nothing is counted twice.
- **Alternatives:** truncate long transcripts; a real tokenizer per provider; "refine" chains.
- **Trade-off:** 4 chars/token is an estimate (hence the 10% margin and the Ollama cut-off warning). The final answer is a summary of summaries, so detail is lost on very long content.

### D23. Topics and skills share one table and are normalized
- **What:** `topics(name)` plus `item_topics(item_id, topic_id, kind)` with kind `topic` or `skill`. Names are lower-cased, whitespace-collapsed and stripped of punctuation. The 40 most-used names of each kind are put in the prompt with an instruction to reuse them.
- **Why:** linking (Phase 3) and the profile (Phase 4) only work if the same idea gets the same name.
- **Alternatives:** a fixed taxonomy; merging near-duplicate tags by embedding similarity.
- **Trade-off:** still relies on the model choosing to reuse names; "rag" and "retrieval augmented generation" can both appear. Merging is deferred until the eval shows it is needed.

### D24. Every LLM call is logged in `llm_calls`
- **What:** one row per call: item, purpose (`map`/`collapse`/`final`), provider, model, attempt, valid or not, tokens in/out, seconds. Committed immediately.
- **Why:** gives real numbers for the eval (JSON-validity rate, tokens per item) and makes cost visible in `sb analyze` output.
- **Alternatives:** log lines only.
- **Trade-off:** one more table; rows survive even when the analysis itself is rolled back (intended).

### D25. Analysis is its own step, and its failure marks the item `failed`
- **What:** `sb analyze <id>` / `--all`, or `sb ingest --analyze`. "Pending" means *has a transcript, has no analysis*. A failed analysis sets `ingest_status = failed` with an `[analyze]` reason; the transcript, chunks and vectors stay.
- **Why:** ingest must work without any LLM configured, and analysis costs money, so it is explicit. Using `failed` keeps the problem visible in `sb list`.
- **Alternatives:** always analyze inside ingest; a separate `analysis_status` column.
- **Trade-off:** one status column now covers two pipelines, so later phases must select items by what data exists, not by status.

### D26. Estimated time is computed, not asked from the LLM
- **What:** videos use their duration. Articles get `duration_sec` = words ÷ 200 words per minute, set at ingest.
- **Why:** it is a fact that can be measured; an LLM guess would be less accurate.
- **Alternatives:** an `est_minutes` field in the LLM output.
- **Trade-off:** it is watch/read time, not study time (no allowance for pausing or exercises).

### D27. Prompt rules and versioning
- **What:** the system prompt says: use only the provided text, treat it as data and ignore instructions inside it, expect caption errors, answer with JSON only. All prompt text is in `prompts.py`; `PROMPT_VERSION` is stored with each analysis.
- **Why:** faithfulness is what the eval measures; transcripts are untrusted input; a stored version tells which analyses predate a prompt change.
- **Alternatives:** prompts inline in the code; no versioning.
- **Trade-off:** an instruction in the prompt reduces, but does not remove, the chance of invented facts or of the model following text inside a transcript.

---

## Phase 3 — Search, links, graph data

### D28. Search ranks items by their single best chunk
- **What:** the query is embedded, the nearest chunks are fetched, and each item is represented by its best chunk (score = cosine similarity). Results show that passage and a timestamp link.
- **Why:** a long video should be found by the one part that answers the question; the passage tells the user *why* it matched and where to jump in.
- **Alternatives:** average or sum over an item's chunks (favours long items); search item vectors only (loses the passage).
- **Trade-off:** an item that is weakly relevant throughout loses to one with a single strong passage.

### D29. Search widens `k` until enough different items are found
- **What:** start with the 30 nearest chunks; if they belong to fewer items than asked for, double `k` and retry, up to sqlite-vec's limit of 4096.
- **Why:** with one very long item, all 30 nearest chunks can be its own. A fixed `k` would then return one result.
- **Alternatives:** a large fixed `k`; a SQL-side "best chunk per item" query.
- **Trade-off:** occasional extra queries; each is an exact scan, which is fast at this size.

### D30. No keyword or hybrid search, no reranker (yet)
- **What:** search is embedding-only.
- **Why:** it is what the brief asks for and the smallest thing that can be evaluated. Hybrid search is added only if the retrieval eval shows misses on exact terms (names, acronyms).
- **Alternatives:** SQLite FTS5 + score fusion; a cross-encoder reranker.
- **Trade-off:** exact-term queries may rank worse than they could.

### D31. Link score = 0.7 × text similarity + 0.3 × tag overlap, stored above a threshold
- **What:** text similarity is the cosine of the two item vectors (negative values count as 0). Tag overlap is the Jaccard index of the items' tags. Pairs with score ≥ `LINK_THRESHOLD` are stored in `links`.
- **Why:** embeddings find related items even before analysis; shared tags add an explainable signal. Both parts are stored per link so the UI can say why two items are linked.
- **Alternatives:** embeddings only; tags only; k-nearest-neighbour links with no threshold.
- **Trade-off:** **the weight (0.7) and threshold (0.5) are unvalidated starting values.** They could not be set from a score distribution as DESIGN.md planned, because the build environment has neither the real embedding model nor real data. `sb links --stats` exists so they can be set on the real library before labelling.

### D32. Tag overlap uses topics and skills together
- **What:** one tag set per item (topic names ∪ skill names). The column is `tag_jaccard` (DESIGN.md called it `topic_jaccard`).
- **Why:** a shared skill is as much a sign of relatedness as a shared topic, and items often have few tags, so using both gives a less jumpy number.
- **Alternatives:** topics only; separate weights for topics and skills.
- **Trade-off:** depends on the tag consistency discussed in D23; un-analyzed items have no tags and can only link through text similarity (maximum score 0.7).

### D33. Links are updated when an item is embedded or analyzed
- **What:** `links.update_item()` runs at the end of ingest and again after analysis (tags changed). It recomputes that item against all others. `sb links --rebuild` recomputes everything.
- **Why:** "related to this" is correct right after saving, with no separate command to remember.
- **Alternatives:** compute links only on demand; a nightly rebuild.
- **Trade-off:** each update loads all item vectors (fine for ~1,000 items). Changing the weight or threshold needs a manual rebuild.

### D34. Only links above the threshold are stored; any pair can be scored on demand
- **What:** `links` holds linked pairs only. `pair_score()` computes one pair, and `all_pair_scores()` every pair, from vectors and tags.
- **Why:** storing all pairs grows quadratically and most would never be read. The eval needs scores for unlinked pairs too, so they must be computable.
- **Alternatives:** store every pair; store top-k per item.
- **Trade-off:** scoring all pairs is recomputed each time `--stats` or the template sampler runs.

### D35. Graph export is plain JSON, limited to each item's strongest links
- **What:** `links.graph()` returns `{nodes, edges}`; an edge is kept if it is among the top 5 links of either endpoint. Phase 5 draws it.
- **Why:** a threshold alone can leave popular topics as an unreadable hairball; capping per item keeps the picture usable. JSON keeps the drawing library replaceable.
- **Alternatives:** export every link; networkx objects.
- **Trade-off:** links weaker than both endpoints' top 5 are stored but not drawn.

### D36. Eval templates refer to items by URL, hide scores, and are never overwritten silently
- **What:** `sb eval-templates` writes `eval/retrieval.yaml`, `links.yaml`, `summaries.yaml`. Pairs are sampled from the top, middle and rest of the score range with a fixed seed and written in shuffled order without scores. Existing files need `--force`.
- **Why:** URLs survive a rebuilt database. Hiding scores keeps labels independent of the system. Refusing to overwrite protects an hour of labelling.
- **Alternatives:** label by database id; purely random pairs (30 random pairs would contain almost no related ones, so recall could not be measured).
- **Trade-off:** the stratified sample is not representative of all pairs; `docs/EVAL.md` must state that precision and recall describe the sample.

---

## Phase 4 — Profile, progress, recommendations

### D37. Derived skill levels are recomputed from scratch on every progress change
- **What:** `refresh_derived()` deletes all derived rows and rebuilds them from the items currently marked done. Rule: 1 finished item → beginner; 2 with one intermediate-or-harder → intermediate; 3 with one advanced → advanced.
- **Why:** undoing a "done", re-analyzing an item, or changing tags all stay correct without tracking increments. The rule is simple enough to explain in one line.
- **Alternatives:** increment counters on each "done"; ask the LLM to assess skill.
- **Trade-off:** finishing videos is weak evidence of skill; the levels describe exposure, not tested ability. Levels depend on the tag consistency discussed in D23.

### D38. Manual skill levels always win
- **What:** `sb profile set-skill` stores a `manual` row. Recomputation only updates its evidence count, never its level. Removing it falls back to the derived level.
- **Why:** the user knows things the app never saw (courses, work). Automatic changes to something set by hand would be surprising.
- **Alternatives:** take the maximum of manual and derived; ask before overwriting.
- **Trade-off:** a manual level can go stale as the user learns more; they must update it themselves.

### D39. The profile uses the same tag names as items
- **What:** `profile_skills` is keyed by `topics.id`. Topic tags and skill tags of finished items both count as skills. Manual skill names go through the same normalization.
- **Why:** recommendations compare "what the item is tagged with" to "what the user has a level in"; one shared vocabulary makes that a lookup.
- **Alternatives:** a separate skill taxonomy with a mapping to topics.
- **Trade-off:** a manual skill named differently from the LLM's tag ("ml" vs "machine learning") will not match.

### D40. Recommendations: rule-based scoring first, LLM for order and reasons second
- **What:** every unfinished, analyzed item gets a score from three parts. The top ten go to the LLM with the profile; it returns an ordered list of item ids with one-line reasons.
- **Why:** the scoring works with no LLM (and keeps the prompt small enough for a local model); the LLM adds what rules do badly — sequencing and a readable reason.
- **Alternatives:** LLM only over the whole library (cost grows with library size, no fallback); rules only (reasons read like templates).
- **Trade-off:** the LLM only sees the shortlist, so a good item ranked 11th by the rules cannot be recommended.

### D41. Score = 0.5 × goal fit + 0.2 × novelty + 0.3 × readiness
- **What:** *goal fit* is the cosine between the item vector and the closest goal, min-max rescaled across the candidates. *Novelty* is the share of the item's tags the user has no level in. *Readiness* is `1 − |difficulty − (level + 1)| / 3`, highest one step above the user's level. With no goals, the last two are rescaled to sum to 1.
- **Why:** raw embedding similarities sit in a narrow band, so without rescaling goals barely changed the order (a test showed an unrelated easy item beating the goal's own topic).
- **Alternatives:** raw similarity; learned weights; LLM-judged relevance per item.
- **Trade-off:** **the weights are unvalidated starting values**, and there is no recommendation metric in the eval plan. Rescaling also means that when nothing matches a goal, the least-bad item still gets goal fit 1.

### D42. LLM answers are checked against the shortlist; failure falls back visibly
- **What:** validation gets the allowed ids and the limit as context: unknown ids, duplicates, too many items or an empty list are rejected and retried once. If the LLM is unreachable, unconfigured or wrong twice, the rule-based order and reasons are returned with a warning that the CLI prints.
- **Why:** a hallucinated item id must never reach the user; a recommendation feature that dies with the LLM would break the "must not depend on a frontier model" rule.
- **Alternatives:** drop invalid entries silently and keep the rest; fail the command.
- **Trade-off:** rule-based reasons are formulaic.

### D43. One shared "validated LLM call" helper
- **What:** the retry-and-log logic from Phase 2 moved to `llm/validated.py` (`call_validated`) and is used by both analysis and recommendations. Recommendation calls are logged in `llm_calls` with no item id.
- **Why:** the second user of the logic arrived; copying it would let the two drift apart.
- **Alternatives:** keep it private to `analyze.py` and duplicate.
- **Trade-off:** none significant; `prompts.py` lost the retry text, which now lives next to the code that uses it.

### D44. Goals are plain rows; progress stays on the item
- **What:** `profile_goals(id, text)`; removing a goal deletes it (DESIGN.md had an `active` flag). Progress is `items.progress_status` + `done_at`, as created in Phase 1.
- **Why:** nothing uses an inactive goal, and one status column per item is all tracking needs.
- **Alternatives:** soft-delete goals; a progress history table.
- **Trade-off:** no history of when an item moved between states, only when it was finished.

---

## Phase 5 — Streamlit UI

### D45. One Streamlit script; the UI only calls the core package
- **What:** `app/streamlit_app.py` has one function per page. Each draws widgets and calls the same functions the CLI uses (`pipeline`, `analyze`, `search`, `links`, `profile`, `recommend`). Streamlit-free helpers (table rows, filters, graph text) are in `secondbrain/ui.py`.
- **Why:** there is one implementation of every feature, already tested in Phases 1–4, and the UI file stays readable for someone new to frontend work.
- **Alternatives:** a FastAPI backend plus a JavaScript frontend; logic inside the page functions.
- **Trade-off:** the UI and the Python process are one thing: a long operation blocks the page.

### D46. Navigation by a sidebar radio button and `st.session_state`
- **What:** the current page is the radio's value (`key="page"`); buttons switch page through a small `go_to()` callback. The open item is remembered in `current_item`, separate from the drop-down's own state, because Streamlit forgets a widget's state when the widget is not drawn.
- **Why:** it is the simplest pattern, works in one file, and can be driven by Streamlit's headless test tool.
- **Alternatives:** `st.navigation` / a `pages/` folder (gives each page a URL).
- **Trade-off:** no per-page URLs, so no bookmarking a page and no browser back button between pages.

### D47. Heavy objects are cached once; the database connection is per redraw
- **What:** settings, the embedding model and the LLM provider are created once with `st.cache_resource`. A SQLite connection is opened at the start of every script run and closed at the end.
- **Why:** loading the embedding model takes seconds, so it must not happen on every click. SQLite connections are cheap and are tied to the thread that created them, while Streamlit runs each redraw on its own thread.
- **Alternatives:** one shared connection with `check_same_thread=False`.
- **Trade-off:** a change to `.env` needs an app restart.

### D48. Graph view with Graphviz inside Streamlit, not pyvis
- **What:** `ui.graph_dot()` builds Graphviz DOT text; the page shows it with `st.graphviz_chart`, passed as a `graphviz.Source` so the force-directed `neato` engine is used (a plain string always gets the top-down `dot` engine). Nodes are coloured by progress or difficulty, marked ✓/▶, and link to the source; a "focus" option shows one item's neighbourhood.
- **Why:** Streamlit renders Graphviz itself, in the browser, with no internet and no extra JavaScript; the only new dependency is the small `graphviz` Python package. The DOT text comes from a pure function that is unit-tested. DESIGN.md planned pyvis, which adds a JavaScript bundle and several dependencies for the same picture.
- **Alternatives:** pyvis (draggable nodes); `streamlit-agraph`; Plotly.
- **Trade-off:** nodes cannot be dragged and clicking a node opens the source URL rather than the item page. **How the picture actually looks was not checked in the build environment (no browser there).**

### D49. LLM calls happen only on an explicit button
- **What:** analysis runs on "Save" (if the box is ticked) or "Analyze"; recommendations run on "What should I learn next?" and the result is kept in `st.session_state` until asked again.
- **Why:** Streamlit re-runs the whole script on every interaction; without this, every click would cost an LLM call.
- **Alternatives:** time-based caching of recommendations.
- **Trade-off:** recommendations on screen can be stale after you mark something done; press the button again.

### D50. The app is tested headless on every page
- **What:** `tests/test_app.py` uses Streamlit's `AppTest` to run the real script with a temp database, a fake embedder and a fake LLM: every page with an empty and a seeded database, plus the main actions (save, filter, open, progress change, analyze, search, recommend, profile edits, error display).
- **Why:** it catches exceptions and wrong wiring without a browser, which the build environment does not have.
- **Alternatives:** browser automation (Playwright); manual testing only.
- **Trade-off:** it proves the script runs and shows the right text, not that the layout looks good or that the graph draws.

### D51. Streamlit usage statistics are switched off
- **What:** `.streamlit/config.toml` sets `gatherUsageStats = false`.
- **Why:** the brief says everything runs on the laptop except the LLM call.
- **Alternatives:** leave Streamlit's default (on).
- **Trade-off:** none.

---

## Phase 6 — Evaluation and polish

### D52. The report is generated; missing parts say "Not run"
- **What:** `scripts/eval.py` calls `evaluation.run()`, which returns one `Section` per area, and writes `docs/EVAL.md` from them. A section that cannot be measured (no labels, unknown URL, no LLM, no data) carries a reason instead of numbers. The committed `docs/EVAL.md` is a placeholder with no metrics, and a test checks that.
- **Why:** the rule for this project is "only real measured numbers". Making the script the only writer, and giving it no way to fill gaps, enforces that mechanically.
- **Alternatives:** a hand-written results page; default values for missing parts.
- **Trade-off:** until the script is run on a real library there are no results to show.

### D53. Every count is shown next to every rate
- **What:** rates are printed as `0.750 (3/4)`; a rate with nothing to divide by is `n/a (0/0)`. The report states that the label sets are small and, for linking, that the pairs are a stratified sample.
- **Why:** with 25 queries or 30 pairs a single label moves a metric by several points; the reader needs the denominators to judge that.
- **Alternatives:** confidence intervals (hard to read, and shaky at these sizes).
- **Trade-off:** tables are a little wider.

### D54. Retrieval: recall@5 and MRR over items, latency measured warm
- **What:** results are item-level. recall@5 = relevant items in the top 5 ÷ all relevant items; MRR uses the first relevant item within the top 20, else 0; both are means over queries. Latency times `search()` with the model already loaded, a set number of times per query, and reports p50 and p95.
- **Why:** these are the metrics the brief asks for, defined so they can be recomputed by hand from the per-query table in the report.
- **Alternatives:** chunk-level relevance labels; nDCG.
- **Trade-off:** latency excludes model loading (a one-time cost that the CLI pays on every call) and UI rendering.

### D55. Linking is scored from the formula and the configured threshold
- **What:** for each labelled pair the score is recomputed with `pair_score()` and compared with `LINK_THRESHOLD`; the stored `links` table is not consulted. A second table shows precision and recall at other thresholds, marked "for information only".
- **Why:** the result then depends only on the configuration printed in the report, not on whether `sb links --rebuild` was run. The sweep helps explain a result without pretending it is an unbiased estimate.
- **Alternatives:** read stored links; omit the sweep to remove the temptation to tune.
- **Trade-off:** the sweep is tuning information; the report says that choosing from it is tuning on the eval set.

### D56. Faithfulness is judged per claim, by an LLM, against a written rubric
- **What:** each summary is split into sentences and key points ("claims"). The judge gets `eval/RUBRIC.md` as its instructions, the source text and the numbered claims, and must return exactly one verdict per claim (`supported` / `not_supported` / `contradicted`). Faithfulness = supported ÷ all claims. The judge goes through the same provider interface; `--judge-model` picks a different model than the app's.
- **Why:** claim-level verdicts are checkable by a person, unlike a 1–5 score. Requiring one verdict per claim number (validated, with a retry) stops the judge from skipping claims.
- **Alternatives:** a single score per summary; NLI models; embedding similarity.
- **Trade-off:** an LLM judge can be wrong, and shares blind spots with the summarizer when both use the same model (the default). The human spot check exists to measure that.

### D57. Long sources are judged part by part
- **What:** if the source does not fit the judge's input budget, each part is judged separately: a claim is supported if any part supports it, else contradicted if any part contradicts it, else not supported.
- **Why:** it checks claims against the whole source and still works with a small local model.
- **Alternatives:** judge against the top retrieved chunks per claim (cheaper, but can miss the supporting passage).
- **Trade-off:** more judge calls for long items; a claim that combines facts from two parts can be marked not supported.

### D58. The human spot check is blind, and judge verdicts are cached
- **What:** the first run writes `eval/spot_check.yaml` with the claims of the items marked `spot_check: true` and an empty `human:` field, without the judge's verdicts. Later runs report exact agreement and supported/not-supported agreement. Verdicts are cached in `data/eval_judge_cache.json`, keyed by judge model, rubric text, URL and claims.
- **Why:** seeing the judge's answer first would bias the human label. The cache keeps the second run (after labelling) free and makes sure the agreement is computed against the same verdicts that produced the faithfulness number.
- **Alternatives:** show verdicts for review (faster, but anchored); no cache.
- **Trade-off:** changing the rubric, the judge model or a summary invalidates the cached verdicts for it, as it should.

### D59. Speed comes from timings recorded at ingest
- **What:** the speed section reads `ingest_timings` (the latest run of each stage per item) and `llm_calls`; the eval does not re-ingest anything.
- **Why:** these are real measurements of real use and cost nothing extra; re-downloading 30 videos during an eval would be slow and would hit YouTube for no reason.
- **Alternatives:** time a fresh ingest of a fixed list inside the eval.
- **Trade-off:** the items may have been saved on different days, networks or settings; the report says so.

### D60. Items can be deleted
- **What:** `library.delete_item()` removes the item row (transcript, chunks, analysis, tags, links and timings go with it through foreign keys), its vectors, and recomputes derived skills. Available as `sb delete` and on the Item page behind a confirmation tick.
- **Why:** a wrong or failed save could not be removed, which also made it awkward to keep the eval library clean.
- **Alternatives:** a hidden/archived flag.
- **Trade-off:** deletion is permanent; unused tag names stay in the `topics` table.

---

## Phase 7 — Hybrid search (BM25 + dense, fused with RRF)

### D61. Keyword search uses SQLite FTS5 with BM25
- **What:** a migration adds `chunks_fts`, an FTS5 index over `chunks.text`, and `search.bm25_items()` ranks chunks by FTS5's built-in `bm25()`, keeping the best chunk per item (as dense search already did).
- **Why:** dense embeddings match meaning but can miss exact terms: names, acronyms, error codes, library names ("tsvector", "okapi"). BM25 is strong exactly there. FTS5 ships with SQLite, so this adds no dependency, no server and no second store to keep in step with the database.
- **Alternatives:** `rank_bm25` in Python (the whole corpus in memory, rebuilt on every start); a search server such as Elasticsearch or Tantivy (far too heavy for a single-user laptop app).
- **Trade-off:** FTS5's BM25 parameters are fixed (k1 = 1.2, b = 0.75); per-column weights are the only tuning it offers.

### D62. The index is an external-content table kept in step by triggers
- **What:** `chunks_fts` stores only the index (`content='chunks'`) and reads text from `chunks`. Triggers on insert, update and delete keep it current; the migration runs FTS5's `rebuild` so a database from Phases 1–6 is indexed on upgrade without re-ingesting.
- **Why:** no text is stored twice, and no code path (re-chunking, deleting an item, cascaded deletes) can forget to update the index, because the database does it. Tests run FTS5's `integrity-check` after each kind of change.
- **Alternatives:** a contentful FTS table filled from Python in `items.replace_chunks()`.
- **Trade-off:** writing to `chunks` directly in SQL also writes the index; that is the point, but it makes the triggers part of the schema to remember.

### D63. Words keep their combining marks
- **What:** the tokenizer is `porter unicode61 categories 'L* N* M* Co'`, and the query side splits words by the same character classes (`search._words`).
- **Why:** FTS5's default and Python's `\w` both treat combining marks as separators, which cuts Telugu or Hindi words into single consonants ("తెలుగు" becomes త, ల, గ), so a one-letter query matched unrelated text. Found by a test while building this phase; the same class of bug as the Unicode word-boundary fix in the bilingual IR project.
- **Alternatives:** the default tokenizer (fine for English only).
- **Trade-off:** the Porter stemmer still only stems English; other scripts are matched as whole words.

### D64. User text never reaches FTS5 as query syntax
- **What:** `fts_query()` lower-cases the query, splits it into words, quotes each word and joins them with OR.
- **Why:** FTS5 has its own query language: a raw query containing `-`, `"`, `*`, `:`, `AND` or `NEAR(` either fails with a syntax error or means something the user did not type. OR (rather than FTS5's default AND) lets an item that matches only some words still be found; BM25 already ranks items matching more and rarer words higher.
- **Alternatives:** pass the query through and catch errors; AND semantics.
- **Trade-off:** no phrase or prefix search for power users.

### D65. Rankings are merged with reciprocal rank fusion, k = 60
- **What:** hybrid mode takes the top `SEARCH_DEPTH` items (default 50) from dense and from BM25 and scores each item `Σ 1 / (RRF_K + rank)` over the lists it appears in. The passage shown is the one from the list that ranked the item higher. Each hit records its dense and BM25 rank.
- **Why:** cosine similarity and BM25 scores live on different scales, and BM25's scale changes with the query and the corpus, so a weighted sum of raw scores needs a normalisation that is itself a tuning problem. RRF uses only ranks and has one constant; k = 60 is the value from the original paper (Cormack, Clarke and Büttcher, SIGIR 2009) and is the common default.
- **Alternatives:** weighted sum of min-max-normalised scores; a cross-encoder reranker (better, but a second model to run on a CPU).
- **Trade-off:** RRF ignores how far ahead a top result is. k and depth are defaults, **not tuned on this project's data**.

### D66. Hybrid is the default; the eval compares all three modes
- **What:** `SEARCH_MODE` (dense / bm25 / hybrid) defaults to hybrid; `sb search --mode` and a selector on the Search page override it. The retrieval evaluation reports the configured mode in the headline and adds a table with recall@5 and MRR for all three modes on the same queries, plus each query's first-relevant rank per mode.
- **Why:** hybrid is the usual safe choice, but this project only states measured numbers, so the evaluation has to show whether it actually helps on this library. Running all three on the same labels makes that a direct comparison.
- **Alternatives:** keep dense as the default until measured.
- **Trade-off:** with about 25 queries, small differences between modes are noise; the report says so. If hybrid does not beat dense on your labels, set `SEARCH_MODE=dense`.
