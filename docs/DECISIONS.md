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
