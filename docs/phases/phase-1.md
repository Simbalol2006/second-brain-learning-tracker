# Phase 1 — Ingest + storage

Date: 2026-11-16 · Built and tested on CPython 3.12.3 (uv-managed), Linux, CPU only.

## What was built

URL → metadata → transcript → chunks → embeddings → SQLite, driven from a CLI.

- **Config** from `.env` into one typed `Settings` object.
- **Database**: SQLite file with schema migrations and the sqlite-vec extension.
- **URL parsing**: YouTube URL forms (`watch`, `youtu.be`, `shorts`, `embed`, `live`) map to one video id; other http(s) URLs are treated as articles. Playlists/channels are rejected with a reason.
- **YouTube fetch**: metadata (yt-dlp, oEmbed fallback), captions (manual first, then auto).
- **Article fetch**: main text + metadata via trafilatura.
- **Optional Whisper** transcription for videos without captions.
- **Chunking** with overlap and timestamps.
- **Embeddings** on CPU and **vector storage** (per chunk and per item).
- **Pipeline** that resumes failed items and records per-stage timings.
- **CLI**: `sb ingest`, `sb transcribe`, `sb list`, `sb show`.

## Files

| File | Purpose |
|------|---------|
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencies and Python version |
| `.env.example`, `.gitignore`, `README.md` | Setup |
| `src/secondbrain/config.py` | Settings |
| `src/secondbrain/errors.py` | Exception types |
| `src/secondbrain/models.py` | `Segment`, `Chunk`, `Metadata`, `Transcript` |
| `src/secondbrain/db.py` | Connection, migrations, extension loading |
| `src/secondbrain/items.py` | SQL for items, transcripts, chunks, timings |
| `src/secondbrain/vectors.py` | sqlite-vec tables, insert, nearest-neighbour query |
| `src/secondbrain/embed.py` | `Embedder` protocol, fastembed wrapper, vector helpers |
| `src/secondbrain/ingest/urls.py` | URL classification |
| `src/secondbrain/ingest/youtube.py` | Metadata, captions, audio download |
| `src/secondbrain/ingest/article.py` | Article extraction |
| `src/secondbrain/ingest/transcribe.py` | Whisper wrapper |
| `src/secondbrain/ingest/chunking.py` | Chunking |
| `src/secondbrain/ingest/pipeline.py` | Stage orchestration |
| `src/secondbrain/cli.py` | Commands |
| `tests/` | 89 offline tests + 2 live tests |

## How to run

Setup is in the README. Then:

```bash
uv run pytest                      # offline tests
uv run sb ingest "<youtube url>"
uv run sb list
uv run sb show 1
```

## Test result

`uv run pytest` → **89 passed, 2 deselected** (the 2 deselected are the live tests).

## Not verified in the build environment — please run on the laptop

The build environment cannot reach YouTube or Hugging Face and has no GPU. The code paths below
are covered only by fakes so far. Each line is one command; tell me what fails.

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 1 | Embedding model downloads and works | `uv run pytest -m live` | 1 passed, 1 skipped |
| 2 | Real YouTube ingest (video with English captions) | `uv run sb ingest "<url>"` then `uv run sb show 1` | status `embedded`, title + duration filled, timings listed |
| 3 | Real article ingest | `uv run sb ingest "<article url>"` | status `embedded` |
| 4 | Video without captions | `uv run sb ingest "<url>"` | status `no_transcript` with a reason |
| 5 | Whisper on CPU *(optional)* | `uv sync --extra whisper && uv run sb transcribe <id>` | status `embedded`, source `whisper` |
| 6 | Whisper on GPU *(optional)* | README "Whisper on the GPU", then `sb transcribe <id>`; watch `nvidia-smi` | works; note peak VRAM |

What *was* observed here: with the network blocked, `sb ingest` on a YouTube URL exits with code 1,
prints the yt-dlp and oEmbed causes, and leaves the item as `failed` in `sb list` — the failure
path works as designed.

## How it connects to Phase 0

Implements Phase 1 of `docs/DESIGN.md` §10, including the stage-timing adjustment. Differences
from the design, each explained in `docs/DECISIONS.md`:

- Schema version is stored in `PRAGMA user_version`, not in `meta` (D5).
- New table `ingest_timings` (D11) and new initial status `new`.
- `WHISPER_DEVICE` defaults to `cpu`, not `auto` (D10).
- Only the tables Phase 1 needs exist so far (`items`, `transcripts`, `chunks`, `ingest_timings`, `meta`, vector tables). `analyses`, `topics`, `links` and profile tables arrive with their phases as new migrations.
- LLM settings are not in `.env.example` yet; they arrive in Phase 2.

What later phases will use: `items.load_segments()` (Phase 2 analysis input),
`vectors.knn_chunks()` and `vec_items` (Phase 3 search and links), `ingest_timings` (Phase 6
speed numbers), `items.progress_status` (Phase 4 tracking).

## Known limitations

- **Live behaviour unverified** — see the table above.
- No command to delete an item or to re-embed with a different model.
- Only caption languages listed in `CAPTION_LANGUAGES` are used; captions are never auto-translated.
- Articles that need JavaScript or a login yield no text and fail with a message.
- Article chunks overlap only when paragraphs are short (overlap works on whole segments).
- Chunk size is counted in words, not model tokens (D7).
- Ingest is synchronous: a long Whisper run blocks the command until it finishes.
- The test fixtures for captions are synthetic, not recorded from YouTube.
- Windows is untested.
