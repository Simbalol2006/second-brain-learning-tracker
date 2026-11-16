# Second Brain

Save long learning content (mostly YouTube), understand it, link related items, and get
"what to learn next" suggestions. Everything runs on your laptop except the LLM call.

**Status: Phase 3 of 6 — ingest, understand, search, connect.** You can save a URL, have an
LLM summarize and tag it, search everything by meaning, and see which items are related.
Recommendations, progress tracking and the UI come in later phases.
Design: [docs/DESIGN.md](docs/DESIGN.md) · Decisions: [docs/DECISIONS.md](docs/DECISIONS.md) ·
Phase notes: [docs/phases/](docs/phases/)

## Setup (fresh Kubuntu 26.04)

```bash
# 1. System tools
sudo apt update && sudo apt install -y git curl unzip

# 2. uv (Python package + version manager). Open a new terminal afterwards.
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Deno (JavaScript runtime that yt-dlp needs for YouTube). Open a new terminal afterwards.
curl -fsSL https://deno.land/install.sh | sh

# 4. Project
git clone <your-repo-url> second-brain && cd second-brain
uv sync                 # installs Python 3.12 and all dependencies into .venv
cp .env.example .env    # then add your ANTHROPIC_API_KEY (or switch to Ollama, see below)
```

`uv sync` downloads Python 3.12 itself (see `.python-version`); the system Python is not touched.
The first `sb ingest` downloads the embedding model (about 70 MB) into `data/models/`.

## Run

```bash
uv run sb ingest "https://www.youtube.com/watch?v=VIDEO_ID"   # save one video
uv run sb ingest "https://example.com/some-article"           # articles work too
uv run sb ingest --file urls.txt                              # one URL per line, # for comments
uv run sb list                                                # all saved items
uv run sb analyze 1                                           # LLM summary, tags, difficulty for item 1
uv run sb analyze --all                                       # every item not analyzed yet
uv run sb ingest "<url>" --analyze                            # save and analyze in one go
uv run sb show 1                                              # details, analysis and stage timings
uv run sb search "how do embeddings work"                     # semantic search, best passage per item
uv run sb related 1                                           # items linked to item 1
uv run sb graph --out graph.json                              # nodes + edges for a graph view
uv run sb --help
```

Statuses: `embedded` (saved and searchable), `analyzed` (also summarized and tagged),
`no_transcript` (video has no captions), `failed` (the reason is printed and stored; run the same
command again to retry — finished stages are not repeated).

## Related items (links)

Two items are linked when `0.7 x text similarity + 0.3 x shared tags` reaches `LINK_THRESHOLD`
(tags = the topics and skills from the analysis). Links are updated automatically when an item
is ingested or analyzed.

```bash
uv run sb links --stats      # score distribution over all pairs: use it to choose the threshold
uv run sb links --rebuild    # recompute all links after changing LINK_THRESHOLD or LINK_EMBED_WEIGHT
```

The default threshold (0.5) is a starting value that has **not been checked on real data**.
After saving 20-30 items, run `sb links --stats` and `sb related <id>` on a few items and adjust.
If you used the app before Phase 3, run `sb links --rebuild` once.

## Evaluation labels

`uv run sb eval-templates` writes three files into `eval/` from your saved items. How to fill
them in: [eval/README.md](eval/README.md).

## The LLM (summaries and tags)

All LLM calls go through one provider interface, chosen in `.env`.

**Anthropic API (default).** Set `ANTHROPIC_API_KEY=...` in `.env`. The default model is the
low-cost `claude-haiku-5-5`; change it with `LLM_MODEL`.

**Using a local model (Ollama).** Nothing leaves your laptop.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b          # example of a model small enough for 4 GB VRAM
```

Then in `.env`:

```
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
LLM_MAX_INPUT_TOKENS=4000
LLM_MAX_OUTPUT_TOKENS=1024
```

The model name is only an example and was **not tested in the build environment**; any model
you have pulled works. Small models make more JSON mistakes: `sb analyze` reports failures, and
`sb show <id>` shows what was stored. Content longer than `LLM_MAX_INPUT_TOKENS` is summarized
in parts and then combined, so long videos take several calls.

### Videos without captions (optional Whisper)

```bash
uv sync --extra whisper                 # adds faster-whisper
uv run sb transcribe 3                  # item id from `sb list`
uv run sb ingest "<url>" --transcribe   # or: transcribe automatically when captions are missing
```

Default is the `small` model on CPU (`int8`). It works without any GPU setup but is slow for
long videos.

### Whisper on the GPU (RTX 3050, 4 GB)

Needs the NVIDIA driver (`nvidia-smi` must work) plus cuBLAS and cuDNN 9 for CUDA 12:

```bash
uv sync --extra whisper-gpu
export LD_LIBRARY_PATH=$(uv run python -c 'import nvidia.cublas.lib, nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0] + ":" + nvidia.cudnn.lib.__path__[0])')
```

Then set `WHISPER_DEVICE=cuda` in `.env`. `LD_LIBRARY_PATH` must be set in every terminal that
runs Whisper. **These GPU steps were not verified in the build environment (no GPU there).** If
they fail, follow the GPU section of the
[faster-whisper README](https://github.com/SYSTRAN/faster-whisper#gpu) and set
`WHISPER_DEVICE=cpu` meanwhile.

## Tests

```bash
uv run pytest            # offline tests, no network needed
uv run pytest -m live -s # live checks: downloads the embedding model, makes one small LLM call
SB_LIVE_YOUTUBE_URL="https://www.youtube.com/watch?v=VIDEO_ID" uv run pytest -m live   # + real YouTube ingest
```

## Configuration

All settings live in `.env` (see `.env.example` for the full list with defaults).
`.env` and `data/` are git-ignored: your saved items, database and keys are never committed.

## Troubleshooting

| Message | What to do |
|---------|-----------|
| `yt-dlp needs an external JavaScript runtime` / metadata failed | Check `deno --version` works in the same terminal. Upgrade yt-dlp: `uv lock --upgrade-package yt-dlp && uv sync`. |
| `YouTube blocked the caption request from this IP` | Common on VPN/cloud IPs. Switch network or retry later. |
| `No captions in languages ['en']` | The video has no English captions. Use `sb transcribe <id>`, or set `CAPTION_LANGUAGES=en,hi`. |
| `Could not load the sqlite-vec extension` | You are not using the uv-managed Python. Run commands with `uv run`. |
| `The database holds vectors from ...` | `EMBEDDING_MODEL` was changed. Change it back, or use a fresh `DATA_DIR` and ingest again. |
| `ANTHROPIC_API_KEY is not set` | Add the key to `.env`, or switch to Ollama (see "The LLM"). |
| `Ollama is not reachable` / `does not have the model` | Run `ollama serve`, and `ollama pull <model>`. |
| `The model returned invalid output 2 times` | The model could not follow the JSON schema. Retry, or use a stronger model. |
| `ran out of output tokens` / `hit its output limit` | Raise `LLM_MAX_OUTPUT_TOKENS`. |
| `sb related` shows nothing, or far too much | Check `sb links --stats`, change `LINK_THRESHOLD`, then `sb links --rebuild`. |
| Whisper `libcublas` / `libcudnn` error | Set `WHISPER_DEVICE=cpu`, or complete "Whisper on the GPU". |

Windows is not tested yet; development targets Kubuntu.
