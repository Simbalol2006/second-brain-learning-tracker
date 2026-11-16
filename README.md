# Second Brain

Save long learning content (mostly YouTube), understand it, link related items, and get
"what to learn next" suggestions. Everything runs on your laptop except the LLM call.

**Status: Phase 1 of 6 — ingest + storage.** You can save a URL and it is fetched, chunked,
embedded and stored. Summaries, search, links, recommendations and the UI come in later phases.
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
cp .env.example .env    # defaults work as they are
```

`uv sync` downloads Python 3.12 itself (see `.python-version`); the system Python is not touched.
The first `sb ingest` downloads the embedding model (about 70 MB) into `data/models/`.

## Run

```bash
uv run sb ingest "https://www.youtube.com/watch?v=VIDEO_ID"   # save one video
uv run sb ingest "https://example.com/some-article"           # articles work too
uv run sb ingest --file urls.txt                              # one URL per line, # for comments
uv run sb list                                                # all saved items
uv run sb show 1                                              # details + stage timings for item 1
uv run sb --help
```

Ingest statuses: `embedded` (done for this phase), `no_transcript` (video has no captions),
`failed` (the reason is printed and stored; run the same `sb ingest` again to retry — finished
stages are not repeated).

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
uv run pytest -m live    # live checks: downloads the embedding model
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
| Whisper `libcublas` / `libcudnn` error | Set `WHISPER_DEVICE=cpu`, or complete "Whisper on the GPU". |

Windows is not tested yet; development targets Kubuntu.
