# Second Brain

A local app that saves long learning content (mostly YouTube), understands it, links related
items, and suggests what to learn next. Everything runs on your laptop except the LLM call,
and that can be switched to a local model.

| Feature | What you get |
|---------|--------------|
| **Save** | Paste a YouTube or article URL: metadata and transcript are fetched, chunked and embedded. Videos without captions can be transcribed locally with Whisper. |
| **Understand** | Summary, key points, topic and skill tags, difficulty — LLM output as schema-validated JSON. |
| **Search** | Hybrid search over everything saved: by meaning (embeddings) and by keywords (BM25), merged by rank. Shows the matching passage and a timestamp link. |
| **Connect** | Related items from text similarity and shared tags, plus a graph view. |
| **Recommend** | What to learn next, in order, with a one-line reason each, based on your goals, skills and finished items. |
| **Track** | to watch / watching / done. Finishing an item updates your skill levels. |

**Project status.** All seven build phases are complete and covered by offline tests. Two things
are still open and need your machine: the live checks listed in
[docs/phases/](docs/phases/) (real YouTube, real LLM, the UI in a browser), and the evaluation
numbers in [docs/EVAL.md](docs/EVAL.md), which only `scripts/eval.py` may write.

Docs: [design](docs/DESIGN.md) · [decision log](docs/DECISIONS.md) ·
[phase notes](docs/phases/) · [evaluation results](docs/EVAL.md) · [labelling guide](eval/README.md)

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
cp .env.example .env    # then add your ANTHROPIC_API_KEY (or switch to Ollama, see "The LLM")
```

`uv sync` downloads Python 3.12 itself (see `.python-version`); the system Python is not touched.
The first save or search downloads the embedding model (about 70 MB) into `data/models/`.

## Run the app

```bash
uv run streamlit run app/streamlit_app.py     # opens http://localhost:8501 in your browser
```

Pages (sidebar): **Add** a URL · **Library** with filters · **Item** (summary, progress, related
items, analyze / transcribe / delete) · **Search** · **Graph** · **Recommend** · **Profile**.

The first search or save loads the embedding model, which takes a few seconds. Saving and
analyzing run while the page waits; for many URLs or long Whisper jobs the command line is more
comfortable. Stop the app with Ctrl+C in the terminal. The app has no login: keep it on
`localhost`.

## Command line

The `sb` command does everything the app does, on the same database.

```bash
# Save and understand
uv run sb ingest "https://www.youtube.com/watch?v=VIDEO_ID"   # save one video
uv run sb ingest "https://example.com/some-article"           # articles work too
uv run sb ingest --file urls.txt --analyze                    # many URLs, and analyze each
uv run sb analyze 1                                           # summary, tags, difficulty for item 1
uv run sb analyze --all                                       # every item not analyzed yet
uv run sb list                                                # all saved items
uv run sb show 1                                              # details, analysis, stage timings
uv run sb delete 1                                            # remove an item completely

# Find and connect
uv run sb search "how do embeddings work"                     # best passage per item (hybrid)
uv run sb search "tsvector" --mode bm25                       # keywords only; or --mode dense
uv run sb related 1                                           # items linked to item 1
uv run sb graph --out graph.json                              # nodes + edges as JSON

# Track and plan
uv run sb status 3 done                                       # to_watch | watching | done
uv run sb profile add-goal "become an ML engineer"
uv run sb profile set-skill python 2                          # 0 none, 1 beginner, 2 intermediate, 3 advanced
uv run sb profile show
uv run sb recommend                                           # add --no-llm for rule-based reasons only

uv run sb --help
```

Item statuses: `embedded` (saved and searchable), `analyzed` (also summarized and tagged),
`no_transcript` (video has no captions), `failed` (the reason is printed and stored; run the same
command again to retry — finished stages are not repeated).

## The LLM

All LLM calls go through one provider interface, chosen in `.env`.

**Anthropic API (default).** Set `ANTHROPIC_API_KEY=...` in `.env`. The default model is the
low-cost `claude-haiku-5-5`; change it with `LLM_MODEL`.

**A local model with Ollama.** Nothing leaves your laptop.

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
you have pulled works. Small models make more JSON mistakes: failures are reported, never
hidden. Content longer than `LLM_MAX_INPUT_TOKENS` is summarized in parts and then combined, so
long videos take several calls.

## Videos without captions (optional Whisper)

```bash
uv sync --extra whisper                 # adds faster-whisper
uv run sb transcribe 3                  # item id from `sb list`
uv run sb ingest "<url>" --transcribe   # or: transcribe automatically when captions are missing
```

Default is the `small` model on CPU (`int8`). It works without any GPU setup but is slow for
long videos.

**On the GPU (RTX 3050, 4 GB).** Needs the NVIDIA driver (`nvidia-smi` must work) plus cuBLAS
and cuDNN 9 for CUDA 12:

```bash
uv sync --extra whisper-gpu
export LD_LIBRARY_PATH=$(uv run python -c 'import nvidia.cublas.lib, nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0] + ":" + nvidia.cudnn.lib.__path__[0])')
```

Then set `WHISPER_DEVICE=cuda` in `.env`. `LD_LIBRARY_PATH` must be set in every terminal that
runs Whisper. **These GPU steps were not verified in the build environment (no GPU there).** If
they fail, follow the GPU section of the
[faster-whisper README](https://github.com/SYSTRAN/faster-whisper#gpu) and set
`WHISPER_DEVICE=cpu` meanwhile.

## Search modes

| Mode | How it ranks | Good at |
|------|--------------|---------|
| `dense` | cosine similarity of embeddings (sqlite-vec) | the meaning, in other words than the content uses |
| `bm25` | BM25 keyword score (SQLite FTS5) | exact terms: names, acronyms, library names, error messages |
| `hybrid` (default) | both lists merged with reciprocal rank fusion: `Σ 1 / (RRF_K + rank)` | both |

Set the default with `SEARCH_MODE` in `.env`. The evaluation compares all three on your labels;
the defaults (`RRF_K=60`, `SEARCH_DEPTH=50`) are **not tuned on real data**.

## Related items

Two items are linked when `0.7 x text similarity + 0.3 x shared tags` reaches `LINK_THRESHOLD`
(tags = the topics and skills from the analysis). Links are updated automatically when an item
is saved or analyzed.

```bash
uv run sb links --stats      # score distribution over all pairs: use it to choose the threshold
uv run sb links --rebuild    # recompute all links after changing LINK_THRESHOLD or LINK_EMBED_WEIGHT
```

The default threshold (0.5) is a starting value that has **not been checked on real data**.
After saving 20-30 items, run `sb links --stats` and `sb related <id>` on a few items and adjust.

## Skills and recommendations

Skill levels come from the tags of the items you finish: 1 finished item = beginner; 2 with one
intermediate or harder = intermediate; 3 with one advanced = advanced. A level you set by hand
is never changed automatically.

Recommendations first score every unfinished, analyzed item on goal fit, novelty and readiness,
then ask the LLM to order the best ten and give a one-line reason each. If the LLM is missing or
answers badly you still get the rule-based order, with a warning.

## Evaluation

```bash
uv run sb eval-templates              # create the label files from your saved items
#   ...fill them in: see eval/README.md...
uv run python scripts/eval.py         # measure, and write docs/EVAL.md
```

The script reports retrieval recall@5 and MRR (for the configured search mode, plus a
comparison of dense, bm25 and hybrid on the same queries), linking precision and recall, summary
faithfulness (LLM judge with the rubric in `eval/RUBRIC.md`, plus agreement with your own spot
check), ingest time per stage, search latency p50/p95 and the LLM's JSON-validity rate — with
the date, commit, machine and configuration of the run. Anything it cannot measure is reported
as "Not run"; it never estimates.

## Tests

```bash
uv run pytest            # offline tests, no network needed
uv run pytest -m live -s # live checks: downloads the embedding model, makes one small LLM call
SB_LIVE_YOUTUBE_URL="https://www.youtube.com/watch?v=VIDEO_ID" uv run pytest -m live   # + real YouTube ingest
```

## Configuration

All settings live in `.env`; `.env.example` lists every one with its default.
`.env` and `data/` are git-ignored: your keys, database and saved items are never committed.

## Project layout

```
app/streamlit_app.py     the web UI (only draws widgets and calls the package)
src/secondbrain/
  config.py  db.py  items.py  vectors.py  embed.py      settings, storage, embeddings
  ingest/                                               URL parsing, YouTube, articles, Whisper, chunking, pipeline
  llm/                                                  provider interface, Anthropic + Ollama, validated calls
  schemas.py  prompts.py  analyze.py                    LLM output schemas, prompt text, analysis
  search.py  links.py                                   dense/BM25/hybrid search, related items, graph data
  profile.py  recommend.py  library.py                  skills and goals, recommendations, delete
  evalset.py  evaluation.py                             label templates, metrics, report
  ui.py  cli.py                                         UI helpers, command line
scripts/eval.py          runs the evaluation
eval/                    labelling guide, judge rubric, your label files
docs/                    design, decisions, phase notes, evaluation results
tests/                   offline tests (+ 3 live tests)
```

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
| Whisper `libcublas` / `libcudnn` error | Set `WHISPER_DEVICE=cpu`, or complete the GPU steps above. |
| App shows old settings | `.env` is read at start: stop the app (Ctrl+C) and start it again. |

## Known limitations

- Unofficial YouTube access can break when YouTube changes; keeping `yt-dlp` and
  `youtube-transcript-api` updated is part of running this.
- Link and recommendation weights are starting values, not tuned on data.
- Long operations block the page; there is no background queue.
- Single user, English-first (captions and the embedding model), tested on Linux only.
- Each phase note lists what was not verified outside the build environment.
