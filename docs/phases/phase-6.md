# Phase 6 — Evaluation and polish

Date: 2026-11-16 · Built and tested on CPython 3.12.3 (uv-managed), Linux, no GPU, no LLM access.

## Evaluation results

**There are none yet.** `docs/EVAL.md` is a placeholder that says so. The evaluation needs the
real embedding model, a real library, an LLM for the judge and your labels; none of these exist
in the build environment. The script that produces the results is finished and tested, and it is
the only thing that may write that file.

## What was built

- **`scripts/eval.py`** — runs the evaluation and writes `docs/EVAL.md`: a run table (date,
  commit, machine, GPU, package versions, library size, full configuration), a summary table and
  one section per area.
- **`src/secondbrain/evaluation.py`** — the measuring code:
  - *Retrieval*: recall@5, MRR, per-query table, search latency p50/p95.
  - *Linking*: precision, recall, confusion counts, plus an information-only threshold table.
  - *Summary faithfulness*: claim-level LLM judge with the rubric, long sources judged in parts,
    cached verdicts, human spot-check file and judge-vs-human agreement.
  - *Ingest speed*: per-stage and per-item times from the timings recorded at ingest.
  - *LLM output*: JSON-valid-on-first-try rate and tokens, from the call log.
  - Anything that cannot be measured is reported as **Not run** with the reason.
- **`eval/RUBRIC.md`** — the written rubric; it is both the judge's instructions and your guide
  for the spot check.
- **`eval/README.md`** — extended with the run and spot-check steps.
- **Polish**: `sb delete` and a Delete control on the Item page; README rewritten as the final
  project page (features, layout, limitations); DESIGN.md marked approved.

## Files

| File | Change |
|------|--------|
| `scripts/eval.py` | new |
| `src/secondbrain/evaluation.py` | new |
| `eval/RUBRIC.md` | new |
| `docs/EVAL.md` | new — placeholder, no numbers |
| `src/secondbrain/library.py` | new — `delete_item` |
| `src/secondbrain/vectors.py` | `delete_item` |
| `src/secondbrain/cli.py`, `app/streamlit_app.py` | delete command / control |
| `eval/README.md`, `README.md`, `docs/DESIGN.md` | updated |
| `tests/test_evaluation.py`, `tests/test_library.py` | new |
| `tests/test_cli.py`, `tests/test_app.py` | extended |

## How to run

```bash
uv sync
uv run pytest
uv run sb eval-templates            # after saving + analyzing ~30-40 items and freezing LINK_THRESHOLD
#   fill in eval/retrieval.yaml, eval/links.yaml, eval/summaries.yaml  (eval/README.md)
uv run python scripts/eval.py       # writes docs/EVAL.md and creates eval/spot_check.yaml
#   fill in eval/spot_check.yaml
uv run python scripts/eval.py       # adds judge-vs-human agreement (judge verdicts come from the cache)
```

Options: `--skip-summaries` (no judge calls), `--judge-model <name>`, `--rejudge`,
`--repeats <n>`, `--eval-dir`, `--out`.

## Test result

`uv run pytest` → **336 passed, 3 deselected** (the 3 deselected are the live tests).

The 26 evaluation tests check the measuring code on tiny made-up libraries with a fake embedder
and a scripted judge: metric formulas, label loading, every "Not run" path, the judge cache, the
blind spot-check file, agreement counting, and that an empty database yields a report with no
metric values at all. They do not, and cannot, say anything about how good the app is.

## What you need to do to get the numbers

1. Run the live checks from the earlier phase notes first (they decide whether ingest, the LLM
   call and the UI work on your machine at all). In order of importance:
   `phase-1.md` checks 1–2, `phase-2.md` check 1, `phase-5.md` checks 1–3.
2. Save and analyze 30–40 items: `uv run sb ingest --file urls.txt --analyze`.
3. Send me `uv run sb links --stats`; we fix `LINK_THRESHOLD`, then `uv run sb links --rebuild`.
4. `uv run sb eval-templates`, label the three files, run `scripts/eval.py`, do the spot check,
   run it again.
5. Commit `docs/EVAL.md` and the label files.

If any step fails, send me the error text. Fixes for problems found in live use are expected:
this code has never touched real YouTube, a real LLM or a browser.

## How it connects to earlier phases

- Uses `search.search` (Phase 3) for retrieval and latency, `links.pair_score` (Phase 3) for
  linking, `analyze.stored_analysis` (Phase 2) for the claims, the provider interface and
  `call_validated` (Phases 2, 4) for the judge, `ingest_timings` (Phase 1) and `llm_calls`
  (Phase 2) for speed and validity, and the label templates from Phase 3.
- Implements DESIGN.md §11. Differences: speed is read from stored timings rather than measured
  in the eval run (D59); the linking section adds an information-only threshold table (D55).

## Known limitations

- **No results yet.**
- Small label sets: every metric is rough. The report prints the counts.
- Linking precision/recall describe a stratified sample of pairs, not all pairs.
- The judge is an LLM. By default it is the same model that wrote the summaries.
- Recommendations are not evaluated (no labels for them were planned).
- Ingest times mix whatever conditions the items were saved under.
- Search latency excludes model loading and page rendering.
- Not verified live, across all phases: YouTube and article ingest, Whisper (CPU and GPU), both
  LLM providers, the real embedding model's search and link behaviour, and the UI in a browser.
