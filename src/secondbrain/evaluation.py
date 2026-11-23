"""Evaluation: measure the app against hand-written labels and stored timings.

Every number in the report is computed here from a real run. A part that cannot
be measured (labels missing, no LLM, no data) is reported as "Not run" with the
reason; nothing is estimated or filled in.

Entry point: run() returns a list of Section objects; render() turns them into
the Markdown written to docs/EVAL.md by scripts/eval.py.
"""

import hashlib
import json
import logging
import platform
import re
import sqlite3
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

from secondbrain import analyze, evalset, items, links, prompts, search
from secondbrain.config import Settings
from secondbrain.embed import Embedder
from secondbrain.errors import InvalidUrlError, SecondBrainError
from secondbrain.ingest.urls import parse_url
from secondbrain.llm.base import LLMProvider
from secondbrain.llm.validated import call_validated

logger = logging.getLogger(__name__)

SPOT_CHECK_FILE = "spot_check.yaml"
RUBRIC_FILE = "RUBRIC.md"
JUDGE_CACHE_FILE = "eval_judge_cache.json"
RANK_CUTOFF = 20  # MRR looks at the first 20 results
VERDICTS = ("supported", "not_supported", "contradicted")
Verdict = Literal["supported", "not_supported", "contradicted"]


class NotRun(Exception):
    """A section could not be measured. The message is shown in the report."""


@dataclass
class Section:
    title: str
    body: str = ""  # Markdown
    headline: dict[str, str] = field(default_factory=dict)  # metric name -> value, for the summary table
    not_run: str | None = None


# ---------- small metric functions ----------


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """Share of the relevant items that appear in the first k results."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    """1 / position of the first relevant result; 0 if none is in the list."""
    for position, item_id in enumerate(ranked, start=1):
        if item_id in relevant:
            return 1.0 / position
    return 0.0


def confusion(labels: list[bool], predictions: list[bool]) -> dict[str, int]:
    pairs = list(zip(labels, predictions))
    return {
        "tp": sum(1 for label, pred in pairs if label and pred),
        "fp": sum(1 for label, pred in pairs if not label and pred),
        "fn": sum(1 for label, pred in pairs if label and not pred),
        "tn": sum(1 for label, pred in pairs if not label and not pred),
    }


def ratio(numerator: int, denominator: int) -> str:
    """'0.750 (3/4)', or 'n/a (0/0)' when there is nothing to divide by."""
    if denominator == 0:
        return f"n/a ({numerator}/0)"
    return f"{numerator / denominator:.3f} ({numerator}/{denominator})"


def split_claims(summary: str, key_points: list[str]) -> list[str]:
    """The checkable statements of an analysis: summary sentences, then key points."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
    return sentences + [p.strip() for p in key_points if p.strip()]


def combine_verdicts(per_window: list[list[Verdict]]) -> list[Verdict]:
    """One verdict per claim from several source parts.

    supported if any part supports it; otherwise contradicted if any part
    contradicts it; otherwise not_supported.
    """
    combined: list[Verdict] = []
    for verdicts in zip(*per_window):
        if "supported" in verdicts:
            combined.append("supported")
        elif "contradicted" in verdicts:
            combined.append("contradicted")
        else:
            combined.append("not_supported")
    return combined


# ---------- label files ----------


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise NotRun(f"`{path}` does not exist. Create the label files with `sb eval-templates`.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise NotRun(f"`{path}` is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise NotRun(f"`{path}` does not have the expected structure.")
    return data


def _item_id(conn: sqlite3.Connection, url: object) -> int | None:
    try:
        source_type, source_id, _ = parse_url(str(url))
    except InvalidUrlError:
        return None
    row = conn.execute(
        "SELECT id FROM items WHERE source_type = ? AND source_id = ?", (source_type, source_id)
    ).fetchone()
    return int(row[0]) if row else None


def _resolve(conn: sqlite3.Connection, urls: list[object], path: Path) -> list[int]:
    ids = [_item_id(conn, url) for url in urls]
    unknown = [str(url) for url, item_id in zip(urls, ids) if item_id is None]
    if unknown:
        raise NotRun(f"`{path}` names URLs that are not saved items: {', '.join(unknown[:5])}")
    return [item_id for item_id in ids if item_id is not None]


# ---------- sections ----------


def eval_retrieval(
    conn: sqlite3.Connection,
    get_embedder: Callable[[], Embedder],
    path: Path,
    repeats: int,
    settings: Settings | None = None,
) -> Section:
    """Score the configured search mode, and compare all three modes on the same labels.

    The headline numbers are for SEARCH_MODE. The comparison table runs dense, bm25 and hybrid
    over the same queries so the effect of adding BM25 can be read off directly.
    """
    section = Section("Retrieval")
    entries = [q for q in _load_yaml(path).get("queries") or [] if q.get("query") and q.get("relevant")]
    if not entries:
        raise NotRun(f"`{path}` has no queries with relevant items yet.")
    configured = settings.search_mode if settings else "hybrid"
    options = {"rrf_k": settings.rrf_k, "depth": settings.search_depth} if settings else {}
    embedder = get_embedder()  # loaded only once there is something to measure
    search.search(conn, embedder, "warm up", limit=5, mode=configured, **options)  # one-time costs

    per_mode: dict[str, tuple[list[float], list[float]]] = {m: ([], []) for m in search.SEARCH_MODES}
    rows, timings = [], []
    for entry in entries:
        relevant = set(_resolve(conn, list(entry["relevant"]), path))
        firsts: dict[str, str] = {}
        for mode in search.SEARCH_MODES:
            ranked = [
                hit.item_id
                for hit in search.search(conn, embedder, entry["query"], limit=RANK_CUTOFF, mode=mode, **options)
            ]
            recall, rrank = recall_at_k(ranked, relevant, 5), reciprocal_rank(ranked, relevant)
            per_mode[mode][0].append(recall)
            per_mode[mode][1].append(rrank)
            firsts[mode] = f"{round(1 / rrank)}" if rrank else f"not in top {RANK_CUTOFF}"
        for _ in range(repeats):
            start = time.perf_counter()
            search.search(conn, embedder, entry["query"], limit=5, mode=configured, **options)
            timings.append((time.perf_counter() - start) * 1000)
        recall = per_mode[configured][0][-1]
        rows.append(
            f"| {_cell(entry['query'])} | {len(relevant)} | {recall:.2f} | {firsts[configured]} | "
            f"{firsts['dense']} | {firsts['bm25']} | {firsts['hybrid']} |"
        )

    recalls, rranks = per_mode[configured]
    p50, p95 = np.percentile(timings, 50), np.percentile(timings, 95)
    section.headline = {
        "Retrieval recall@5": f"{np.mean(recalls):.3f} (mean over {len(entries)} queries, {configured})",
        "Retrieval MRR": f"{np.mean(rranks):.3f} (mean over {len(entries)} queries, {configured})",
        "Search latency p50 / p95": f"{p50:.1f} ms / {p95:.1f} ms ({len(timings)} timed searches)",
    }
    missed = sum(1 for r in rranks if r == 0)
    comparison = [
        f"| {mode}{' (configured)' if mode == configured else ''} | {np.mean(r):.3f} | {np.mean(m):.3f} | "
        f"{sum(1 for x in m if x == 0)} |"
        for mode, (r, m) in per_mode.items()
    ]
    section.body = "\n".join(
        [
            f"- Queries: **{len(entries)}**",
            f"- Search mode: **{configured}**"
            + (f" (RRF k {settings.rrf_k}, depth {settings.search_depth})" if settings and configured == "hybrid" else ""),
            f"- recall@5 (mean): **{np.mean(recalls):.3f}**",
            f"- MRR (first relevant result within top {RANK_CUTOFF}; mean): **{np.mean(rranks):.3f}**",
            f"- Queries with no relevant item in the top {RANK_CUTOFF}: **{missed}**",
            f"- Search latency over {len(timings)} timed searches ({repeats} per query, model already "
            f"loaded, includes embedding the query): p50 **{p50:.1f} ms**, p95 **{p95:.1f} ms**",
            "",
            "recall@5 = relevant items found in the top 5 ÷ all relevant items for that query.",
            "",
            "### Search modes compared on the same queries",
            "",
            "| Mode | recall@5 | MRR | Queries with no relevant item in the top "
            f"{RANK_CUTOFF} |",
            "|---|---|---|---|",
            *comparison,
            "",
            f"With {len(entries)} queries, one query moves recall@5 by up to {1 / len(entries):.3f}: "
            "read small differences between modes as noise.",
            "",
            "### Per query",
            "",
            "| Query | Relevant items | recall@5 | Rank of first relevant | dense | bm25 | hybrid |",
            "|---|---|---|---|---|---|---|",
            *rows,
        ]
    )
    return section


def eval_links(conn: sqlite3.Connection, settings: Settings, path: Path, sweep: bool = True) -> Section:
    section = Section("Linking")
    pairs = _load_yaml(path).get("pairs") or []
    labelled = [p for p in pairs if isinstance(p.get("related"), bool)]
    if not labelled:
        raise NotRun(f"`{path}` has no pairs with `related: true/false` yet ({len(pairs)} pairs unlabelled).")
    ids = _resolve(conn, [url for p in labelled for url in (p["a"], p["b"])], path)
    scores = [links.pair_score(conn, settings, ids[i], ids[i + 1]).score for i in range(0, len(ids), 2)]
    labels = [bool(p["related"]) for p in labelled]
    counts = confusion(labels, [score >= settings.link_threshold for score in scores])
    precision = ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = ratio(counts["tp"], counts["tp"] + counts["fn"])
    section.headline = {"Linking precision": precision, "Linking recall": recall}
    lines = [
        f"- Labelled pairs: **{len(labelled)}** of {len(pairs)} ({sum(labels)} related, {len(labels) - sum(labels)} unrelated)",
        f"- Predicted related when score ≥ `LINK_THRESHOLD` = {settings.link_threshold} "
        f"(text weight {settings.link_embed_weight})",
        f"- Precision: **{precision}** · Recall: **{recall}**",
        f"- Counts: TP {counts['tp']}, FP {counts['fp']}, FN {counts['fn']}, TN {counts['tn']}",
        "",
        "The pairs are not a random sample: a third are the highest-scoring pairs, a third come "
        "from the middle of the score range and a third are random (see `eval/README.md`). "
        "Precision and recall describe this sample, not all pairs in the library.",
    ]
    if sweep:
        lines += [
            "",
            "**For information only — other thresholds on the same labelled pairs.** Picking a "
            "threshold from this table is tuning on the eval set; the headline numbers above use "
            "the configured threshold.",
            "",
            "| Threshold | Precision | Recall |",
            "|---|---|---|",
        ]
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            c = confusion(labels, [score >= threshold for score in scores])
            lines.append(f"| {threshold:.1f} | {ratio(c['tp'], c['tp'] + c['fp'])} | {ratio(c['tp'], c['tp'] + c['fn'])} |")
    section.body = "\n".join(lines)
    return section


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: int
    verdict: Verdict
    evidence: str


class JudgeAnswer(BaseModel):
    """The judge's verdicts. Validation context carries 'n_claims'."""

    model_config = ConfigDict(extra="forbid")
    verdicts: list[ClaimVerdict]

    @model_validator(mode="after")
    def _one_verdict_per_claim(self, info: ValidationInfo) -> "JudgeAnswer":
        expected = (info.context or {}).get("n_claims")
        numbers = sorted(v.claim for v in self.verdicts)
        if expected is not None and numbers != list(range(1, expected + 1)):
            raise ValueError(f"need exactly one verdict for each claim number 1..{expected}, got {numbers}")
        return self


def judge_claims(
    conn: sqlite3.Connection,
    settings: Settings,
    judge: LLMProvider,
    rubric: str,
    item_id: int,
    claims: list[str],
) -> list[Verdict]:
    """Ask the judge about every claim, part by part if the source is long."""
    segments = items.load_segments(conn, item_id) or []
    claim_text = "\n".join(f"{n}. {claim}" for n, claim in enumerate(claims, start=1))
    budget = int(settings.llm_max_input_tokens * analyze.CHARS_PER_TOKEN * analyze.BUDGET_SAFETY)
    budget -= len(rubric) + len(claim_text) + 500
    if budget < analyze.MIN_WINDOW_CHARS:
        raise NotRun("LLM_MAX_INPUT_TOKENS is too small for the judge prompt (rubric + claims). Raise it.")
    windows = [" ".join(group) for group in analyze.pack([s.text for s in segments], budget)]
    per_window: list[list[Verdict]] = []
    for number, window in enumerate(windows, start=1):
        part = "" if len(windows) == 1 else f" (part {number} of {len(windows)} of a longer text)"
        user = f"SOURCE{part}:\n{window}\n\nCLAIMS:\n{claim_text}"
        answer = call_validated(
            conn, judge, "judge", rubric, user, JudgeAnswer, item_id=item_id, context={"n_claims": len(claims)}
        )
        ordered = sorted(answer.verdicts, key=lambda v: v.claim)
        per_window.append([v.verdict for v in ordered])
    return combine_verdicts(per_window)


def eval_summaries(
    conn: sqlite3.Connection,
    settings: Settings,
    get_judge: Callable[[], LLMProvider],
    eval_dir: Path,
    rejudge: bool = False,
) -> Section:
    section = Section("Summary faithfulness")
    path = eval_dir / evalset.SUMMARIES_FILE
    entries = [e for e in _load_yaml(path).get("items") or [] if e.get("url")]
    if not entries:
        raise NotRun(f"`{path}` lists no items.")
    rubric_path = eval_dir / RUBRIC_FILE
    if not rubric_path.exists():
        raise NotRun(f"`{rubric_path}` is missing; the judge needs the written rubric.")
    rubric = rubric_path.read_text(encoding="utf-8")
    ids = _resolve(conn, [e["url"] for e in entries], path)
    try:
        judge = get_judge()
    except SecondBrainError as exc:
        raise NotRun(f"No LLM judge available: {exc}") from exc

    cache_path = settings.data_dir / JUDGE_CACHE_FILE
    cache: dict[str, list[Verdict]] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    judged: dict[str, tuple[list[str], list[Verdict]]] = {}
    rows, totals, reused = [], dict.fromkeys(VERDICTS, 0), 0
    for entry, item_id in zip(entries, ids):
        stored = analyze.stored_analysis(conn, item_id)
        if stored is None:
            raise NotRun(f"Item {entry['url']} has no analysis. Run `sb analyze --all` first.")
        claims = split_claims(stored.summary, stored.key_points)
        key = _cache_key(judge, rubric, str(entry["url"]), claims)
        if key in cache and not rejudge:
            reused += 1
        else:
            try:
                cache[key] = judge_claims(conn, settings, judge, rubric, item_id, claims)
            except SecondBrainError as exc:
                raise NotRun(f"The judge failed on {entry['url']}: {exc}") from exc
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        verdicts = cache[key]
        judged[str(entry["url"])] = (claims, verdicts)
        counts = {v: verdicts.count(v) for v in VERDICTS}
        for verdict in VERDICTS:
            totals[verdict] += counts[verdict]
        title = items.get(conn, item_id)["title"] or entry["url"]
        rows.append(
            f"| {_cell(title)} | {len(claims)} | {counts['supported']} | {counts['not_supported']} | "
            f"{counts['contradicted']} | {counts['supported'] / len(claims):.2f} |"
        )

    all_claims = sum(totals.values())
    faithfulness = ratio(totals["supported"], all_claims)
    section.headline = {"Summary faithfulness (LLM judge)": faithfulness}
    spot_lines, agreement = _spot_check(eval_dir, entries, judged)
    if agreement:
        section.headline["Judge vs human agreement"] = agreement
    section.body = "\n".join(
        [
            f"- Items judged: **{len(entries)}** ({reused} taken from the judge cache, {len(entries) - reused} judged in this run)",
            f"- Judge: `{judge.name}` / `{judge.model}`, rubric `{rubric_path}`",
            f"- Claims (summary sentences + key points): **{all_claims}**",
            f"- Faithfulness = supported ÷ all claims: **{faithfulness}**",
            f"- Not supported: {totals['not_supported']} · Contradicted: {totals['contradicted']}",
            "",
            "For texts longer than the judge's input budget, each part is judged separately: a claim "
            "is supported if any part supports it, otherwise contradicted if any part contradicts it.",
            "",
            *spot_lines,
            "",
            "| Item | Claims | Supported | Not supported | Contradicted | Faithfulness |",
            "|---|---|---|---|---|---|",
            *rows,
        ]
    )
    return section


def _cache_key(judge: LLMProvider, rubric: str, url: str, claims: list[str]) -> str:
    payload = json.dumps([judge.name, judge.model, rubric, url, claims], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _spot_check(
    eval_dir: Path, entries: list[dict[str, Any]], judged: dict[str, tuple[list[str], list[Verdict]]]
) -> tuple[list[str], str | None]:
    """Create or read the human spot-check file and compare it with the judge."""
    chosen = [str(e["url"]) for e in entries if e.get("spot_check") is True]
    path = eval_dir / SPOT_CHECK_FILE
    if not chosen:
        return ["**Human spot check:** not done — no item in `summaries.yaml` has `spot_check: true`."], None
    if not path.exists():
        template = {"items": [{"url": url, "claims": [{"text": c, "human": None} for c in judged[url][0]]} for url in chosen]}
        header = (
            "# Human spot check. For every claim set `human:` to one of\n"
            "#   supported | not_supported | contradicted\n"
            "# following eval/RUBRIC.md, by checking the claim against the video or article.\n"
            "# The judge's verdicts are not shown here on purpose. Then run scripts/eval.py again.\n\n"
        )
        path.write_text(header + yaml.safe_dump(template, allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")
        return [f"**Human spot check:** `{path}` was created in this run. Fill it in and run the eval again."], None

    human_items = {str(i.get("url")): i.get("claims") or [] for i in (_load_yaml(path).get("items") or [])}
    total = same = same_binary = 0
    for url in chosen:
        judge_by_text = dict(zip(*judged[url]))
        for claim in human_items.get(url, []):
            label, text = claim.get("human"), claim.get("text")
            if label in VERDICTS and text in judge_by_text:
                total += 1
                same += label == judge_by_text[text]
                same_binary += (label == "supported") == (judge_by_text[text] == "supported")
    if total == 0:
        return [f"**Human spot check:** `{path}` has no usable human labels yet (set `human:` for each claim)."], None
    agreement = ratio(same, total)
    return [
        f"**Human spot check:** {total} claims from {len(chosen)} items labelled by hand.",
        f"- Judge and human give the same verdict: **{agreement}**",
        f"- Same when reduced to supported / not supported: **{ratio(same_binary, total)}**",
    ], agreement


def eval_speed(conn: sqlite3.Connection) -> Section:
    section = Section("Ingest speed")
    rows = conn.execute(
        """
        SELECT t.item_id, t.stage, t.seconds FROM ingest_timings t
        JOIN (SELECT max(id) AS id FROM ingest_timings GROUP BY item_id, stage) latest ON latest.id = t.id
        """
    ).fetchall()
    if not rows:
        raise NotRun("No ingest timings are stored yet (nothing has been ingested).")
    by_stage: dict[str, list[float]] = {}
    by_item: dict[int, float] = {}
    for item_id, stage, seconds in rows:
        by_stage.setdefault(stage, []).append(float(seconds))
        by_item[int(item_id)] = by_item.get(int(item_id), 0.0) + float(seconds)
    totals = list(by_item.values())
    section.headline = {
        "Ingest time per item (median / p95)": f"{np.median(totals):.1f} s / {np.percentile(totals, 95):.1f} s ({len(totals)} items)"
    }
    table = [
        f"| {stage} | {len(v)} | {np.median(v):.2f} | {np.percentile(v, 95):.2f} | {max(v):.2f} |"
        for stage, v in sorted(by_stage.items())
    ]
    minutes = [r[0] / 60 for r in conn.execute("SELECT duration_sec FROM items WHERE duration_sec IS NOT NULL")]
    length = f"{np.median(minutes):.0f} min median, {max(minutes):.0f} min longest" if minutes else "unknown"
    section.body = "\n".join(
        [
            f"- Items with timings: **{len(totals)}** (content length: {length})",
            f"- Total per item, all recorded stages added up: median **{np.median(totals):.1f} s**, "
            f"p95 **{np.percentile(totals, 95):.1f} s**, max **{max(totals):.1f} s**",
            "",
            "These times were recorded when each item was saved (latest run of each stage), on the "
            "machine and network used then. `analyze` is the LLM time; `whisper` appears only for "
            "items transcribed locally.",
            "",
            "| Stage | Items | Median s | p95 s | Max s |",
            "|---|---|---|---|---|",
            *table,
        ]
    )
    return section


def eval_llm_output(conn: sqlite3.Connection) -> Section:
    section = Section("LLM output validity and tokens")
    rows = conn.execute(
        """
        SELECT provider, model, purpose,
               sum(attempt = 1) AS first_tries, sum(attempt = 1 AND valid = 1) AS first_valid,
               count(*) AS calls, sum(valid = 0) AS invalid,
               coalesce(sum(input_tokens), 0) AS tokens_in, coalesce(sum(output_tokens), 0) AS tokens_out
        FROM llm_calls GROUP BY provider, model, purpose ORDER BY provider, model, purpose
        """
    ).fetchall()
    if not rows:
        raise NotRun("No LLM calls are logged yet.")
    app_rows = [r for r in rows if r["purpose"] != "judge"]  # the judge is part of the eval, not the app
    first_tries = sum(r["first_tries"] for r in app_rows)
    first_valid = sum(r["first_valid"] for r in app_rows)
    section.headline = {"JSON valid on first try (app calls)": ratio(first_valid, first_tries)}
    per_item = conn.execute(
        """
        SELECT coalesce(sum(input_tokens), 0), coalesce(sum(output_tokens), 0) FROM llm_calls
        WHERE item_id IS NOT NULL AND purpose IN ('map', 'collapse', 'final') GROUP BY item_id
        """
    ).fetchall()
    lines = [
        f"- Schema-valid answers on the first try, app calls (analysis and recommendations): "
        f"**{ratio(first_valid, first_tries)}**",
    ]
    if per_item:
        lines.append(
            f"- Analysis tokens per item (all analysis calls ever made for it, {len(per_item)} items): "
            f"median **{np.median([r[0] for r in per_item]):.0f} in / {np.median([r[1] for r in per_item]):.0f} out**"
        )
    lines += [
        "",
        "From the `llm_calls` log: every call made with this database, not only this run. "
        "`judge` rows are the eval's own calls.",
        "",
        "| Provider | Model | Step | Calls | Valid on first try | Invalid answers | Tokens in | Tokens out |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['provider']} | {r['model']} | {r['purpose']} | {r['calls']} | "
            f"{ratio(r['first_valid'], r['first_tries'])} | {r['invalid']} | {r['tokens_in']} | {r['tokens_out']} |"
        )
    section.body = "\n".join(lines)
    return section


# ---------- running and rendering ----------


def run(
    conn: sqlite3.Connection,
    settings: Settings,
    get_embedder: Callable[[], Embedder],
    get_judge: Callable[[], LLMProvider],
    eval_dir: Path,
    *,
    repeats: int = 5,
    skip_summaries: bool = False,
    rejudge: bool = False,
) -> list[Section]:
    """Run every section. A section that cannot run is returned with not_run set."""

    def skipped() -> Section:
        raise NotRun("Skipped on request (--skip-summaries).")

    jobs: list[tuple[str, Callable[[], Section]]] = [
        ("Retrieval", lambda: eval_retrieval(conn, get_embedder, eval_dir / evalset.RETRIEVAL_FILE, repeats, settings)),
        ("Linking", lambda: eval_links(conn, settings, eval_dir / evalset.LINKS_FILE)),
        ("Summary faithfulness", skipped if skip_summaries else lambda: eval_summaries(conn, settings, get_judge, eval_dir, rejudge)),
        ("Ingest speed", lambda: eval_speed(conn)),
        ("LLM output validity and tokens", lambda: eval_llm_output(conn)),
    ]
    sections = []
    for title, job in jobs:
        logger.info("Evaluating: %s", title)
        try:
            sections.append(job())
        except NotRun as reason:
            logger.warning("%s not run: %s", title, reason)
            sections.append(Section(title, not_run=str(reason)))
        except SecondBrainError as exc:
            logger.error("%s failed: %s", title, exc)
            sections.append(Section(title, not_run=f"Failed: {exc}"))
    return sections


def environment(conn: sqlite3.Connection, settings: Settings, judge_model: str) -> dict[str, str]:
    """Facts about this run: when, what code, what machine, what data, what settings."""
    count = lambda sql: str(conn.execute(sql).fetchone()[0])  # noqa: E731
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    return {
        "Date": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M UTC%z"),
        "Git commit": _command(["git", "rev-parse", "--short", "HEAD"]) + (" (with uncommitted changes)" if _command(["git", "status", "--porcelain", "--", ".", ":!docs/EVAL.md"]) else ""),
        "Machine": f"{platform.system()} {platform.release()}, {platform.machine()}, {_cpu_name()}",
        "GPU": _command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]) or "not detected",
        "Python": platform.python_version(),
        "Packages": ", ".join(f"{name} {_version(name)}" for name in ("fastembed", "sqlite-vec", "anthropic")),
        "Items": f"{count('SELECT count(*) FROM items')} saved, {count('SELECT count(DISTINCT item_id) FROM chunks')} embedded, "
        f"{count('SELECT count(*) FROM analyses')} analyzed, {count('SELECT count(*) FROM chunks')} chunks",
        "Embedding model": f"{meta.get('embedding_model', settings.embedding_model)} (dim {meta.get('embedding_dim', '?')})",
        "Chunking": f"{settings.chunk_words} words, {settings.chunk_overlap_words} overlap",
        "Search": f"{settings.search_mode} (RRF k {settings.rrf_k}, depth {settings.search_depth})",
        "Links": f"text weight {settings.link_embed_weight}, threshold {settings.link_threshold}",
        "App LLM": f"{settings.llm_provider} / {settings.llm_model_name}, prompt {prompts.PROMPT_VERSION}, "
        f"input budget {settings.llm_max_input_tokens} tokens",
        "Judge LLM": judge_model,
    }


def render(sections: list[Section], env: dict[str, str]) -> str:
    """The Markdown report."""
    lines = [
        "# Evaluation results",
        "",
        "Generated by `scripts/eval.py`. Do not edit by hand: every number below was measured in "
        "the run described here. Parts that could not be measured say **Not run** and why.",
        "",
        "## Run",
        "",
        "| | |",
        "|---|---|",
        *[f"| {key} | {_cell(value)} |" for key, value in env.items()],
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---|",
    ]
    for section in sections:
        if section.not_run:
            lines.append(f"| {section.title} | **Not run** |")
        lines += [f"| {name} | {value} |" for name, value in section.headline.items()]
    lines += [
        "",
        "The label sets are small (about 25 queries, 30 pairs, 15 summaries), so these numbers "
        "are rough: one query or pair moves a result by several points.",
    ]
    for section in sections:
        lines += ["", f"## {section.title}", "", f"**Not run.** {section.not_run}" if section.not_run else section.body]
    return "\n".join(lines) + "\n"


def _cell(text: object) -> str:
    """Make text safe inside a Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _command(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "?"
