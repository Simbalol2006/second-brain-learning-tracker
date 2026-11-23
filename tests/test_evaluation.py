"""Evaluation tests. They check the measuring code on small made-up libraries;
they do not produce the project's real results (those come from scripts/eval.py)."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest
import yaml
from conftest import FakeEmbedder, FakeProvider, add_text_item

from secondbrain import analyze, evaluation, items
from secondbrain.config import Settings
from secondbrain.errors import LLMError
from secondbrain.evaluation import NotRun

ML = "gradient descent trains neural network weights by reducing loss"
BREAD = "sourdough bread needs flour water salt and slow fermentation"
SQL = "relational database tables are combined with joins in sql queries"
URL = "https://www.youtube.com/watch?v={}"
RUBRIC = Path(__file__).parent.parent / "eval" / "RUBRIC.md"


def analysis_json(topic: str) -> str:
    return json.dumps(
        {"summary": "First sentence. Second sentence!", "key_points": ["Point one.", "Point two."],
         "topics": [topic], "skills": [], "difficulty": "beginner"}
    )


def verdicts(*values: str) -> str:
    return json.dumps({"verdicts": [{"claim": n, "verdict": v, "evidence": "e"} for n, v in enumerate(values, 1)]})


@pytest.fixture
def library(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder) -> dict[str, int]:
    ids = {
        "ml": add_text_item(conn, settings, embedder, "a" * 11, "Neural nets", ML),
        "bread": add_text_item(conn, settings, embedder, "b" * 11, "Baking | bread", BREAD),
        "sql": add_text_item(conn, settings, embedder, "c" * 11, "SQL joins", SQL),
    }
    for key, topic in (("ml", "machine learning"), ("bread", "baking"), ("sql", "sql")):
        analyze.analyze_item(conn, settings, FakeProvider([analysis_json(topic)]), ids[key])
    return ids


@pytest.fixture
def eval_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "eval"
    folder.mkdir()
    (folder / "RUBRIC.md").write_text(RUBRIC.read_text(encoding="utf-8"), encoding="utf-8")
    return folder


def write(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------- metric functions ----------


def test_recall_at_k() -> None:
    assert evaluation.recall_at_k([1, 2, 3, 4, 5, 6], {2, 6}, 5) == 0.5
    assert evaluation.recall_at_k([1, 2, 3], {1, 2}, 5) == 1.0
    assert evaluation.recall_at_k([], {1}, 5) == 0.0


def test_reciprocal_rank() -> None:
    assert evaluation.reciprocal_rank([7, 8, 9], {9, 8}) == 0.5
    assert evaluation.reciprocal_rank([9], {9}) == 1.0
    assert evaluation.reciprocal_rank([1, 2], {9}) == 0.0


def test_confusion_and_ratio() -> None:
    counts = evaluation.confusion([True, True, False, False, True], [True, False, True, False, True])
    assert counts == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    assert evaluation.ratio(2, 3) == "0.667 (2/3)"
    assert evaluation.ratio(0, 0) == "n/a (0/0)"


def test_split_claims() -> None:
    claims = evaluation.split_claims("One thing. Another thing!  A third?", ["Point A.", " ", "Point B."])
    assert claims == ["One thing.", "Another thing!", "A third?", "Point A.", "Point B."]


def test_combine_verdicts() -> None:
    parts = [["not_supported", "contradicted", "not_supported"], ["supported", "not_supported", "not_supported"]]
    assert evaluation.combine_verdicts(parts) == ["supported", "contradicted", "not_supported"]
    assert evaluation.combine_verdicts([["contradicted"], ["supported"]]) == ["supported"]


# ---------- retrieval ----------


def test_retrieval_section(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int], eval_dir: Path) -> None:
    queries = [
        {"query": "sourdough bread fermentation", "relevant": [URL.format("b" * 11)]},
        {"query": "sql joins on database tables", "relevant": ["https://youtu.be/" + "c" * 11, URL.format("a" * 11)]},
        {"query": "", "relevant": []},  # unfinished entries are ignored
    ]
    section = evaluation.eval_retrieval(conn, lambda: embedder, write(eval_dir / "retrieval.yaml", {"queries": queries}), repeats=3)
    assert section.headline["Retrieval recall@5"] == "1.000 (mean over 2 queries, hybrid)"  # only 3 items exist
    assert section.headline["Retrieval MRR"] == "1.000 (mean over 2 queries, hybrid)"
    assert "(6 timed searches)" in section.headline["Search latency p50 / p95"]
    assert "| sourdough bread fermentation | 1 | 1.00 | 1 |" in section.body
    assert "| sql joins on database tables | 2 | 1.00 | 1 |" in section.body


def test_retrieval_counts_misses(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int], eval_dir: Path) -> None:
    queries = [{"query": "sourdough bread fermentation", "relevant": [URL.format("a" * 11)]}]  # labelled "wrong" on purpose
    section = evaluation.eval_retrieval(conn, lambda: embedder, write(eval_dir / "retrieval.yaml", {"queries": queries}), repeats=1)
    assert "1.000" not in section.headline["Retrieval MRR"]  # the relevant item is not ranked first
    assert section.headline["Retrieval MRR"].startswith(("0.500", "0.333"))


def test_retrieval_compares_all_three_modes(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, library: dict[str, int], eval_dir: Path) -> None:
    queries = [{"query": "sourdough bread fermentation", "relevant": [URL.format("b" * 11)]}]
    configured = settings.model_copy(update={"search_mode": "bm25"})
    section = evaluation.eval_retrieval(conn, lambda: embedder, write(eval_dir / "retrieval.yaml", {"queries": queries}), repeats=1, settings=configured)
    assert section.headline["Retrieval recall@5"] == "1.000 (mean over 1 queries, bm25)"
    assert "- Search mode: **bm25**" in section.body
    for row in ("| dense | 1.000 | 1.000 | 0 |", "| bm25 (configured) | 1.000 | 1.000 | 0 |", "| hybrid | 1.000 | 1.000 | 0 |"):
        assert row in section.body
    assert "| sourdough bread fermentation | 1 | 1.00 | 1 | 1 | 1 | 1 |" in section.body


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "does not exist"),
        ({"queries": []}, "no queries with relevant items yet"),
        ({"queries": [{"query": "q", "relevant": ["https://youtu.be/zzzzzzzzzzz"]}]}, "not saved items"),
        ("queries: [unclosed", "not valid YAML"),
    ],
)
def test_retrieval_not_run(conn: sqlite3.Connection, embedder: FakeEmbedder, library: dict[str, int], eval_dir: Path, content: object, reason: str) -> None:
    path = eval_dir / "retrieval.yaml"
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    elif content is not None:
        write(path, content)
    with pytest.raises(NotRun, match=reason):
        evaluation.eval_retrieval(conn, lambda: embedder, path, repeats=1)


# ---------- linking ----------


def test_links_section(conn: sqlite3.Connection, library: dict[str, int], eval_dir: Path) -> None:
    pairs = [
        {"a": URL.format("a" * 11), "b": URL.format("b" * 11), "related": True},
        {"a": URL.format("a" * 11), "b": URL.format("c" * 11), "related": False},
        {"a": URL.format("b" * 11), "b": URL.format("c" * 11), "related": None},  # not labelled yet
    ]
    path = write(eval_dir / "links.yaml", {"pairs": pairs})
    everything = evaluation.eval_links(conn, Settings(link_threshold=0.0, _env_file=None), path)
    assert everything.headline == {"Linking precision": "0.500 (1/2)", "Linking recall": "1.000 (1/1)"}
    assert "Labelled pairs: **2** of 3 (1 related, 1 unrelated)" in everything.body
    assert "TP 1, FP 1, FN 0, TN 0" in everything.body
    assert "tuning on the eval set" in everything.body and "| 0.8 |" in everything.body

    nothing = evaluation.eval_links(conn, Settings(link_threshold=1.0, _env_file=None), path)
    assert nothing.headline == {"Linking precision": "n/a (0/0)", "Linking recall": "0.000 (0/1)"}


def test_links_not_run_without_labels(conn: sqlite3.Connection, settings: Settings, library: dict[str, int], eval_dir: Path) -> None:
    path = write(eval_dir / "links.yaml", {"pairs": [{"a": URL.format("a" * 11), "b": URL.format("b" * 11), "related": None}]})
    with pytest.raises(NotRun, match=r"no pairs with `related: true/false` yet \(1 pairs unlabelled\)"):
        evaluation.eval_links(conn, settings, path)


# ---------- summaries ----------


def summaries_file(eval_dir: Path, spot_check: bool = False) -> None:
    write(eval_dir / "summaries.yaml", {"items": [
        {"url": URL.format("a" * 11), "spot_check": spot_check},
        {"url": URL.format("b" * 11), "spot_check": False},
    ]})


def test_summaries_section_and_cache(conn: sqlite3.Connection, settings: Settings, library: dict[str, int], eval_dir: Path) -> None:
    summaries_file(eval_dir)
    judge = FakeProvider([
        verdicts("supported", "supported", "not_supported", "supported"),
        verdicts("supported", "contradicted", "supported", "supported"),
    ])
    section = evaluation.eval_summaries(conn, settings, lambda: judge, eval_dir)
    assert section.headline == {"Summary faithfulness (LLM judge)": "0.750 (6/8)"}
    assert "Items judged: **2** (0 taken from the judge cache, 2 judged in this run)" in section.body
    assert "Not supported: 1 · Contradicted: 1" in section.body
    assert "| Neural nets | 4 | 3 | 1 | 0 | 0.75 |" in section.body
    assert "| Baking \\| bread | 4 | 3 | 0 | 1 | 0.75 |" in section.body  # the | in the title is escaped
    assert "no item in `summaries.yaml` has `spot_check: true`" in section.body

    prompt = judge.calls[0]
    assert prompt["system"].startswith("You are checking whether a summary")
    assert "SOURCE:\n" + ML in prompt["user"]
    assert "CLAIMS:\n1. First sentence.\n2. Second sentence!\n3. Point one.\n4. Point two." in prompt["user"]

    again = evaluation.eval_summaries(conn, settings, lambda: FakeProvider([]), eval_dir)  # no calls allowed
    assert again.headline == section.headline
    assert "(2 taken from the judge cache, 0 judged in this run)" in again.body

    fresh = FakeProvider([verdicts(*["supported"] * 4), verdicts(*["supported"] * 4)])
    rejudged = evaluation.eval_summaries(conn, settings, lambda: fresh, eval_dir, rejudge=True)
    assert rejudged.headline == {"Summary faithfulness (LLM judge)": "1.000 (8/8)"}


def test_judge_must_answer_every_claim(conn: sqlite3.Connection, settings: Settings, library: dict[str, int], eval_dir: Path) -> None:
    summaries_file(eval_dir)
    judge = FakeProvider([verdicts("supported", "supported"), verdicts(*["supported"] * 4), verdicts(*["supported"] * 4)])
    evaluation.eval_summaries(conn, settings, lambda: judge, eval_dir)
    assert "need exactly one verdict for each claim number 1..4" in judge.calls[1]["user"]  # the retry prompt


def test_long_sources_are_judged_in_parts(conn: sqlite3.Connection, embedder: FakeEmbedder, eval_dir: Path) -> None:
    small = Settings(llm_max_input_tokens=1500, _env_file=None)
    item_id = add_text_item(conn, small, embedder, "d" * 11, "Long talk", ML, repeat=120)  # about 7600 chars
    analyze.analyze_item(conn, Settings(_env_file=None), FakeProvider([analysis_json("ml")]), item_id)
    judge = FakeProvider([
        verdicts("not_supported", "contradicted", "supported", "not_supported"),
        verdicts("supported", "not_supported", "not_supported", "not_supported"),
    ])
    result = evaluation.judge_claims(conn, small, judge, "RUBRIC", item_id, ["a", "b", "c", "d"])
    assert len(judge.calls) == 2  # the source did not fit into one prompt
    assert "(part 2 of 2 of a longer text)" in judge.calls[1]["user"]
    assert result == ["supported", "contradicted", "supported", "not_supported"]


def test_spot_check_file_and_agreement(conn: sqlite3.Connection, settings: Settings, library: dict[str, int], eval_dir: Path) -> None:
    summaries_file(eval_dir, spot_check=True)
    judge = FakeProvider([verdicts("supported", "supported", "not_supported", "contradicted"), verdicts(*["supported"] * 4)])
    first = evaluation.eval_summaries(conn, settings, lambda: judge, eval_dir)
    spot_path = eval_dir / "spot_check.yaml"
    assert "was created in this run" in first.body and "Judge vs human agreement" not in first.headline
    template = yaml.safe_load(spot_path.read_text())
    assert [c["text"] for c in template["items"][0]["claims"]] == ["First sentence.", "Second sentence!", "Point one.", "Point two."]
    assert all(c["human"] is None for c in template["items"][0]["claims"])
    assert "not_supported" not in spot_path.read_text().split("\n\n", 1)[1]  # the judge's verdicts are not revealed

    unlabelled = evaluation.eval_summaries(conn, settings, lambda: judge, eval_dir)
    assert "no usable human labels yet" in unlabelled.body

    for claim, label in zip(template["items"][0]["claims"], ["supported", "not_supported", "not_supported", "not_supported"]):
        claim["human"] = label
    write(spot_path, template)
    done = evaluation.eval_summaries(conn, settings, lambda: judge, eval_dir)
    assert done.headline["Judge vs human agreement"] == "0.500 (2/4)"  # claims 1 and 3 match exactly
    assert "4 claims from 1 items labelled by hand" in done.body
    assert "supported / not supported: **0.750 (3/4)**" in done.body  # claim 4: contradicted vs not_supported


def test_summaries_not_run_cases(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, library: dict[str, int], eval_dir: Path) -> None:
    def no_llm() -> FakeProvider:
        raise LLMError("ANTHROPIC_API_KEY is not set")

    summaries_file(eval_dir)
    with pytest.raises(NotRun, match="No LLM judge available: ANTHROPIC_API_KEY is not set"):
        evaluation.eval_summaries(conn, settings, no_llm, eval_dir)
    with pytest.raises(NotRun, match="The judge failed on .*invalid output 2 times"):
        evaluation.eval_summaries(conn, settings, lambda: FakeProvider(["junk", "junk"]), eval_dir)

    add_text_item(conn, settings, embedder, "e" * 11, "Not analyzed", "other words entirely")
    write(eval_dir / "summaries.yaml", {"items": [{"url": URL.format("e" * 11)}]})
    with pytest.raises(NotRun, match="has no analysis"):
        evaluation.eval_summaries(conn, settings, lambda: FakeProvider([]), eval_dir)

    (eval_dir / "RUBRIC.md").unlink()
    with pytest.raises(NotRun, match="RUBRIC.md` is missing"):
        evaluation.eval_summaries(conn, settings, lambda: FakeProvider([]), eval_dir)


# ---------- speed and LLM log ----------


def test_speed_section_uses_the_latest_run_of_each_stage(conn: sqlite3.Connection, library: dict[str, int]) -> None:
    conn.execute("DELETE FROM ingest_timings")
    for item_id, embed_seconds in ((library["ml"], 2.0), (library["bread"], 4.0)):
        items.add_timing(conn, item_id, "embed", 99.0)  # an earlier, failed attempt
        items.add_timing(conn, item_id, "embed", embed_seconds)
        items.add_timing(conn, item_id, "chunk", 1.0)
    section = evaluation.eval_speed(conn)
    assert section.headline == {"Ingest time per item (median / p95)": "4.0 s / 4.9 s (2 items)"}
    assert "| embed | 2 | 3.00 | 3.90 | 4.00 |" in section.body
    assert "| chunk | 2 | 1.00 | 1.00 | 1.00 |" in section.body
    assert "10 min median" in section.body


def test_llm_output_section(conn: sqlite3.Connection, library: dict[str, int]) -> None:
    items.add_llm_call(conn, library["ml"], "final", "fake", "fake-model", 1, False, 50, 5, 0.1)
    items.add_llm_call(conn, library["ml"], "final", "fake", "fake-model", 2, True, 60, 5, 0.1)
    section = evaluation.eval_llm_output(conn)
    assert section.headline == {"JSON valid on first try (app calls)": "0.750 (3/4)"}  # 3 library analyses + 1 failed first try
    assert "| fake | fake-model | final | 5 | 0.750 (3/4) | 1 | 410 | 70 |" in section.body
    assert "median **100 in / 20 out**" in section.body


def test_sections_are_not_run_on_an_empty_database(conn: sqlite3.Connection) -> None:
    with pytest.raises(NotRun, match="No ingest timings"):
        evaluation.eval_speed(conn)
    with pytest.raises(NotRun, match="No LLM calls"):
        evaluation.eval_llm_output(conn)


# ---------- whole run and report ----------


def test_empty_run_reports_not_run_everywhere_and_no_numbers(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, eval_dir: Path) -> None:
    sections = evaluation.run(conn, settings, lambda: embedder, lambda: FakeProvider([]), eval_dir)
    assert [s.title for s in sections] == ["Retrieval", "Linking", "Summary faithfulness", "Ingest speed", "LLM output validity and tokens"]
    assert all(s.not_run and not s.headline for s in sections)
    report = evaluation.render(sections, {"Date": "today"})
    assert report.count("**Not run.**") == 5
    assert "| Retrieval | **Not run** |" in report
    summary = report.split("## Summary")[1].split("## Retrieval")[0]
    assert not any(ch.isdigit() for ch in summary.split("The label sets")[0])  # no metric values at all


def test_full_run_and_report(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, library: dict[str, int], eval_dir: Path) -> None:
    write(eval_dir / "retrieval.yaml", {"queries": [{"query": "sourdough bread", "relevant": [URL.format("b" * 11)]}]})
    write(eval_dir / "links.yaml", {"pairs": [{"a": URL.format("a" * 11), "b": URL.format("b" * 11), "related": False}]})
    summaries_file(eval_dir)
    judge = FakeProvider([verdicts(*["supported"] * 4)] * 2)
    sections = evaluation.run(conn, settings, lambda: embedder, lambda: judge, eval_dir, repeats=2)
    assert not any(s.not_run for s in sections)
    env = evaluation.environment(conn, settings, "fake / fake-model")
    assert env["Items"] == "3 saved, 3 embedded, 3 analyzed, " + env["Items"].split(", ")[-1]
    assert env["Embedding model"] == "fake-embedder (dim 64)" and env["Links"] == "text weight 0.7, threshold 0.5"
    assert env["Search"] == "hybrid (RRF k 60, depth 50)"
    report = evaluation.render(sections, env)
    for expected in ("| Retrieval recall@5 | 1.000", "| Linking precision | n/a (0/0)", "| Summary faithfulness (LLM judge) | 1.000 (8/8)",
                     "| Ingest time per item", "| JSON valid on first try (app calls) | 1.000 (3/3) |", "| Judge LLM | fake / fake-model |", "## Linking"):
        assert expected in report


def test_skip_summaries_makes_no_judge_calls(conn: sqlite3.Connection, settings: Settings, embedder: FakeEmbedder, library: dict[str, int], eval_dir: Path) -> None:
    summaries_file(eval_dir)
    sections = evaluation.run(conn, settings, lambda: embedder, lambda: FakeProvider([]), eval_dir, skip_summaries=True)
    assert sections[2].not_run == "Skipped on request (--skip-summaries)."


def test_script_writes_the_report_where_told(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_dir: Path) -> None:
    spec = importlib.util.spec_from_file_location("eval_script", Path(__file__).parent.parent / "scripts" / "eval.py")
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(script, "FastEmbedEmbedder", lambda model, cache_dir: FakeEmbedder())
    out = tmp_path / "out" / "EVAL.md"
    assert script.main(["--eval-dir", str(eval_dir), "--out", str(out), "--skip-summaries"]) == 0
    report = out.read_text()
    assert report.startswith("# Evaluation results") and report.count("**Not run.**") == 5
    assert "| Judge LLM | anthropic / claude-haiku-5-5 |" in report


def test_committed_eval_file_contains_no_results() -> None:
    """Until scripts/eval.py is run on real data, docs/EVAL.md must not show numbers."""
    committed = (Path(__file__).parent.parent / "docs" / "EVAL.md").read_text(encoding="utf-8")
    if "No evaluation has been run yet" in committed:
        assert "| Retrieval recall@5 |" not in committed
