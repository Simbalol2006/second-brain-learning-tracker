"""CLI tests: commands run against a temp DATA_DIR with fake network and embedder."""

from pathlib import Path

import json

import pytest
import yaml
from conftest import FakeEmbedder, FakeProvider, make_segments
from typer.testing import CliRunner

from secondbrain import cli
from secondbrain.errors import FetchError, TranscriptUnavailable
from secondbrain.ingest import pipeline
from secondbrain.models import Metadata, Transcript

runner = CliRunner()
URL = "https://youtu.be/dQw4w9WgXcQ"


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "FastEmbedEmbedder", lambda model, cache_dir: FakeEmbedder())
    monkeypatch.setattr(
        pipeline.youtube, "fetch_metadata", lambda video_id: Metadata(title="A talk", duration_sec=630)
    )
    monkeypatch.setattr(
        pipeline.youtube,
        "fetch_captions",
        lambda video_id, languages: Transcript(make_segments(60), "auto", "en"),
    )


def test_list_when_empty() -> None:
    result = runner.invoke(cli.app, ["list"])
    assert result.exit_code == 0
    assert "No items yet" in result.output


def test_ingest_then_list_and_show() -> None:
    result = runner.invoke(cli.app, ["ingest", URL])
    assert result.exit_code == 0, result.output
    assert "item 1: embedded" in result.output

    listing = runner.invoke(cli.app, ["list"])
    assert "A talk" in listing.output
    assert "embedded" in listing.output
    assert "  10  " in listing.output  # 630 s shown as 10 (rounded) minutes

    shown = runner.invoke(cli.app, ["show", "1"])
    assert shown.exit_code == 0
    assert "transcript_source  auto" in shown.output
    assert "timing             embed" in shown.output


def test_ingest_again_says_already_saved() -> None:
    runner.invoke(cli.app, ["ingest", URL])
    result = runner.invoke(cli.app, ["ingest", URL])
    assert result.exit_code == 0
    assert "already saved" in result.output


def test_ingest_from_file_reports_failures_and_exits_nonzero(tmp_path: Path) -> None:
    urls = tmp_path / "urls.txt"
    urls.write_text(f"# my list\n{URL}\n\nnot-a-url\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["ingest", "--file", str(urls)])
    assert result.exit_code == 1
    assert "item 1: embedded" in result.output
    assert "not-a-url -> FAILED" in result.output


def test_ingest_failure_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(video_id: str, languages: list[str]) -> Transcript:
        raise FetchError("YouTube blocked the caption request")

    monkeypatch.setattr(pipeline.youtube, "fetch_captions", blocked)
    result = runner.invoke(cli.app, ["ingest", URL])
    assert result.exit_code == 1
    assert "FAILED" in result.output and "blocked" in result.output
    assert "failed" in runner.invoke(cli.app, ["list"]).output


def test_no_transcript_points_to_transcribe_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def none(video_id: str, languages: list[str]) -> Transcript:
        raise TranscriptUnavailable("Captions are disabled for this video.")

    monkeypatch.setattr(pipeline.youtube, "fetch_captions", none)
    result = runner.invoke(cli.app, ["ingest", URL])
    assert result.exit_code == 0
    assert "no_transcript" in result.output
    assert "sb transcribe 1" in result.output
    assert "no_transcript" in runner.invoke(cli.app, ["list", "--status", "no_transcript"]).output


def test_ingest_without_url_or_file() -> None:
    result = runner.invoke(cli.app, ["ingest"])
    assert result.exit_code == 1
    assert "Give a URL or --file" in result.output


def test_show_unknown_item() -> None:
    result = runner.invoke(cli.app, ["show", "99"])
    assert result.exit_code == 1
    assert "No item with id 99" in result.output


def test_bad_config_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_WORDS", "0")
    result = runner.invoke(cli.app, ["list"])
    assert result.exit_code == 1
    assert "Invalid configuration" in result.output


ANSWER = (
    '{"summary": "It explains things.", "key_points": ["One.", "Two."], '
    '"topics": ["vector search"], "skills": [], "difficulty": "beginner"}'
)


def use_fake_llm(monkeypatch: pytest.MonkeyPatch, answers: list[str | Exception]) -> FakeProvider:
    provider = FakeProvider(answers)
    monkeypatch.setattr(cli, "build_provider", lambda settings: provider)
    return provider


def test_analyze_one_item_and_show_it(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_llm(monkeypatch, [ANSWER])
    runner.invoke(cli.app, ["ingest", URL])
    result = runner.invoke(cli.app, ["analyze", "1"])
    assert result.exit_code == 0, result.output
    assert "item 1: analyzed - beginner, topics: vector search" in result.output
    assert "tokens in/out: 100/20" in result.output

    shown = runner.invoke(cli.app, ["show", "1"]).output
    assert "summary            It explains things." in shown
    assert "key point          Two." in shown
    assert "analyzed" in runner.invoke(cli.app, ["list"]).output


def test_ingest_with_analyze_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = use_fake_llm(monkeypatch, [ANSWER])
    result = runner.invoke(cli.app, ["ingest", URL, "--analyze"])
    assert result.exit_code == 0, result.output
    assert "item 1: embedded" in result.output and "item 1: analyzed" in result.output
    assert len(provider.calls) == 1


def test_analyze_all_only_touches_pending_items(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = use_fake_llm(monkeypatch, [ANSWER, ANSWER])
    runner.invoke(cli.app, ["ingest", URL])
    runner.invoke(cli.app, ["ingest", "https://youtu.be/aaaaaaaaaaa"])
    first = runner.invoke(cli.app, ["analyze", "--all"])
    assert first.exit_code == 0, first.output
    assert len(provider.calls) == 2
    second = runner.invoke(cli.app, ["analyze", "--all"])
    assert "Nothing to analyze" in second.output


def test_analyze_failure_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_llm(monkeypatch, ["nope", "nope"])
    runner.invoke(cli.app, ["ingest", URL])
    result = runner.invoke(cli.app, ["analyze", "1"])
    assert result.exit_code == 1
    assert "FAILED" in result.output and "invalid output 2 times" in result.output
    assert "failed" in runner.invoke(cli.app, ["list"]).output


def test_analyze_without_api_key_explains_what_to_set() -> None:
    runner.invoke(cli.app, ["ingest", URL])
    result = runner.invoke(cli.app, ["analyze", "1"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY is not set" in result.output


def test_analyze_needs_an_id_or_all() -> None:
    result = runner.invoke(cli.app, ["analyze"])
    assert result.exit_code == 1
    assert "Give an item id, or --all" in result.output


def ingest_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two videos with different words, so search and links have something to tell apart."""
    texts = {"dQw4w9WgXcQ": "gradient descent trains neural networks", "aaaaaaaaaaa": "sourdough bread needs slow fermentation"}

    def captions(video_id: str, languages: list[str]) -> Transcript:
        segments = make_segments(40)
        return Transcript([type(s)(texts[video_id], s.start, s.end) for s in segments], "auto", "en")

    monkeypatch.setattr(pipeline.youtube, "fetch_captions", captions)
    monkeypatch.setattr(pipeline.youtube, "fetch_metadata", lambda video_id: Metadata(title=f"Video {video_id[:3]}"))
    assert runner.invoke(cli.app, ["ingest", URL]).exit_code == 0
    assert runner.invoke(cli.app, ["ingest", "https://youtu.be/aaaaaaaaaaa"]).exit_code == 0


def test_search_command(monkeypatch: pytest.MonkeyPatch) -> None:
    ingest_two(monkeypatch)
    result = runner.invoke(cli.app, ["search", "sourdough bread", "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert "1. [" in result.output and "Video aaa (item 2)" in result.output
    assert "https://www.youtube.com/watch?v=aaaaaaaaaaa&t=" in result.output
    assert "2. [" not in result.output


def test_search_on_empty_library_and_empty_query() -> None:
    assert "No results" in runner.invoke(cli.app, ["search", "anything"]).output
    result = runner.invoke(cli.app, ["search", "  "])
    assert result.exit_code == 1 and "query is empty" in result.output


def test_related_links_and_graph_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ingest_two(monkeypatch)
    assert "no links above the threshold" in runner.invoke(cli.app, ["related", "1"]).output

    monkeypatch.setenv("LINK_THRESHOLD", "0")
    rebuilt = runner.invoke(cli.app, ["links", "--rebuild", "--stats"])
    assert rebuilt.exit_code == 0, rebuilt.output
    assert "1 links stored (threshold 0.0)" in rebuilt.output
    assert "pairs                    1" in rebuilt.output and "p50" in rebuilt.output

    related = runner.invoke(cli.app, ["related", "1"]).output
    assert "Video aaa (item 2)" in related and "text similarity" in related

    out = tmp_path / "graph.json"
    assert runner.invoke(cli.app, ["graph", "--out", str(out)]).exit_code == 0
    graph = json.loads(out.read_text())
    assert len(graph["nodes"]) == 2 and len(graph["edges"]) == 1
    assert json.loads(runner.invoke(cli.app, ["graph"]).output)["edges"] == graph["edges"]


def test_links_command_needs_an_option() -> None:
    result = runner.invoke(cli.app, ["links"])
    assert result.exit_code == 1 and "--rebuild and/or --stats" in result.output


def test_related_unknown_item() -> None:
    result = runner.invoke(cli.app, ["related", "42"])
    assert result.exit_code == 1 and "No item with id 42" in result.output


def test_eval_templates_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ingest_two(monkeypatch)
    out = tmp_path / "eval"
    result = runner.invoke(cli.app, ["eval-templates", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert len(yaml.safe_load((out / "links.yaml").read_text())["pairs"]) == 1
    again = runner.invoke(cli.app, ["eval-templates", "--out", str(out)])
    assert again.exit_code == 1 and "Not overwriting" in again.output


def test_status_profile_and_rule_based_recommend(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_llm(monkeypatch, [ANSWER, ANSWER])
    ingest_two(monkeypatch)
    assert runner.invoke(cli.app, ["analyze", "--all"]).exit_code == 0

    assert "goal 1 added" in runner.invoke(cli.app, ["profile", "add-goal", "bake great bread"]).output
    assert "skill 'python' set to intermediate" in runner.invoke(cli.app, ["profile", "set-skill", "Python", "2"]).output
    assert "item 1: done" in runner.invoke(cli.app, ["status", "1", "done"]).output

    shown = runner.invoke(cli.app, ["profile", "show"]).output
    assert "  1  bake great bread" in shown
    assert "python" in shown and "intermediate" in shown and "(manual, 0 finished)" in shown
    assert "vector search" in shown and "(derived, 1 finished)" in shown
    assert "Progress: 1 to_watch, 0 watching, 1 done" in shown
    assert "done" in runner.invoke(cli.app, ["list"]).output

    result = runner.invoke(cli.app, ["recommend", "--no-llm"])
    assert result.exit_code == 0, result.output
    assert "1. Video aaa (item 2) - beginner" in result.output
    assert 'Closest to your goal "bake great bread"' in result.output
    assert "item 1)" not in result.output  # finished items are not suggested


def test_recommend_with_llm_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = '{"recommendations": [{"item_id": 2, "reason": "Start here."}, {"item_id": 1, "reason": "Then this."}]}'
    use_fake_llm(monkeypatch, [ANSWER, ANSWER, reply])
    ingest_two(monkeypatch)
    runner.invoke(cli.app, ["analyze", "--all"])
    result = runner.invoke(cli.app, ["recommend"])
    assert result.exit_code == 0, result.output
    assert "1. Video aaa (item 2)" in result.output and "   Start here." in result.output
    assert "2. Video dQw (item 1)" in result.output and "   Then this." in result.output
    assert "Tip: no goals set" in result.output


def test_recommend_warns_when_the_llm_fails_or_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_llm(monkeypatch, [ANSWER, ANSWER, "garbage", "garbage"])
    ingest_two(monkeypatch)
    runner.invoke(cli.app, ["analyze", "--all"])
    failed = runner.invoke(cli.app, ["recommend"])
    assert failed.exit_code == 0
    assert "Warning: The LLM step failed" in failed.output and "rule-based" in failed.output
    assert "1. " in failed.output

    monkeypatch.undo()  # back to the real build_provider: no API key is configured
    missing = runner.invoke(cli.app, ["recommend"])
    assert "Warning: no LLM available (ANTHROPIC_API_KEY is not set" in missing.output


def test_recommend_with_nothing_analyzed() -> None:
    result = runner.invoke(cli.app, ["recommend", "--no-llm"])
    assert result.exit_code == 0 and "Nothing to recommend" in result.output


def test_status_and_profile_errors() -> None:
    assert "No item with id 7" in runner.invoke(cli.app, ["status", "7", "done"]).output
    runner.invoke(cli.app, ["ingest", URL])
    bad = runner.invoke(cli.app, ["status", "1", "finished"])
    assert bad.exit_code == 1 and "Unknown status 'finished'" in bad.output
    assert runner.invoke(cli.app, ["profile", "set-skill", "python", "9"]).exit_code == 1
    assert runner.invoke(cli.app, ["profile", "remove-goal", "3"]).exit_code == 1
    assert runner.invoke(cli.app, ["profile", "remove-skill", "nothing"]).exit_code == 1


def test_delete_command(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_llm(monkeypatch, [ANSWER, ANSWER])
    ingest_two(monkeypatch)
    runner.invoke(cli.app, ["analyze", "--all"])
    runner.invoke(cli.app, ["status", "1", "done"])

    declined = runner.invoke(cli.app, ["delete", "1"], input="n\n")
    assert declined.exit_code == 1 and "Video dQw" in runner.invoke(cli.app, ["list"]).output

    deleted = runner.invoke(cli.app, ["delete", "1", "--yes"])
    assert deleted.exit_code == 0 and "item 1 deleted" in deleted.output
    assert "Video dQw" not in runner.invoke(cli.app, ["list"]).output
    assert "(derived" not in runner.invoke(cli.app, ["profile", "show"]).output  # its skills are gone too
    assert "No item with id 1" in runner.invoke(cli.app, ["delete", "1", "--yes"]).output
