"""CLI tests: commands run against a temp DATA_DIR with fake network and embedder."""

from pathlib import Path

import pytest
from conftest import FakeEmbedder, make_segments
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
