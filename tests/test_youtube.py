"""YouTube module tests. Network libraries are replaced by fakes."""

import io
import json
import urllib.error
from typing import Any

import pytest
import yt_dlp
from youtube_transcript_api import (
    FetchedTranscriptSnippet,
    IpBlocked,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from secondbrain.errors import FetchError, TranscriptUnavailable
from secondbrain.ingest import youtube
from secondbrain.models import Segment

VID = "dQw4w9WgXcQ"
SNIPPETS = [
    FetchedTranscriptSnippet(text="hello there", start=0.0, duration=2.5),
    FetchedTranscriptSnippet(text="general topic", start=2.5, duration=3.0),
]


class FakeTranscript:
    def __init__(self, language_code: str) -> None:
        self.language_code = language_code

    def fetch(self) -> list[FetchedTranscriptSnippet]:
        return SNIPPETS


class FakeTranscriptList:
    def __init__(self, manual: list[str], generated: list[str]) -> None:
        self.manual, self.generated = manual, generated

    def _find(self, have: list[str], wanted: list[str]) -> FakeTranscript:
        for code in wanted:
            if code in have:
                return FakeTranscript(code)
        raise NoTranscriptFound(VID, wanted, self)

    def find_manually_created_transcript(self, wanted: list[str]) -> FakeTranscript:
        return self._find(self.manual, wanted)

    def find_generated_transcript(self, wanted: list[str]) -> FakeTranscript:
        return self._find(self.generated, wanted)


def fake_api(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    """Make YouTubeTranscriptApi().list() return `result`, or raise it if it is an exception."""

    class Api:
        def list(self, video_id: str) -> Any:
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", Api)


def test_manual_captions_are_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_api(monkeypatch, FakeTranscriptList(manual=["en"], generated=["en"]))
    transcript = youtube.fetch_captions(VID, ["en"])
    assert transcript.source == "manual"
    assert transcript.language == "en"
    assert transcript.segments == [
        Segment("hello there", 0.0, 2.5),
        Segment("general topic", 2.5, 5.5),
    ]


def test_auto_captions_are_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_api(monkeypatch, FakeTranscriptList(manual=[], generated=["en"]))
    assert youtube.fetch_captions(VID, ["en"]).source == "auto"


def test_language_order_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_api(monkeypatch, FakeTranscriptList(manual=["hi", "en"], generated=[]))
    assert youtube.fetch_captions(VID, ["hi", "en"]).language == "hi"


def test_no_captions_in_wanted_language(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_api(monkeypatch, FakeTranscriptList(manual=["de"], generated=["de"]))
    with pytest.raises(TranscriptUnavailable, match="No captions in languages"):
        youtube.fetch_captions(VID, ["en"])


def test_disabled_captions(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_api(monkeypatch, TranscriptsDisabled(VID))
    with pytest.raises(TranscriptUnavailable, match="disabled"):
        youtube.fetch_captions(VID, ["en"])


def test_blocked_ip_is_a_clear_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_api(monkeypatch, IpBlocked(VID))
    with pytest.raises(FetchError, match="blocked the caption request"):
        youtube.fetch_captions(VID, ["en"])


def test_unavailable_video_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_api(monkeypatch, VideoUnavailable(VID))
    with pytest.raises(FetchError, match="VideoUnavailable"):
        youtube.fetch_captions(VID, ["en"])


def test_metadata_from_ytdlp(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {"title": "T", "channel": "C", "duration": 754.0, "upload_date": "20260115", "language": "en"}
    monkeypatch.setattr(youtube, "_ytdlp_extract", lambda *a, **k: info)
    meta = youtube.fetch_metadata(VID)
    assert (meta.title, meta.author, meta.duration_sec) == ("T", "C", 754)
    assert meta.published_at == "2026-01-15"
    assert meta.language == "en"


def _raise_download_error(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise yt_dlp.utils.DownloadError("no js runtime")


def test_metadata_falls_back_to_oembed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(youtube, "_ytdlp_extract", _raise_download_error)
    body = json.dumps({"title": "T2", "author_name": "C2"}).encode()
    monkeypatch.setattr(youtube.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(body))
    meta = youtube.fetch_metadata(VID)
    assert (meta.title, meta.author, meta.duration_sec) == ("T2", "C2", None)


def test_metadata_failure_names_both_causes(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(url: str, timeout: int) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(youtube, "_ytdlp_extract", _raise_download_error)
    monkeypatch.setattr(youtube.urllib.request, "urlopen", no_network)
    with pytest.raises(FetchError, match="no js runtime.*offline.*Deno"):
        youtube.fetch_metadata(VID)


def test_audio_download_failure_has_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setattr(youtube, "_ytdlp_extract", _raise_download_error)
    with pytest.raises(FetchError, match="Audio download failed.*Deno"):
        youtube.download_audio(VID, tmp_path)


def test_audio_download_returns_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    target = tmp_path / f"{VID}.webm"
    target.write_bytes(b"x")
    monkeypatch.setattr(youtube, "_ytdlp_extract", lambda *a, **k: {"_filename": str(target)})
    assert youtube.download_audio(VID, tmp_path) == target


def test_duration_from_segments() -> None:
    assert youtube.duration_from_segments([Segment("a", 0, 2.5), Segment("b", 2.5, 61.2)]) == 62
    assert youtube.duration_from_segments([Segment("a")]) is None
