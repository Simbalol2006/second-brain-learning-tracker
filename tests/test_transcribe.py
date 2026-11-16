"""Whisper wrapper tests with a fake faster_whisper module (no model, no GPU)."""

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from secondbrain.config import Settings
from secondbrain.errors import TranscribeError
from secondbrain.ingest.transcribe import transcribe_audio


class FakeSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start, self.end, self.text = start, end, text


class FakeInfo:
    language = "en"


def install_fake_whisper(monkeypatch: pytest.MonkeyPatch, segments: list[FakeSegment], fail: bool = False) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class WhisperModel:
        def __init__(self, name: str, **kwargs: Any) -> None:
            seen.update(name=name, **kwargs)
            if fail:
                raise RuntimeError("Library libcublas.so.12 is not found")

        def transcribe(self, path: str, **kwargs: Any) -> tuple[Any, FakeInfo]:
            return iter(segments), FakeInfo()

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return seen


def test_transcribe_maps_segments_and_passes_settings(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    seen = install_fake_whisper(monkeypatch, [FakeSegment(0, 2, " hi "), FakeSegment(2, 4, "there")])
    transcript = transcribe_audio(tmp_path / "a.webm", settings)
    assert transcript.source == "whisper"
    assert transcript.language == "en"
    assert [(s.text, s.start, s.end) for s in transcript.segments] == [("hi", 0.0, 2.0), ("there", 2.0, 4.0)]
    assert seen["name"] == "small"
    assert seen["device"] == "cpu"
    assert seen["compute_type"] == "int8"


def test_missing_package_tells_how_to_install(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # makes the import fail
    with pytest.raises(TranscribeError, match="uv sync --extra whisper"):
        transcribe_audio(tmp_path / "a.webm", settings)


def test_gpu_library_error_has_hint(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    install_fake_whisper(monkeypatch, [], fail=True)
    with pytest.raises(TranscribeError, match="libcublas.*WHISPER_DEVICE=cpu"):
        transcribe_audio(tmp_path / "a.webm", settings)


def test_silent_audio_is_an_error(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    install_fake_whisper(monkeypatch, [])
    with pytest.raises(TranscribeError, match="no speech"):
        transcribe_audio(tmp_path / "a.webm", settings)
