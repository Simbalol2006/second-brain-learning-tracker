from pathlib import Path

import pytest
from pydantic import ValidationError

from secondbrain.config import Settings


def test_defaults(settings: Settings) -> None:
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.chunk_words == 220
    assert settings.whisper_model == "small"
    assert settings.db_path == settings.data_dir / "secondbrain.db"
    assert settings.caption_language_list == ["en"]


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CHUNK_WORDS", "100")
    monkeypatch.setenv("CAPTION_LANGUAGES", "en, hi")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")  # unknown keys from later phases are ignored
    loaded = Settings(_env_file=None)
    assert loaded.chunk_words == 100
    assert loaded.caption_language_list == ["en", "hi"]


def test_reads_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("WHISPER_DEVICE=cpu\n", encoding="utf-8")
    assert Settings().whisper_device == "cpu"


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_WORDS"):
        Settings(chunk_words=50, chunk_overlap_words=50, _env_file=None)


def test_bad_whisper_device_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(whisper_device="tpu", _env_file=None)
