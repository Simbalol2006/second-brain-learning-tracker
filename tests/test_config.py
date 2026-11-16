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
    monkeypatch.setenv("SOME_FUTURE_SETTING", "x")  # unknown keys are ignored
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


def test_llm_defaults(settings: Settings) -> None:
    assert settings.llm_provider == "anthropic"
    assert settings.llm_model_name == "claude-haiku-5-5"
    assert settings.anthropic_api_key is None


def test_llm_model_override() -> None:
    assert Settings(llm_model="some-other-model", _env_file=None).llm_model_name == "some-other-model"


def test_ollama_needs_a_model_name() -> None:
    with pytest.raises(ValidationError, match="LLM_MODEL must name an Ollama model"):
        Settings(llm_provider="ollama", _env_file=None)
    assert Settings(llm_provider="ollama", llm_model="m:3b", _env_file=None).llm_model_name == "m:3b"


def test_api_key_is_not_shown_in_repr() -> None:
    loaded = Settings(anthropic_api_key="sk-secret-value", _env_file=None)
    assert "sk-secret-value" not in repr(loaded)
