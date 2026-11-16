"""Typed settings loaded from environment variables and the .env file."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration in one place. Field names map to upper-case env vars."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("data")
    log_level: str = "INFO"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    caption_languages: str = "en"
    chunk_words: int = Field(default=220, gt=0)
    chunk_overlap_words: int = Field(default=40, ge=0)

    whisper_model: str = "small"
    whisper_device: Literal["auto", "cpu", "cuda"] = "cpu"
    whisper_compute_type: str = "int8"

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap_words >= self.chunk_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_WORDS")
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "secondbrain.db"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def caption_language_list(self) -> list[str]:
        return [lang.strip() for lang in self.caption_languages.split(",") if lang.strip()]
