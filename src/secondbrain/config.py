"""Typed settings loaded from environment variables and the .env file."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"


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

    link_embed_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    link_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    search_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    rrf_k: int = Field(default=60, ge=1)
    search_depth: int = Field(default=50, ge=1)

    llm_provider: Literal["anthropic", "ollama"] = "anthropic"
    llm_model: str = ""
    llm_max_input_tokens: int = Field(default=12000, ge=1000)
    llm_max_output_tokens: int = Field(default=4096, ge=256)
    anthropic_api_key: SecretStr | None = None
    ollama_host: str = "http://localhost:11434"

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap_words >= self.chunk_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_WORDS")
        return self

    @model_validator(mode="after")
    def _check_llm_model(self) -> "Settings":
        if self.llm_provider == "ollama" and not self.llm_model.strip():
            raise ValueError(
                "LLM_MODEL must name an Ollama model you have pulled when LLM_PROVIDER=ollama "
                "(see README, 'Using a local model')"
            )
        return self

    @property
    def llm_model_name(self) -> str:
        """The model to call: LLM_MODEL, or the low-cost Anthropic default."""
        return self.llm_model.strip() or DEFAULT_ANTHROPIC_MODEL

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
