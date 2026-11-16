"""Exception types. Every failure the user can hit maps to one of these."""


class SecondBrainError(Exception):
    """Base class for all expected, user-facing errors."""


class StorageError(SecondBrainError):
    """Database or vector store problem."""


class InvalidUrlError(SecondBrainError):
    """The URL is malformed or of a kind the app does not support."""


class FetchError(SecondBrainError):
    """Could not download metadata, captions, audio or an article."""


class TranscriptUnavailable(SecondBrainError):
    """The source has no usable captions. Expected case, not a failure."""


class TranscribeError(SecondBrainError):
    """Local Whisper transcription failed or is not installed."""


class EmbeddingError(SecondBrainError):
    """The embedding model could not be loaded or run."""


class LLMError(SecondBrainError):
    """The LLM provider is misconfigured, unreachable, or returned an unusable response."""


class AnalysisError(SecondBrainError):
    """Analysis of an item failed. The reason is also stored on the item."""


class IngestError(SecondBrainError):
    """An ingest stage failed. The reason is also stored on the item."""

    def __init__(self, item_id: int, stage: str, message: str) -> None:
        super().__init__(f"item {item_id} failed at stage '{stage}': {message}")
        self.item_id = item_id
        self.stage = stage
        self.message = message
