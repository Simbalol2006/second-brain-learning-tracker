"""Plain data containers passed between ingest stages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """One piece of source text. Times are None for articles."""

    text: str
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class Chunk:
    """A group of segments sized for embedding."""

    idx: int
    text: str
    start: float | None
    end: float | None


@dataclass(frozen=True)
class Metadata:
    title: str | None = None
    author: str | None = None
    duration_sec: int | None = None
    language: str | None = None
    published_at: str | None = None


@dataclass(frozen=True)
class Transcript:
    """Segments plus where they came from: manual, auto, whisper or article."""

    segments: list[Segment]
    source: str
    language: str | None = None
