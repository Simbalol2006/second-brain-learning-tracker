from pathlib import Path

import pytest

from secondbrain.errors import FetchError
from secondbrain.ingest import article

FIXTURE = Path(__file__).parent / "fixtures" / "article.html"
URL = "https://example.com/vector-search"


def test_extract_gets_metadata_and_paragraphs() -> None:
    meta, transcript = article.extract(FIXTURE.read_text(encoding="utf-8"), URL)
    assert meta.title == "How Vector Search Works"
    assert meta.author == "Test Author"
    assert meta.published_at == "2026-03-04"
    assert transcript.source == "article"
    assert len(transcript.segments) == 4
    assert transcript.segments[0].text.startswith("Vector search finds documents")
    assert all(s.start is None and s.end is None for s in transcript.segments)
    words = sum(len(s.text.split()) for s in transcript.segments)
    assert meta.duration_sec == round(words / 200 * 60) > 0  # estimated reading time


def test_extract_drops_navigation_and_footer() -> None:
    _, transcript = article.extract(FIXTURE.read_text(encoding="utf-8"), URL)
    text = " ".join(s.text for s in transcript.segments)
    assert "newsletter" not in text
    assert "About" not in text


def test_page_without_text_fails_with_reason() -> None:
    with pytest.raises(FetchError, match="No article text found"):
        article.extract("<html><body></body></html>", URL)


def test_download_failure_fails_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(article.trafilatura, "fetch_url", lambda url: None)
    with pytest.raises(FetchError, match="Could not download"):
        article.fetch(URL)
