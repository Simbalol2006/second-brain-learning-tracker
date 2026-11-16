"""Article access: download a page and keep its main text."""

import json
import logging

import trafilatura

from secondbrain.errors import FetchError
from secondbrain.models import Metadata, Segment, Transcript

logger = logging.getLogger(__name__)


def fetch(url: str) -> tuple[Metadata, Transcript]:
    """Download the page and extract metadata + paragraphs."""
    html = trafilatura.fetch_url(url)
    if html is None:
        raise FetchError(f"Could not download {url} (network error, or the site refused).")
    return extract(html, url)


def extract(html: str, url: str) -> tuple[Metadata, Transcript]:
    """Pull main text out of HTML. One Segment per paragraph, no timestamps."""
    raw = trafilatura.extract(
        html,
        url=url,
        output_format="json",
        with_metadata=True,
        include_comments=False,
        include_tables=False,
    )
    data = json.loads(raw) if raw else {}
    title = data.get("title")
    paragraphs = [p.strip() for p in (data.get("text") or "").split("\n")]
    segments = [Segment(text=p) for p in paragraphs if p and p != title]
    if not segments:
        raise FetchError(f"No article text found at {url}. The page may need JavaScript or a login.")
    metadata = Metadata(
        title=title,
        author=data.get("author"),
        language=data.get("language"),
        published_at=data.get("date"),
    )
    return metadata, Transcript(segments=segments, source="article", language=data.get("language"))
