"""Turn a pasted URL into (source_type, source_id, canonical_url)."""

import hashlib
import re
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

from secondbrain.errors import InvalidUrlError

_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}
_SHORT_HOST = "youtu.be"
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_ID_PATH_PREFIXES = ("shorts", "embed", "live", "v")
_TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "si")


def parse_url(url: str) -> tuple[str, str, str]:
    """Classify a URL. Raises InvalidUrlError with a reason the user can act on."""
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise InvalidUrlError(f"Not a valid http(s) URL: '{url}'")
    host = parts.netloc.lower().removeprefix("www.")
    if host in _YOUTUBE_HOSTS or host == _SHORT_HOST:
        video_id = _youtube_video_id(host, parts.path, parts.query)
        return "youtube", video_id, f"https://www.youtube.com/watch?v={video_id}"
    canonical = _canonical_article_url(url)
    source_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return "article", source_id, canonical


def _youtube_video_id(host: str, path: str, query: str) -> str:
    segments = [s for s in path.split("/") if s]
    candidate: str | None = None
    if host == _SHORT_HOST:
        candidate = segments[0] if segments else None
    elif segments[:1] == ["watch"]:
        candidate = parse_qs(query).get("v", [None])[0]
    elif len(segments) >= 2 and segments[0] in _ID_PATH_PREFIXES:
        candidate = segments[1]
    if candidate is None or not _VIDEO_ID.match(candidate):
        raise InvalidUrlError(
            "This YouTube URL does not point to a single video. "
            "Playlist and channel URLs are not supported; paste a video URL."
        )
    return candidate


def _canonical_article_url(url: str) -> str:
    """Drop the fragment and tracking parameters so the same page maps to one item."""
    parts = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKING_PREFIXES)
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))
