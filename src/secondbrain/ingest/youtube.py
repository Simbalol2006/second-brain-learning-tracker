"""YouTube access: metadata, captions and audio download."""

import json
import logging
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import requests
import yt_dlp
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from secondbrain.errors import FetchError, TranscriptUnavailable
from secondbrain.models import Metadata, Segment, Transcript

logger = logging.getLogger(__name__)
_ytdlp_logger = logging.getLogger("secondbrain.ytdlp")

_OEMBED_URL = "https://www.youtube.com/oembed?format=json&url="
_JS_RUNTIME_HINT = (
    "yt-dlp needs an external JavaScript runtime (Deno) for YouTube; see README 'Setup'. "
    "Also try upgrading yt-dlp: uv lock --upgrade-package yt-dlp && uv sync."
)


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def fetch_metadata(video_id: str) -> Metadata:
    """Title, channel, duration and date. Falls back to oEmbed (title + channel only)."""
    try:
        info = _ytdlp_extract(video_id, download=False, options={})
    except yt_dlp.utils.DownloadError as exc:
        logger.warning("yt-dlp metadata failed for %s (%s); trying oEmbed", video_id, _short(exc))
        return _fetch_oembed(video_id, ytdlp_error=_short(exc))
    return Metadata(
        title=info.get("title"),
        author=info.get("channel") or info.get("uploader"),
        duration_sec=_to_int(info.get("duration")),
        language=info.get("language"),
        published_at=_format_date(info.get("upload_date")),
    )


def fetch_captions(video_id: str, languages: list[str]) -> Transcript:
    """Manual captions if present, else auto-generated ones.

    Raises TranscriptUnavailable when the video has no captions in the wanted
    languages, and FetchError for real failures (blocked IP, bad video, network).
    """
    try:
        available = YouTubeTranscriptApi().list(video_id)
        chosen, source = _pick_transcript(available, languages)
        snippets = chosen.fetch()
    except TranscriptsDisabled as exc:
        raise TranscriptUnavailable("Captions are disabled for this video.") from exc
    except NoTranscriptFound as exc:
        raise TranscriptUnavailable(
            f"No captions in languages {languages} (set CAPTION_LANGUAGES to change)."
        ) from exc
    except RequestBlocked as exc:
        raise FetchError(
            "YouTube blocked the caption request from this IP address. This is common on "
            "cloud/VPN IPs. Try again later or from another network."
        ) from exc
    except CouldNotRetrieveTranscript as exc:
        raise FetchError(f"Could not get captions for {video_id}: {type(exc).__name__}") from exc
    except requests.RequestException as exc:
        raise FetchError(f"Network error while getting captions: {exc}") from exc
    segments = [
        Segment(text=s.text, start=float(s.start), end=float(s.start) + float(s.duration))
        for s in snippets
    ]
    return Transcript(segments=segments, source=source, language=chosen.language_code)


def download_audio(video_id: str, out_dir: Path) -> Path:
    """Download the audio track for Whisper. Returns the file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"format": "bestaudio/best", "outtmpl": str(out_dir / "%(id)s.%(ext)s")}
    try:
        info = _ytdlp_extract(video_id, download=True, options=options)
    except yt_dlp.utils.DownloadError as exc:
        raise FetchError(f"Audio download failed: {_short(exc)}. {_JS_RUNTIME_HINT}") from exc
    path = Path(info["_filename"])
    if not path.exists():
        raise FetchError(f"Audio download reported success but {path} is missing.")
    return path


def duration_from_segments(segments: list[Segment]) -> int | None:
    """Approximate video length from the last caption's end time."""
    ends = [s.end for s in segments if s.end is not None]
    return math.ceil(max(ends)) if ends else None


def _pick_transcript(available: Any, languages: list[str]) -> tuple[Any, str]:
    try:
        return available.find_manually_created_transcript(languages), "manual"
    except NoTranscriptFound:
        return available.find_generated_transcript(languages), "auto"


def _ytdlp_extract(video_id: str, download: bool, options: dict[str, Any]) -> dict[str, Any]:
    base = {"quiet": True, "no_warnings": True, "noplaylist": True, "logger": _ytdlp_logger}
    with yt_dlp.YoutubeDL({**base, **options}) as ydl:
        info = ydl.extract_info(watch_url(video_id), download=download)
        if download:
            info["_filename"] = ydl.prepare_filename(info)
        return info


def _fetch_oembed(video_id: str, ytdlp_error: str) -> Metadata:
    url = _OEMBED_URL + urllib.parse.quote(watch_url(video_id), safe="")
    try:
        with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310 - fixed https host
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise FetchError(
            f"Could not get video metadata. yt-dlp said: {ytdlp_error}. "
            f"oEmbed fallback said: {exc}. {_JS_RUNTIME_HINT}"
        ) from exc
    return Metadata(title=data.get("title"), author=data.get("author_name"))


def _short(exc: Exception) -> str:
    """yt-dlp errors end with a long 'please report this issue' note; drop it."""
    return str(exc).split("; please report")[0]


def _format_date(upload_date: str | None) -> str | None:
    if not upload_date or len(upload_date) != 8:
        return None
    return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"


def _to_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None
