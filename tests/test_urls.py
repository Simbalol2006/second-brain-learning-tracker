import pytest

from secondbrain.errors import InvalidUrlError
from secondbrain.ingest.urls import parse_url

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=42s&list=PLabc",
        f"https://m.youtube.com/watch?feature=share&v={VID}",
        f"https://youtu.be/{VID}?si=abcdef",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube.com/live/{VID}?feature=share",
        f"  http://www.youtube.com/watch?v={VID}  ",
    ],
)
def test_youtube_variants_map_to_one_id(url: str) -> None:
    assert parse_url(url) == ("youtube", VID, f"https://www.youtube.com/watch?v={VID}")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=PLabc",
        "https://www.youtube.com/@somechannel",
        "https://www.youtube.com/watch?v=short",
        "https://youtu.be/",
    ],
)
def test_youtube_non_video_urls_rejected(url: str) -> None:
    with pytest.raises(InvalidUrlError, match="single video"):
        parse_url(url)


@pytest.mark.parametrize("url", ["", "not a url", "ftp://example.com/x", "youtube.com/watch?v=abc"])
def test_invalid_urls_rejected(url: str) -> None:
    with pytest.raises(InvalidUrlError, match="Not a valid"):
        parse_url(url)


def test_article_url_is_canonicalised() -> None:
    a = parse_url("https://Example.com/blog/post/?utm_source=x&id=7#section")
    b = parse_url("https://example.com/blog/post?id=7")
    assert a == b
    assert a[0] == "article"
    assert a[2] == "https://example.com/blog/post?id=7"
    assert len(a[1]) == 16


def test_different_articles_get_different_ids() -> None:
    assert parse_url("https://example.com/a")[1] != parse_url("https://example.com/b")[1]
