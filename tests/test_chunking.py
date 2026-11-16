from conftest import make_segments

from secondbrain.ingest.chunking import chunk_segments
from secondbrain.models import Segment


def test_empty_input_gives_no_chunks() -> None:
    assert chunk_segments([]) == []
    assert chunk_segments([Segment("   ", 0, 1)]) == []


def test_short_input_is_one_chunk() -> None:
    chunks = chunk_segments(make_segments(5), target_words=220, overlap_words=40)
    assert len(chunks) == 1
    assert chunks[0].idx == 0
    assert chunks[0].start == 0.0
    assert chunks[0].end == 15.0
    assert len(chunks[0].text.split()) == 40


def test_chunks_respect_size_limit_and_cover_everything() -> None:
    segments = make_segments(200)  # 1600 words
    chunks = chunk_segments(segments, target_words=220, overlap_words=40)
    assert len(chunks) > 1
    assert [c.idx for c in chunks] == list(range(len(chunks)))
    assert all(len(c.text.split()) <= 220 + 40 for c in chunks)
    all_words = {w for s in segments for w in s.text.split()}
    assert {w for c in chunks for w in c.text.split()} == all_words


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_segments(make_segments(200), target_words=220, overlap_words=40)
    for prev, nxt in zip(chunks, chunks[1:]):
        shared = set(prev.text.split()) & set(nxt.text.split())
        assert 0 < len(shared) <= 40
        assert nxt.start < prev.end  # the next chunk starts inside the previous one


def test_zero_overlap() -> None:
    chunks = chunk_segments(make_segments(100), target_words=80, overlap_words=0)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert not set(prev.text.split()) & set(nxt.text.split())
        assert nxt.start == prev.end


def test_timestamps_come_from_first_and_last_segment() -> None:
    chunks = chunk_segments(make_segments(30, words_each=10), target_words=50, overlap_words=10)
    assert chunks[0].start == 0.0
    assert chunks[0].end == 15.0  # 5 segments of 10 words, 3 s each
    assert chunks[-1].end == 90.0


def test_long_segment_is_split() -> None:
    long = Segment(" ".join(f"p{i}" for i in range(500)))
    chunks = chunk_segments([long], target_words=220, overlap_words=40)
    assert len(chunks) == 3
    assert all(len(c.text.split()) <= 220 for c in chunks)
    assert chunks[0].start is None and chunks[0].end is None


def test_whitespace_is_collapsed() -> None:
    chunks = chunk_segments([Segment("hello\n  world", 0, 1), Segment("again", 1, 2)])
    assert chunks[0].text == "hello world again"
