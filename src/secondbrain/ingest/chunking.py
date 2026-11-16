"""Group transcript segments into overlapping chunks. Pure functions, no I/O."""

from collections.abc import Iterator

from secondbrain.models import Chunk, Segment


def chunk_segments(
    segments: list[Segment], target_words: int = 220, overlap_words: int = 40
) -> list[Chunk]:
    """Pack segments into chunks of about target_words words.

    Cuts only between segments, so timestamps stay exact. The last segments of a
    chunk (up to overlap_words words) are repeated at the start of the next one.
    A chunk never exceeds target_words + overlap_words words.
    """
    pieces = [piece for seg in segments for piece in _split_long(_clean(seg), target_words)]
    chunks: list[Chunk] = []
    current: list[Segment] = []
    count = 0
    for piece in pieces:
        words = _word_count(piece)
        if current and count + words > target_words:
            chunks.append(_make_chunk(len(chunks), current))
            current = _tail(current, overlap_words)
            count = sum(_word_count(s) for s in current)
        current.append(piece)
        count += words
    if current:
        chunks.append(_make_chunk(len(chunks), current))
    return chunks


def _clean(segment: Segment) -> Segment:
    return Segment(" ".join(segment.text.split()), segment.start, segment.end)


def _word_count(segment: Segment) -> int:
    return len(segment.text.split())


def _split_long(segment: Segment, max_words: int) -> Iterator[Segment]:
    """Yield the segment as-is, or in max_words-sized parts if it is too long.

    Parts of one long segment share its start and end time.
    """
    words = segment.text.split()
    for i in range(0, len(words), max_words):
        yield Segment(" ".join(words[i : i + max_words]), segment.start, segment.end)


def _tail(segments: list[Segment], overlap_words: int) -> list[Segment]:
    """Trailing segments whose total length fits in overlap_words."""
    tail: list[Segment] = []
    count = 0
    for segment in reversed(segments):
        count += _word_count(segment)
        if count > overlap_words:
            break
        tail.insert(0, segment)
    return tail


def _make_chunk(idx: int, segments: list[Segment]) -> Chunk:
    return Chunk(
        idx=idx,
        text=" ".join(s.text for s in segments),
        start=segments[0].start,
        end=segments[-1].end,
    )
