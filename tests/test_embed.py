from pathlib import Path

import numpy as np
import pytest

from secondbrain.embed import FastEmbedEmbedder, mean_vector, normalize
from secondbrain.errors import EmbeddingError


def test_normalize_makes_unit_rows_and_keeps_zero_rows() -> None:
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert out[0] == pytest.approx([0.6, 0.8])
    assert out[1] == pytest.approx([0.0, 0.0])


def test_mean_vector_is_unit_length() -> None:
    out = mean_vector(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert out[0] == pytest.approx(out[1])


def test_unknown_model_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError, match="Could not load embedding model 'not/a-model'"):
        FastEmbedEmbedder("not/a-model", tmp_path / "models")
