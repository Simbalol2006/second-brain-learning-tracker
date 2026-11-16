import sqlite3

import pytest

from secondbrain import vectors
from secondbrain.errors import StorageError


def test_knn_returns_nearest_first(conn: sqlite3.Connection) -> None:
    vectors.ensure_vector_tables(conn, "m", 4)
    vectors.set_chunk_vectors(conn, [(1, [1, 0, 0, 0]), (2, [0, 1, 0, 0]), (3, [0.9, 0.1, 0, 0])])
    result = vectors.knn_chunks(conn, [1, 0, 0, 0], k=2)
    assert [chunk_id for chunk_id, _ in result] == [1, 3]
    assert result[0][1] == pytest.approx(0.0, abs=1e-6)
    assert result[0][1] < result[1][1]


def test_setting_a_vector_twice_replaces_it(conn: sqlite3.Connection) -> None:
    vectors.ensure_vector_tables(conn, "m", 4)
    vectors.set_chunk_vectors(conn, [(1, [1, 0, 0, 0])])
    vectors.set_chunk_vectors(conn, [(1, [0, 1, 0, 0])])
    assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == 1
    assert vectors.knn_chunks(conn, [0, 1, 0, 0], k=1)[0][1] == pytest.approx(0.0, abs=1e-6)


def test_item_vector_presence(conn: sqlite3.Connection) -> None:
    assert not vectors.has_item_vector(conn, 1)  # tables not created yet
    vectors.ensure_vector_tables(conn, "m", 4)
    assert not vectors.has_item_vector(conn, 1)
    vectors.set_item_vector(conn, 1, [0, 0, 1, 0])
    assert vectors.has_item_vector(conn, 1)


def test_ensure_is_idempotent_for_same_model(conn: sqlite3.Connection) -> None:
    vectors.ensure_vector_tables(conn, "m", 4)
    vectors.ensure_vector_tables(conn, "m", 4)


@pytest.mark.parametrize(("model", "dim"), [("other", 4), ("m", 8)])
def test_mixing_models_is_refused(conn: sqlite3.Connection, model: str, dim: int) -> None:
    vectors.ensure_vector_tables(conn, "m", 4)
    with pytest.raises(StorageError, match="cannot be mixed"):
        vectors.ensure_vector_tables(conn, model, dim)


def test_wrong_dimension_vector_fails_loudly(conn: sqlite3.Connection) -> None:
    vectors.ensure_vector_tables(conn, "m", 4)
    with pytest.raises(sqlite3.OperationalError):
        vectors.set_chunk_vectors(conn, [(1, [1, 0])])
