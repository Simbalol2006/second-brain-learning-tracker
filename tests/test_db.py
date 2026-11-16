import sqlite3

import pytest

from secondbrain import db
from secondbrain.config import Settings
from secondbrain.errors import StorageError


def test_connect_creates_schema(conn: sqlite3.Connection) -> None:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"items", "transcripts", "chunks", "ingest_timings", "meta"} <= tables
    assert {"analyses", "topics", "item_topics", "llm_calls", "links"} <= tables
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)


def test_sqlite_vec_is_loaded(conn: sqlite3.Connection) -> None:
    assert conn.execute("SELECT vec_version()").fetchone()[0].startswith("v")


def test_reconnect_keeps_data(settings: Settings) -> None:
    first = db.connect(settings.db_path)
    first.execute("INSERT INTO items (source_type, source_id, url) VALUES ('youtube', 'a', 'u')")
    first.commit()
    first.close()
    second = db.connect(settings.db_path)
    assert second.execute("SELECT count(*) FROM items").fetchone()[0] == 1
    second.close()


def test_newer_schema_is_rejected(settings: Settings) -> None:
    first = db.connect(settings.db_path)
    first.execute(f"PRAGMA user_version = {len(db.MIGRATIONS) + 1}")
    first.commit()
    first.close()
    with pytest.raises(StorageError, match="newer than this code"):
        db.connect(settings.db_path)


def test_same_source_cannot_be_inserted_twice(conn: sqlite3.Connection) -> None:
    sql = "INSERT INTO items (source_type, source_id, url) VALUES ('youtube', 'a', 'u')"
    conn.execute(sql)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql)


def test_invalid_status_is_rejected(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO items (source_type, source_id, url, ingest_status) "
            "VALUES ('youtube', 'a', 'u', 'bogus')"
        )


def test_phase1_database_is_upgraded_in_place(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    all_migrations = db.MIGRATIONS
    monkeypatch.setattr(db, "MIGRATIONS", all_migrations[:1])
    old = db.connect(settings.db_path)
    old.execute("INSERT INTO items (source_type, source_id, url) VALUES ('youtube', 'a', 'u')")
    old.commit()
    old.close()

    monkeypatch.setattr(db, "MIGRATIONS", all_migrations)
    new = db.connect(settings.db_path)
    assert new.execute("PRAGMA user_version").fetchone()[0] == len(all_migrations)
    assert new.execute("SELECT count(*) FROM items").fetchone()[0] == 1
    assert new.execute("SELECT count(*) FROM analyses").fetchone()[0] == 0
    new.close()
