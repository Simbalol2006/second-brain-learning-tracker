"""Operations on the whole library that touch several parts of the app at once."""

import logging
import sqlite3

from secondbrain import items, profile, vectors

logger = logging.getLogger(__name__)


def delete_item(conn: sqlite3.Connection, item_id: int) -> str:
    """Remove an item and everything derived from it. Returns its title.

    The items row takes its transcript, chunks, analysis, tags, links and timings
    with it (foreign keys). Vectors live in separate tables and are removed here.
    Skill levels derived from the item are recomputed.
    """
    item = items.get(conn, item_id)
    chunk_ids = [int(row["id"]) for row in items.load_chunks(conn, item_id)]
    vectors.delete_item(conn, item_id, chunk_ids)
    conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
    profile.refresh_derived(conn)
    conn.commit()
    logger.info("Deleted item %d", item_id)
    return str(item["title"] or item["url"])
