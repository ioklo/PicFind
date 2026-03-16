from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import ImageRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    path TEXT PRIMARY KEY,
    size_bytes INTEGER NOT NULL,
    modified_time REAL NOT NULL,
    caption TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_images_updated_at ON images(updated_at);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.executescript(SCHEMA)
    return connection


def upsert_images(connection: sqlite3.Connection, records: Iterable[ImageRecord]) -> int:
    rows = [
        (
            str(record.path),
            record.size_bytes,
            record.modified_time,
            record.caption,
            record.embedding,
            int(np.frombuffer(record.embedding, dtype=np.float32).shape[0]),
        )
        for record in records
    ]
    if not rows:
        return 0
    connection.executemany(
        """
        INSERT INTO images (path, size_bytes, modified_time, caption, embedding, embedding_dim)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            size_bytes = excluded.size_bytes,
            modified_time = excluded.modified_time,
            caption = excluded.caption,
            embedding = excluded.embedding,
            embedding_dim = excluded.embedding_dim,
            updated_at = CURRENT_TIMESTAMP
        """,
        rows,
    )
    connection.commit()
    return len(rows)


def update_captions(connection: sqlite3.Connection, rows: Iterable[tuple[Path, str]]) -> int:
    payload = [(caption, str(path)) for path, caption in rows]
    if not payload:
        return 0
    connection.executemany(
        """
        UPDATE images
        SET caption = ?, updated_at = CURRENT_TIMESTAMP
        WHERE path = ?
        """,
        payload,
    )
    connection.commit()
    return len(payload)


def clear_captions(connection: sqlite3.Connection) -> int:
    cursor = connection.execute(
        """
        UPDATE images
        SET caption = '', updated_at = CURRENT_TIMESTAMP
        WHERE caption != ''
        """
    )
    connection.commit()
    return int(cursor.rowcount)


def get_existing_file_state(connection: sqlite3.Connection, path: Path) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT size_bytes, modified_time FROM images WHERE path = ?",
        (str(path),),
    ).fetchone()


def iter_caption_targets(connection: sqlite3.Connection, skip_existing: bool) -> list[tuple[Path, str]]:
    query = "SELECT path, caption FROM images"
    params: tuple[object, ...] = ()
    if skip_existing:
        query += " WHERE caption = ''"
    query += " ORDER BY path"
    rows = connection.execute(query, params).fetchall()
    return [(Path(row["path"]), row["caption"]) for row in rows]


def load_search_matrix(connection: sqlite3.Connection) -> tuple[np.ndarray, list[tuple[Path, str]]]:
    rows = connection.execute(
        "SELECT path, caption, embedding FROM images ORDER BY path"
    ).fetchall()
    if not rows:
        return np.empty((0, 0), dtype=np.float32), []
    matrix = np.vstack(
        [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
    )
    metadata = [(Path(row["path"]), row["caption"]) for row in rows]
    return matrix, metadata


def count_images(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) AS count FROM images").fetchone()
    return int(row["count"])


def count_captioned_images(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT COUNT(*) AS count FROM images WHERE caption != ''"
    ).fetchone()
    return int(row["count"])
