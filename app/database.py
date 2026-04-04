from __future__ import annotations

import aiosqlite
import json
from app.config import DB_PATH


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS registered_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                image_path TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                item_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                similarity REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (item_id) REFERENCES registered_items(id)
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sightings_item
            ON sightings(item_id, timestamp DESC)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sightings_name
            ON sightings(item_name, timestamp DESC)
        """)
        await db.commit()


def serialize_embedding(embedding: list[float]) -> str:
    return json.dumps(embedding)


def deserialize_embedding(data: str) -> list[float]:
    return json.loads(data)
