from __future__ import annotations

import aiosqlite
import json
import logging
from app.config import DB_PATH

logger = logging.getLogger(__name__)


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row

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
                item_id INTEGER,
                item_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                similarity REAL NOT NULL,
                zone TEXT DEFAULT 'center',
                bbox_x REAL DEFAULT 0.5,
                bbox_y REAL DEFAULT 0.5,
                nearby_objects TEXT DEFAULT '[]',
                source TEXT DEFAULT 'camera',
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

        # Multi-angle embeddings
        await db.execute("""
            CREATE TABLE IF NOT EXISTS item_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                embedding TEXT NOT NULL,
                source TEXT DEFAULT 'original',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (item_id) REFERENCES registered_items(id)
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_item_embeddings_item
            ON item_embeddings(item_id)
        """)

        # Movement events — tracks state changes (appeared / moved / disappeared)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                item_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                zone TEXT,
                details TEXT DEFAULT '{}',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (item_id) REFERENCES registered_items(id)
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_item
            ON events(item_id, timestamp DESC)
        """)

        # Add index on item_name for auto-detected sightings (no item_id)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_sightings_name_ts
            ON sightings(item_name COLLATE NOCASE, timestamp DESC)
        """)

        # Migrate legacy embeddings if needed
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM item_embeddings")
        row = await cursor.fetchone()
        if row["cnt"] == 0:
            cursor = await db.execute(
                "SELECT id, embedding FROM registered_items "
                "WHERE embedding IS NOT NULL AND embedding != ''"
            )
            items = await cursor.fetchall()
            if items:
                for item in items:
                    await db.execute(
                        "INSERT INTO item_embeddings (item_id, embedding, source) "
                        "VALUES (?, ?, ?)",
                        (item["id"], item["embedding"], "migrated"),
                    )
                logger.info(f"Migrated {len(items)} embeddings to item_embeddings")

        await db.commit()


def serialize_embedding(embedding: list[float]) -> str:
    return json.dumps(embedding)


def deserialize_embedding(data: str) -> list[float]:
    return json.loads(data)
