from __future__ import annotations

import json
import re
from fastapi import APIRouter, HTTPException

from app.database import get_db
from app.models import SearchResult, SightingResponse, EventResponse

router = APIRouter(prefix="/api/search", tags=["search"])


def extract_item_name(query: str) -> str:
    query = query.strip().lower().rstrip("?!.")
    patterns = [
        r"where (?:are|is|were) (?:my |the )?(.+)",
        r"(?:find|locate|show) (?:me )?(?:my |the )?(.+)",
        r"when did i last (?:see|have) (?:my |the )?(.+)",
        r"(?:last (?:seen|spotted)) (?:my |the )?(.+)",
        r"(?:my |the )?(.+)",
    ]
    for p in patterns:
        m = re.match(p, query)
        if m:
            return m.group(1).strip()
    return query


@router.get("/", response_model=list[SearchResult])
async def search(q: str):
    item_name = extract_item_name(q)
    if not item_name:
        raise HTTPException(400, "Could not understand the query")

    db = await get_db()
    try:
        # Latest sighting
        cursor = await db.execute(
            """SELECT s.item_name, s.image_path, s.similarity, s.timestamp,
                      s.zone, s.nearby_objects
               FROM sightings s
               JOIN registered_items r ON s.item_id = r.id
               WHERE s.item_name LIKE ?
               ORDER BY s.timestamp DESC LIMIT 1""",
            (f"%{item_name}%",),
        )
        row = await cursor.fetchone()
        if not row:
            return []

        # Total sighting count
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM sightings WHERE item_name LIKE ?",
            (f"%{item_name}%",),
        )
        count_row = await cursor.fetchone()

        # First seen
        cursor = await db.execute(
            "SELECT MIN(timestamp) as first FROM sightings WHERE item_name LIKE ?",
            (f"%{item_name}%",),
        )
        first_row = await cursor.fetchone()
    finally:
        await db.close()

    nearby = json.loads(row["nearby_objects"]) if row["nearby_objects"] else []

    return [SearchResult(
        item_name=row["item_name"],
        last_seen=row["timestamp"],
        image_url=f"/data/images/{row['image_path']}",
        similarity=row["similarity"],
        zone=row["zone"] or "center",
        nearby_objects=nearby,
        sighting_count=count_row["cnt"],
        first_seen=first_row["first"] or row["timestamp"],
    )]


@router.get("/recent", response_model=list[SightingResponse])
async def recent_sightings(limit: int = 30):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM sightings ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        SightingResponse(
            id=row["id"], item_id=row["item_id"], item_name=row["item_name"],
            image_path=row["image_path"], similarity=row["similarity"],
            zone=row["zone"] or "center",
            bbox_x=row["bbox_x"] or 0.5, bbox_y=row["bbox_y"] or 0.5,
            nearby_objects=json.loads(row["nearby_objects"]) if row["nearby_objects"] else [],
            source=row["source"] or "camera",
            timestamp=row["timestamp"],
        )
        for row in rows
    ]


@router.get("/history/{item_name}", response_model=list[SightingResponse])
async def item_history(item_name: str, limit: int = 20):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM sightings WHERE item_name LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{item_name}%", limit),
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        SightingResponse(
            id=row["id"], item_id=row["item_id"], item_name=row["item_name"],
            image_path=row["image_path"], similarity=row["similarity"],
            zone=row["zone"] or "center",
            bbox_x=row["bbox_x"] or 0.5, bbox_y=row["bbox_y"] or 0.5,
            nearby_objects=json.loads(row["nearby_objects"]) if row["nearby_objects"] else [],
            source=row["source"] or "camera",
            timestamp=row["timestamp"],
        )
        for row in rows
    ]


@router.get("/events/{item_name}", response_model=list[EventResponse])
async def item_events(item_name: str, limit: int = 50):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM events WHERE item_name LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{item_name}%", limit),
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        EventResponse(
            id=row["id"], item_name=row["item_name"],
            event_type=row["event_type"], zone=row["zone"],
            details=json.loads(row["details"]) if row["details"] else {},
            timestamp=row["timestamp"],
        )
        for row in rows
    ]
