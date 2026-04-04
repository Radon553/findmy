import re
from fastapi import APIRouter, HTTPException

from app.database import get_db
from app.models import SearchResult, SightingResponse

router = APIRouter(prefix="/api/search", tags=["search"])


def extract_item_name(query: str) -> str:
    """Extract the item name from a natural language query.

    Handles patterns like:
      'where are my keys?'
      'find my wallet'
      'when did I last see my AirPods?'
      'keys'
    """
    query = query.strip().lower().rstrip("?!.")
    # Strip common prefixes
    patterns = [
        r"where (?:are|is|were) (?:my |the )?(.+)",
        r"(?:find|locate|show) (?:me )?(?:my |the )?(.+)",
        r"when did i last (?:see|have) (?:my |the )?(.+)",
        r"(?:last (?:seen|spotted)) (?:my |the )?(.+)",
        r"(?:my |the )?(.+)",
    ]
    for pattern in patterns:
        match = re.match(pattern, query)
        if match:
            return match.group(1).strip()
    return query


@router.get("/", response_model=list[SearchResult])
async def search(q: str):
    """Search for the last known location of an item.

    Accepts natural language queries like 'where are my keys?'
    """
    item_name = extract_item_name(q)
    if not item_name:
        raise HTTPException(400, "Could not understand the query")

    db = await get_db()
    try:
        # Search by substring match against registered item names
        cursor = await db.execute(
            """SELECT s.item_name, s.image_path, s.similarity, s.timestamp
               FROM sightings s
               JOIN registered_items r ON s.item_id = r.id
               WHERE s.item_name LIKE ?
               ORDER BY s.timestamp DESC
               LIMIT 1""",
            (f"%{item_name}%",),
        )
        row = await cursor.fetchone()
    finally:
        await db.close()

    if not row:
        return []

    return [
        SearchResult(
            item_name=row["item_name"],
            last_seen=row["timestamp"],
            image_url=f"/data/images/{row['image_path']}",
            similarity=row["similarity"],
        )
    ]


@router.get("/history/{item_name}", response_model=list[SightingResponse])
async def item_history(item_name: str, limit: int = 20):
    """Get the sighting history for a specific item."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT * FROM sightings
               WHERE item_name LIKE ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (f"%{item_name}%", limit),
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        SightingResponse(
            id=row["id"],
            item_id=row["item_id"],
            item_name=row["item_name"],
            image_path=row["image_path"],
            similarity=row["similarity"],
            timestamp=row["timestamp"],
        )
        for row in rows
    ]
