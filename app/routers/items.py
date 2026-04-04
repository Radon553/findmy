from __future__ import annotations

import cv2
import uuid
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
import io

from app.config import REGISTERED_DIR
from app.database import get_db, serialize_embedding
from app.models import ItemResponse
from app.services.clip_service import clip_service

router = APIRouter(prefix="/api/items", tags=["items"])


@router.post("/register", response_model=ItemResponse)
async def register_item(name: str = Form(...), photo: UploadFile = File(...)):
    """Register a new item by uploading a photo and giving it a name."""
    name = name.strip().lower()
    if not name:
        raise HTTPException(400, "Item name is required")

    # Read and save the uploaded photo
    contents = await photo.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    filename = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = REGISTERED_DIR / filename
    image.save(str(filepath))

    # Generate CLIP embedding
    embedding = clip_service.get_image_embedding(image)

    # Save to database
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO registered_items (name, image_path, embedding)
               VALUES (?, ?, ?)""",
            (name, filename, serialize_embedding(embedding)),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM registered_items WHERE name = ?", (name,)
        )
        row = await cursor.fetchone()
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(409, f"Item '{name}' is already registered")
        raise
    finally:
        await db.close()

    return ItemResponse(
        id=row["id"],
        name=row["name"],
        image_path=row["image_path"],
        created_at=row["created_at"],
    )


@router.get("/", response_model=list[ItemResponse])
async def list_items():
    """List all registered items."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM registered_items ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        ItemResponse(
            id=row["id"],
            name=row["name"],
            image_path=row["image_path"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


@router.delete("/{item_id}")
async def delete_item(item_id: int):
    """Delete a registered item and its sightings."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM registered_items WHERE id = ?", (item_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Item not found")

        await db.execute("DELETE FROM sightings WHERE item_id = ?", (item_id,))
        await db.execute("DELETE FROM registered_items WHERE id = ?", (item_id,))
        await db.commit()
    finally:
        await db.close()

    # Remove image file
    filepath = REGISTERED_DIR / row["image_path"]
    filepath.unlink(missing_ok=True)

    return {"status": "deleted", "name": row["name"]}
