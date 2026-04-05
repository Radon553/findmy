from __future__ import annotations

import logging
import numpy as np
import uuid
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageEnhance, ImageOps
import io

from app.config import REGISTERED_DIR, TEXT_PROMPT_TEMPLATES
from app.database import get_db, serialize_embedding
from app.models import ItemResponse
from app.services.clip_service import clip_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/items", tags=["items"])


def _generate_augmented_crops(image: Image.Image) -> list[Image.Image]:
    """Generate augmented versions of a registration image.

    Returns 5 variants: original, center crop, horizontal flip,
    and brightness adjustments. Kept small for SigLIP performance.
    """
    w, h = image.size
    crops: list[Image.Image] = [image]

    # Center crop (60% of image)
    cw, ch = int(w * 0.6), int(h * 0.6)
    left, top = (w - cw) // 2, (h - ch) // 2
    crops.append(image.crop((left, top, left + cw, top + ch)))

    # Horizontal flip
    crops.append(ImageOps.mirror(image))

    # Brightness adjustments
    enhancer = ImageEnhance.Brightness(image)
    crops.append(enhancer.enhance(0.8))  # darker
    crops.append(enhancer.enhance(1.2))  # brighter

    return crops


def _embed_photo(image: Image.Image) -> list[float]:
    """Generate a single robust embedding from one photo using augmentation."""
    augmented = _generate_augmented_crops(image)
    logger.info(f"  Generating CLIP embeddings for {len(augmented)} augmented crops")
    embeddings = clip_service.get_image_embeddings_batch(augmented)
    avg = np.mean(embeddings, axis=0)
    avg = avg / np.linalg.norm(avg)
    return avg.tolist()


@router.post("/register", response_model=ItemResponse)
async def register_item(
    name: str = Form(...),
    photos: list[UploadFile] = File(...),
):
    """Register a new item by uploading one or more photos."""
    name = name.strip().lower()
    if not name:
        raise HTTPException(400, "Item name is required")

    if not photos or len(photos) == 0:
        raise HTTPException(400, "At least one photo is required")

    # Read first photo and save as display image
    first_contents = await photos[0].read()
    first_image = Image.open(io.BytesIO(first_contents)).convert("RGB")
    filename = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = REGISTERED_DIR / filename
    first_image.save(str(filepath))

    # Generate embedding for the first photo (stored in registered_items for compat)
    logger.info(f"Registering '{name}' with {len(photos)} photo(s)")
    first_embedding = _embed_photo(first_image)

    # Save item to database
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO registered_items (name, image_path, embedding)
               VALUES (?, ?, ?)""",
            (name, filename, serialize_embedding(first_embedding)),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM registered_items WHERE name = ?", (name,)
        )
        row = await cursor.fetchone()
        item_id = row["id"]

        # Store embedding for first photo in item_embeddings
        await db.execute(
            "INSERT INTO item_embeddings (item_id, embedding, source) VALUES (?, ?, ?)",
            (item_id, serialize_embedding(first_embedding), "photo_1"),
        )

        # Process remaining photos
        for i, photo in enumerate(photos[1:], start=2):
            contents = await photo.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            embedding = _embed_photo(image)
            await db.execute(
                "INSERT INTO item_embeddings (item_id, embedding, source) VALUES (?, ?, ?)",
                (item_id, serialize_embedding(embedding), f"photo_{i}"),
            )

        await db.commit()
        logger.info(f"Registered '{name}' with {len(photos)} embedding(s)")
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
        photo_count=len(photos),
    )


@router.post("/register-text", response_model=ItemResponse)
async def register_item_text(body: dict):
    """Register a new item using only its name — CLIP generates text embeddings."""
    name = body.get("name", "").strip().lower()
    if not name:
        raise HTTPException(400, "Item name is required")

    # Generate descriptive prompts and embed them
    prompts = [t.format(name) for t in TEXT_PROMPT_TEMPLATES]
    logger.info(f"Registering '{name}' via text with {len(prompts)} prompts")
    embeddings = clip_service.get_text_embeddings(prompts)

    first_embedding = embeddings[0]

    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO registered_items (name, image_path, embedding)
               VALUES (?, ?, ?)""",
            (name, "text_only", serialize_embedding(first_embedding)),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM registered_items WHERE name = ?", (name,)
        )
        row = await cursor.fetchone()
        item_id = row["id"]

        for i, emb in enumerate(embeddings):
            await db.execute(
                "INSERT INTO item_embeddings (item_id, embedding, source, embedding_type) VALUES (?, ?, ?, ?)",
                (item_id, serialize_embedding(emb), f"text_prompt_{i + 1}", "text"),
            )
        await db.commit()
        logger.info(f"Registered '{name}' with {len(embeddings)} text embeddings")
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(409, f"Item '{name}' is already registered")
        raise
    finally:
        await db.close()

    return ItemResponse(
        id=row["id"],
        name=row["name"],
        image_path=None,
        created_at=row["created_at"],
        photo_count=0,
    )


@router.post("/{item_id}/add-photo", response_model=ItemResponse)
async def add_photo(item_id: int, photo: UploadFile = File(...)):
    """Add another photo to an existing registered item."""
    contents = await photo.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    embedding = _embed_photo(image)

    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM registered_items WHERE id = ?", (item_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Item not found")

        # Count existing embeddings to set source label
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM item_embeddings WHERE item_id = ?",
            (item_id,),
        )
        count_row = await cursor.fetchone()
        next_num = count_row["cnt"] + 1

        await db.execute(
            "INSERT INTO item_embeddings (item_id, embedding, source) VALUES (?, ?, ?)",
            (item_id, serialize_embedding(embedding), f"photo_{next_num}"),
        )
        await db.commit()

        logger.info(f"Added photo {next_num} to '{row['name']}'")

        # Get updated count
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM item_embeddings WHERE item_id = ?",
            (item_id,),
        )
        count_row = await cursor.fetchone()
    finally:
        await db.close()

    return ItemResponse(
        id=row["id"],
        name=row["name"],
        image_path=row["image_path"],
        created_at=row["created_at"],
        photo_count=count_row["cnt"],
    )


@router.get("/", response_model=list[ItemResponse])
async def list_items():
    """List all registered items with their photo counts."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT ri.*, COALESCE(ec.cnt, 0) as photo_count
               FROM registered_items ri
               LEFT JOIN (
                   SELECT item_id, COUNT(*) as cnt FROM item_embeddings GROUP BY item_id
               ) ec ON ec.item_id = ri.id
               ORDER BY ri.created_at DESC"""
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
            photo_count=row["photo_count"],
        )
        for row in rows
    ]


@router.delete("/{item_id}")
async def delete_item(item_id: int):
    """Delete a registered item, its embeddings, and its sightings."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM registered_items WHERE id = ?", (item_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, "Item not found")

        await db.execute("DELETE FROM sightings WHERE item_id = ?", (item_id,))
        await db.execute("DELETE FROM item_embeddings WHERE item_id = ?", (item_id,))
        await db.execute("DELETE FROM registered_items WHERE id = ?", (item_id,))
        await db.commit()
    finally:
        await db.close()

    # Remove image file
    filepath = REGISTERED_DIR / row["image_path"]
    filepath.unlink(missing_ok=True)

    return {"status": "deleted", "name": row["name"]}
