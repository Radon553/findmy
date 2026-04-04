"""
Comprehensive test suite for FindThis.

Tests:
  1. Basic detection pipeline (YOLO -> ByteTrack -> CLIP matching)
  2. Robustness — detection on a transformed video (brightness, crop, blur)
  3. Search endpoint — natural language query parsing + DB lookup
  4. Similarity threshold — verify we don't get false positives
  5. Cooldown — same object shouldn't spam sightings
  6. Item registration + deletion lifecycle
"""

from __future__ import annotations

import asyncio
import cv2
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from app.config import IMAGES_DIR, REGISTERED_DIR, DB_PATH, SIMILARITY_THRESHOLD
from app.database import init_db, get_db, serialize_embedding, deserialize_embedding
from app.services.camera import camera_service
from app.services.clip_service import clip_service
from app.services.detector import detection_service
from app.routers.search import extract_item_name

TEST_VIDEO_1 = Path(__file__).parent / "test_video.mp4"
TEST_VIDEO_2 = Path(__file__).parent / "test_video2.mp4"
TARGET_CLASSES = {"laptop", "cup", "book"}

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if ok:
        passed += 1
    else:
        failed += 1


# ─── Helpers ───

def extract_best_crops(video_path: str, targets: set[str]) -> dict[str, Image.Image]:
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    best: dict[str, tuple[float, Image.Image]] = {}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for f_idx in range(0, frame_count, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)[0]
        h, w = frame.shape[:2]
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls_name not in targets or (cls_name in best and best[cls_name][0] >= conf):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            best[cls_name] = (conf, Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    cap.release()
    return {k: v[1] for k, v in best.items()}


async def clean_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    for f in IMAGES_DIR.glob("*.jpg"):
        f.unlink()
    for f in REGISTERED_DIR.glob("*_ref.jpg"):
        f.unlink()
    await init_db()


async def register_crops(crops: dict[str, Image.Image]):
    clip_service.load()
    for name, image in crops.items():
        embedding = clip_service.get_image_embedding(image)
        filename = f"{name}_ref.jpg"
        image.save(str(REGISTERED_DIR / filename))
        db = await get_db()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO registered_items (name, image_path, embedding) VALUES (?, ?, ?)",
                (name, filename, serialize_embedding(embedding)),
            )
            await db.commit()
        finally:
            await db.close()


async def run_detection(video_path: str, duration: float):
    loop = asyncio.get_event_loop()
    camera_service.start(video_path=video_path, loop=True)
    await asyncio.sleep(0.5)
    detection_service.start(loop)
    await asyncio.sleep(duration)
    detection_service.stop()
    camera_service.stop()


async def count_sightings(item_name: str = None) -> int:
    db = await get_db()
    try:
        if item_name:
            cursor = await db.execute(
                "SELECT COUNT(*) as c FROM sightings WHERE item_name = ?", (item_name,)
            )
        else:
            cursor = await db.execute("SELECT COUNT(*) as c FROM sightings")
        row = await cursor.fetchone()
        return row["c"]
    finally:
        await db.close()


async def get_best_similarity(item_name: str) -> float:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT MAX(similarity) as best FROM sightings WHERE item_name = ?",
            (item_name,),
        )
        row = await cursor.fetchone()
        return row["best"] or 0.0
    finally:
        await db.close()


# ─── Tests ───

async def test_1_basic_detection():
    """Test that YOLO+CLIP pipeline detects registered items in the original video."""
    print("\n[Test 1] Basic detection pipeline")
    await clean_db()
    crops = extract_best_crops(str(TEST_VIDEO_1), TARGET_CLASSES)
    report("Extract reference crops", len(crops) == len(TARGET_CLASSES),
           f"got {list(crops.keys())}")

    await register_crops(crops)
    db = await get_db()
    cursor = await db.execute("SELECT COUNT(*) as c FROM registered_items")
    row = await cursor.fetchone()
    await db.close()
    report("Items registered in DB", row["c"] == len(TARGET_CLASSES), f"{row['c']} items")

    await run_detection(str(TEST_VIDEO_1), duration=15.0)

    total = await count_sightings()
    report("Sightings recorded", total > 0, f"{total} total sightings")

    for name in TARGET_CLASSES:
        count = await count_sightings(name)
        best = await get_best_similarity(name)
        report(f"Detected '{name}'", count > 0,
               f"{count} sightings, best similarity={best:.3f}")


async def test_2_robustness():
    """Test detection on a transformed video (brightness changes, crop, blur)."""
    print("\n[Test 2] Robustness — transformed video")

    if not TEST_VIDEO_2.exists():
        report("test_video2.mp4 exists", False, "generate it first")
        return

    await clean_db()
    # Register from original video, detect in transformed video
    crops = extract_best_crops(str(TEST_VIDEO_1), TARGET_CLASSES)
    await register_crops(crops)

    await run_detection(str(TEST_VIDEO_2), duration=15.0)

    total = await count_sightings()
    report("Sightings on transformed video", total > 0, f"{total} total")

    detected = []
    for name in TARGET_CLASSES:
        count = await count_sightings(name)
        best = await get_best_similarity(name)
        if count > 0:
            detected.append(name)
        report(f"Detected '{name}' (transformed)", count > 0,
               f"{count} sightings, best={best:.3f}")

    report("At least 2/3 items detected on transformed video",
           len(detected) >= 2, f"{len(detected)}/3")


async def test_3_search():
    """Test natural language search query parsing."""
    print("\n[Test 3] Search — natural language parsing")

    cases = [
        ("where are my keys?", "keys"),
        ("find my wallet", "wallet"),
        ("Where is the laptop?", "laptop"),
        ("when did I last see my AirPods?", "airpods"),
        ("cup", "cup"),
        ("show me the book", "book"),
        ("locate my phone", "phone"),
    ]
    for query, expected in cases:
        result = extract_item_name(query)
        report(f"Parse '{query}'", result == expected,
               f"got '{result}', expected '{expected}'")


async def test_4_similarity_threshold():
    """Verify CLIP embeddings — similar items score high, dissimilar items score low."""
    print("\n[Test 4] Similarity threshold validation")

    clip_service.load()
    crops = extract_best_crops(str(TEST_VIDEO_1), TARGET_CLASSES)

    # Same item should have high self-similarity
    for name, img in crops.items():
        emb = clip_service.get_image_embedding(img)
        sim = clip_service.cosine_similarity(emb, emb)
        report(f"Self-similarity '{name}'", sim > 0.99, f"{sim:.4f}")

    # Different items should have lower similarity
    names = list(crops.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            emb_a = clip_service.get_image_embedding(crops[names[i]])
            emb_b = clip_service.get_image_embedding(crops[names[j]])
            sim = clip_service.cosine_similarity(emb_a, emb_b)
            below_threshold = sim < SIMILARITY_THRESHOLD
            report(f"Cross-similarity '{names[i]}' vs '{names[j]}'",
                   below_threshold,
                   f"{sim:.3f} {'<' if below_threshold else '>='} {SIMILARITY_THRESHOLD}")


async def test_5_cooldown():
    """Verify that the cooldown prevents duplicate sightings of the same tracked object."""
    print("\n[Test 5] Cooldown mechanism")

    await clean_db()
    crops = extract_best_crops(str(TEST_VIDEO_1), {"laptop"})
    await register_crops(crops)

    # Run detection for longer than one loop but within cooldown period (30s)
    await run_detection(str(TEST_VIDEO_1), duration=12.0)

    count = await count_sightings("laptop")
    # The laptop is static in the video and tracked, so ByteTrack should assign
    # it the same tracker ID → cooldown should limit sightings
    report("Cooldown limits laptop sightings", count <= 3,
           f"{count} sightings (expected <=3 due to cooldown)")


async def test_6_lifecycle():
    """Test item registration and deletion lifecycle via DB operations."""
    print("\n[Test 6] Item lifecycle (register + delete)")

    await clean_db()
    clip_service.load()

    # Register
    crops = extract_best_crops(str(TEST_VIDEO_1), {"cup"})
    await register_crops(crops)

    db = await get_db()
    cursor = await db.execute("SELECT * FROM registered_items WHERE name = 'cup'")
    row = await cursor.fetchone()
    await db.close()
    report("Cup registered", row is not None)

    item_id = row["id"]

    # Verify embedding is valid
    emb = deserialize_embedding(row["embedding"])
    report("Embedding is valid list", isinstance(emb, list) and len(emb) == 512,
           f"length={len(emb)}")

    # Delete
    db = await get_db()
    await db.execute("DELETE FROM sightings WHERE item_id = ?", (item_id,))
    await db.execute("DELETE FROM registered_items WHERE id = ?", (item_id,))
    await db.commit()
    cursor = await db.execute("SELECT * FROM registered_items WHERE name = 'cup'")
    row = await cursor.fetchone()
    await db.close()
    report("Cup deleted", row is None)


async def main():
    global passed, failed

    print("=" * 60)
    print("FindThis — Comprehensive Test Suite")
    print("=" * 60)

    await test_1_basic_detection()
    await test_2_robustness()
    await test_3_search()
    await test_4_similarity_threshold()
    await test_5_cooldown()
    await test_6_lifecycle()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
