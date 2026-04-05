"""
Comprehensive test suite for FindThis v2 — visual memory system.

Tests:
  1. Basic detection with spatial context (zone, nearby objects)
  2. Robustness on transformed video
  3. Natural language search parsing
  4. Search returns zone + nearby context
  5. Movement events (appeared / moved)
  6. Video upload processing pipeline
  7. Similarity threshold validation
  8. Item lifecycle
"""
from __future__ import annotations

import asyncio
import cv2
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from app.config import IMAGES_DIR, REGISTERED_DIR, DB_PATH, SIMILARITY_THRESHOLD, UPLOADS_DIR
from app.database import init_db, get_db, serialize_embedding, deserialize_embedding
from app.services.camera import camera_service
from app.services.clip_service import clip_service
from app.services.detector import detection_service, compute_zone
from app.services.video_processor import video_processor
from app.routers.search import extract_item_name

TEST_VIDEO_1 = Path(__file__).parent / "test_video.mp4"
TEST_VIDEO_2 = Path(__file__).parent / "test_video2.mp4"
TARGET_CLASSES = {"laptop", "cup", "book"}

passed = 0
failed = 0


def report(name, ok, detail=""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    if ok:
        passed += 1
    else:
        failed += 1


def extract_best_crops(video_path, targets):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    best = {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in range(0, total, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)[0]
        h, w = frame.shape[:2]
        for box in results.boxes:
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls not in targets or (cls in best and best[cls][0] >= conf):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            best[cls] = (conf, Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    cap.release()
    return {k: v[1] for k, v in best.items()}


async def clean():
    if DB_PATH.exists():
        DB_PATH.unlink()
    for f in IMAGES_DIR.glob("*.jpg"):
        f.unlink()
    for f in REGISTERED_DIR.glob("*_ref.jpg"):
        f.unlink()
    await init_db()


async def register(crops):
    clip_service.load()
    for name, img in crops.items():
        emb = clip_service.get_image_embedding(img)
        img.save(str(REGISTERED_DIR / f"{name}_ref.jpg"))
        db = await get_db()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO registered_items (name, image_path, embedding) VALUES (?, ?, ?)",
                (name, f"{name}_ref.jpg", serialize_embedding(emb)),
            )
            await db.execute(
                "INSERT INTO item_embeddings (item_id, embedding, source) "
                "SELECT id, ?, 'test' FROM registered_items WHERE name = ?",
                (serialize_embedding(emb), name),
            )
            await db.commit()
        finally:
            await db.close()


async def run_detect(video, duration):
    loop = asyncio.get_event_loop()
    camera_service.start(video_path=video, loop=True)
    await asyncio.sleep(0.5)
    detection_service.start(loop)
    await asyncio.sleep(duration)
    detection_service.stop()
    camera_service.stop()


async def count_sightings(name=None):
    db = await get_db()
    try:
        if name:
            c = await db.execute("SELECT COUNT(*) as c FROM sightings WHERE item_name=?", (name,))
        else:
            c = await db.execute("SELECT COUNT(*) as c FROM sightings")
        return (await c.fetchone())["c"]
    finally:
        await db.close()


async def test_1_detection_with_context():
    print("\n[Test 1] Detection with spatial context")
    await clean()
    crops = extract_best_crops(str(TEST_VIDEO_1), TARGET_CLASSES)
    report("Extract crops", len(crops) == len(TARGET_CLASSES), f"{list(crops.keys())}")
    await register(crops)
    await run_detect(str(TEST_VIDEO_1), 15.0)

    total = await count_sightings()
    report("Sightings recorded", total > 0, f"{total}")

    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM sightings ORDER BY timestamp DESC LIMIT 1")
        row = await cursor.fetchone()
    finally:
        await db.close()

    report("Sighting has zone field", row["zone"] is not None and len(row["zone"]) > 0, row["zone"])
    report("Sighting has bbox coords", row["bbox_x"] is not None and 0 <= row["bbox_x"] <= 1, f"x={row['bbox_x']:.2f}")
    nearby = json.loads(row["nearby_objects"]) if row["nearby_objects"] else []
    report("nearby_objects is valid JSON list", isinstance(nearby, list), f"{nearby}")
    report("source field populated", row["source"] in ("camera", "video"), row["source"])

    detected = []
    for name in TARGET_CLASSES:
        c = await count_sightings(name)
        if c > 0:
            detected.append(name)
        # Individual items are informational — small objects may not always match
        print(f"    [{name}] {c} sightings")
    report(">=2/3 items detected (camera)", len(detected) >= 2, f"{len(detected)}/3: {detected}")


async def test_2_robustness():
    print("\n[Test 2] Robustness -- transformed video")
    if not TEST_VIDEO_2.exists():
        report("test_video2 exists", False)
        return
    await clean()
    crops = extract_best_crops(str(TEST_VIDEO_1), TARGET_CLASSES)
    await register(crops)
    await run_detect(str(TEST_VIDEO_2), 15.0)
    total = await count_sightings()
    report("Sightings on transformed video", total > 0, f"{total}")
    detected = [n for n in TARGET_CLASSES if await count_sightings(n) > 0]
    report(">=2/3 items detected", len(detected) >= 2, f"{len(detected)}/3: {detected}")


async def test_3_search_parsing():
    print("\n[Test 3] NL search parsing")
    cases = [
        ("where are my keys?", "keys"),
        ("find my wallet", "wallet"),
        ("Where is the laptop?", "laptop"),
        ("when did I last see my AirPods?", "airpods"),
        ("cup", "cup"),
        ("show me the book", "book"),
        ("locate my phone", "phone"),
    ]
    for q, exp in cases:
        report(f"Parse '{q}'", extract_item_name(q) == exp, f"got '{extract_item_name(q)}'")


async def test_4_search_context():
    print("\n[Test 4] Search returns spatial context")
    await clean()
    crops = extract_best_crops(str(TEST_VIDEO_1), TARGET_CLASSES)
    await register(crops)
    await run_detect(str(TEST_VIDEO_1), 12.0)

    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM sightings WHERE item_name = 'laptop' ORDER BY timestamp DESC LIMIT 1"
        )
        row = await cursor.fetchone()
    finally:
        await db.close()

    if not row:
        report("Laptop sighting exists for context check", False)
        return
    report("Zone is a valid zone name", row["zone"] in [
        "top-left", "top-center", "top-right",
        "middle-left", "center", "middle-right",
        "bottom-left", "bottom-center", "bottom-right",
    ], row["zone"])


async def test_5_events():
    print("\n[Test 5] Movement events")
    await clean()
    crops = extract_best_crops(str(TEST_VIDEO_1), {"laptop"})
    await register(crops)
    await run_detect(str(TEST_VIDEO_1), 12.0)

    # Give async event saves a moment to complete
    await asyncio.sleep(1.5)

    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM events WHERE item_name = 'laptop' ORDER BY timestamp ASC"
        )
        events = await cursor.fetchall()
    finally:
        await db.close()

    report("Events recorded", len(events) > 0, f"{len(events)} events")
    if events:
        first = events[0]
        report("First event is 'appeared'", first["event_type"] == "appeared",
               first["event_type"])
        report("Event has zone", first["zone"] is not None and len(first["zone"]) > 0,
               first["zone"])
    else:
        report("First event is 'appeared'", False, "no events")
        report("Event has zone", False, "no events")


async def test_6_video_upload():
    print("\n[Test 6] Video upload processing")
    await clean()
    crops = extract_best_crops(str(TEST_VIDEO_1), TARGET_CLASSES)
    await register(crops)

    loop = asyncio.get_event_loop()
    job_id = video_processor.start_processing(str(TEST_VIDEO_1), loop)
    job = video_processor.get_job(job_id)
    report("Job created", job is not None, job_id)

    # Wait for processing
    for _ in range(60):
        await asyncio.sleep(0.5)
        if job.status != "processing":
            break

    report("Job completed", job.status == "completed", job.status)
    report("Found sightings", job.sightings_found > 0, f"{job.sightings_found}")
    report("Progress reached 1.0", job.progress >= 0.99, f"{job.progress:.2f}")

    # Verify DB sightings have source='video'
    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) as c FROM sightings WHERE source='video'")
        row = await cursor.fetchone()
    finally:
        await db.close()
    report("Sightings marked source=video", row["c"] > 0, f"{row['c']}")


async def test_7_thresholds():
    print("\n[Test 7] Similarity thresholds")
    clip_service.load()
    crops = extract_best_crops(str(TEST_VIDEO_1), TARGET_CLASSES)
    for name, img in crops.items():
        emb = clip_service.get_image_embedding(img)
        sim = clip_service.cosine_similarity(emb, emb)
        report(f"Self-similarity '{name}'", sim > 0.99, f"{sim:.4f}")

    names = list(crops.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ea = clip_service.get_image_embedding(crops[names[i]])
            eb = clip_service.get_image_embedding(crops[names[j]])
            sim = clip_service.cosine_similarity(ea, eb)
            report(f"Cross '{names[i]}' vs '{names[j]}'", sim < SIMILARITY_THRESHOLD, f"{sim:.3f}")


async def test_8_zone_computation():
    print("\n[Test 8] Zone computation")
    report("Top-left", compute_zone(0.1, 0.1) == "top-left")
    report("Center", compute_zone(0.5, 0.5) == "center")
    report("Bottom-right", compute_zone(0.9, 0.9) == "bottom-right")
    report("Top-center", compute_zone(0.5, 0.1) == "top-center")
    report("Middle-right", compute_zone(0.9, 0.5) == "middle-right")


async def test_9_lifecycle():
    print("\n[Test 9] Item lifecycle")
    await clean()
    clip_service.load()
    crops = extract_best_crops(str(TEST_VIDEO_1), {"cup"})
    await register(crops)

    db = await get_db()
    cursor = await db.execute("SELECT * FROM registered_items WHERE name='cup'")
    row = await cursor.fetchone()
    await db.close()
    report("Cup registered", row is not None)

    emb = deserialize_embedding(row["embedding"])
    report("Embedding valid", isinstance(emb, list) and len(emb) == 512, f"len={len(emb)}")

    db = await get_db()
    await db.execute("DELETE FROM sightings WHERE item_id=?", (row["id"],))
    await db.execute("DELETE FROM item_embeddings WHERE item_id=?", (row["id"],))
    await db.execute("DELETE FROM registered_items WHERE id=?", (row["id"],))
    await db.commit()
    cursor = await db.execute("SELECT * FROM registered_items WHERE name='cup'")
    gone = await cursor.fetchone()
    await db.close()
    report("Cup deleted", gone is None)


async def main():
    global passed, failed
    print("=" * 60)
    print("FindThis v2 -- Comprehensive Test Suite")
    print("=" * 60)

    await test_1_detection_with_context()
    await test_2_robustness()
    await test_3_search_parsing()
    await test_4_search_context()
    await test_5_events()
    await test_6_video_upload()
    await test_7_thresholds()
    await test_8_zone_computation()
    await test_9_lifecycle()

    print("\n" + "=" * 60)
    t = passed + failed
    print(f"Results: {passed}/{t} passed, {failed}/{t} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
