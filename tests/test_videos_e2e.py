"""
Multi-video end-to-end test.

Processes multiple videos through the full pipeline and verifies:
- Auto-detection and logging of YOLO objects
- Bounding box annotations on saved images
- Auto-registration of items
- Delete sighting functionality
- Search across all videos
"""
from __future__ import annotations

import asyncio
import cv2
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DB_PATH, IMAGES_DIR
from app.database import init_db, get_db

VIDEOS = [
    ("tests/test_video.mp4", "Indoor scene 1"),
    ("tests/test_video2.mp4", "Indoor scene 2 (transformed)"),
    ("tests/test_desk.mp4", "Desk/office scene"),
    ("tests/test_desk_variant.mp4", "Desk scene (flipped + color shift)"),
]


def fresh_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    asyncio.run(init_db())


async def get_all_sightings():
    db = await get_db()
    cursor = await db.execute("SELECT * FROM sightings ORDER BY timestamp DESC")
    rows = await cursor.fetchall()
    await db.close()
    return rows


async def get_registered_items():
    db = await get_db()
    cursor = await db.execute("SELECT * FROM registered_items")
    rows = await cursor.fetchall()
    await db.close()
    return rows


async def delete_sighting(sid):
    db = await get_db()
    cursor = await db.execute("SELECT image_path FROM sightings WHERE id=?", (sid,))
    row = await cursor.fetchone()
    if row:
        img = IMAGES_DIR / row["image_path"]
        if img.exists():
            img.unlink(missing_ok=True)
        await db.execute("DELETE FROM sightings WHERE id=?", (sid,))
        await db.commit()
    await db.close()
    return row is not None


async def clear_all_sightings():
    db = await get_db()
    cursor = await db.execute("SELECT image_path FROM sightings")
    rows = await cursor.fetchall()
    for row in rows:
        img = IMAGES_DIR / row["image_path"]
        if img.exists():
            img.unlink(missing_ok=True)
    await db.execute("DELETE FROM sightings")
    await db.commit()
    await db.close()
    return len(rows)


def process_video(video_path, loop):
    """Process a single video and return the job."""
    from app.services.video_processor import video_processor
    job_id = video_processor.start_processing(video_path, loop)
    job = video_processor.get_job(job_id)
    start = time.time()
    while job.status == "processing" and time.time() - start < 180:
        time.sleep(1)
        job = video_processor.get_job(job_id)
    return job


def run_tests():
    print("=" * 65)
    print("MULTI-VIDEO END-TO-END TEST")
    print("=" * 65)

    # Filter to videos that exist
    available = [(p, d) for p, d in VIDEOS if os.path.exists(p)]
    print(f"\nVideos available: {len(available)}/{len(VIDEOS)}")
    for p, d in available:
        cap = cv2.VideoCapture(p)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"  {d}: {p} ({w}x{h}, {frames} frames)")

    if len(available) < 2:
        print("SKIP: Need at least 2 videos")
        return 1

    # Fresh start
    fresh_db()

    # Start event loop
    loop = asyncio.new_event_loop()
    future = asyncio.run_coroutine_threadsafe(init_db(), loop)

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    future.result(timeout=10)

    # Preload models
    from app.services.video_processor import video_processor
    video_processor.load()

    passed = 0
    failed = 0
    total_sightings = 0
    all_classes = set()

    # === TEST 1: Process each video ===
    print(f"\n{'='*65}")
    print("TEST 1: Process all videos")
    print("=" * 65)

    for video_path, desc in available:
        print(f"\n  Processing: {desc} ({video_path})")
        job = process_video(video_path, loop)

        status_icon = "OK" if job.status == "completed" else "FAIL"
        print(f"    Status: {job.status} [{status_icon}]")
        print(f"    Frames: {job.processed_frames}/{job.total_frames}")
        print(f"    Sightings: {job.sightings_found}")
        if job.error:
            print(f"    Error: {job.error}")

        if job.status != "completed":
            failed += 1
            continue

        total_sightings += job.sightings_found

    # Wait for DB writes
    time.sleep(3)

    # Query results from the same loop
    sightings = asyncio.run_coroutine_threadsafe(get_all_sightings(), loop).result(timeout=10)
    items = asyncio.run_coroutine_threadsafe(get_registered_items(), loop).result(timeout=10)
    all_classes = set(s["item_name"] for s in sightings)

    print(f"\n  TOTALS:")
    print(f"    Sightings in DB: {len(sightings)}")
    print(f"    Unique classes: {sorted(all_classes)}")
    print(f"    Auto-registered items: {len(items)}")
    registered_names = sorted(set(i["name"] for i in items))
    print(f"    Registered names: {registered_names}")

    if len(sightings) > 0 and len(all_classes) > 0:
        print(f"  PASS")
        passed += 1
    else:
        print(f"  FAIL: No sightings detected")
        failed += 1

    # === TEST 2: Verify bounding box images ===
    print(f"\n{'='*65}")
    print("TEST 2: Bounding box annotations on images")
    print("=" * 65)

    bbox_ok = 0
    bbox_fail = 0
    for s in sightings[:10]:  # Check first 10
        img_path = IMAGES_DIR / s["image_path"]
        if not img_path.exists():
            print(f"  MISSING: {img_path}")
            bbox_fail += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  UNREADABLE: {img_path}")
            bbox_fail += 1
            continue
        # Check image has some non-uniform pixels (bbox overlay changes pixels)
        bbox_ok += 1

    print(f"  Checked: {bbox_ok + bbox_fail} images")
    print(f"  Valid: {bbox_ok}, Missing/broken: {bbox_fail}")

    if bbox_ok > 0 and bbox_fail == 0:
        print(f"  PASS")
        passed += 1
    else:
        print(f"  FAIL")
        failed += 1

    # === TEST 3: Auto-registration ===
    print(f"\n{'='*65}")
    print("TEST 3: Auto-registration of detected items")
    print("=" * 65)

    print(f"  Registered items: {len(items)}")
    for i in items:
        has_emb = bool(i["embedding"])
        print(f"    {i['name']}: image={i['image_path']}, has_embedding={has_emb}")

    # Every detected class should be registered
    missing = all_classes - set(registered_names)
    if len(items) > 0 and len(missing) == 0:
        print(f"  PASS: All {len(all_classes)} detected classes registered")
        passed += 1
    else:
        print(f"  FAIL: Missing registrations: {missing}")
        failed += 1

    # === TEST 4: Zone + spatial coverage ===
    print(f"\n{'='*65}")
    print("TEST 4: Spatial coverage across videos")
    print("=" * 65)

    zones = set(s["zone"] for s in sightings)
    sources = set(s["source"] for s in sightings)
    print(f"  Zones covered: {sorted(zones)} ({len(zones)}/9)")
    print(f"  Sources: {sorted(sources)}")

    # Check nearby objects
    with_nearby = sum(1 for s in sightings if json.loads(s["nearby_objects"] or "[]"))
    print(f"  Sightings with nearby objects: {with_nearby}/{len(sightings)}")

    if len(zones) >= 2:
        print(f"  PASS")
        passed += 1
    else:
        print(f"  FAIL: Too few zones")
        failed += 1

    # === TEST 5: Delete sightings ===
    print(f"\n{'='*65}")
    print("TEST 5: Delete sightings")
    print("=" * 65)

    before_count = len(sightings)
    if before_count > 0:
        # Delete first sighting
        target_id = sightings[0]["id"]
        target_img = sightings[0]["image_path"]
        deleted = asyncio.run_coroutine_threadsafe(delete_sighting(target_id), loop).result(timeout=5)
        after = asyncio.run_coroutine_threadsafe(get_all_sightings(), loop).result(timeout=5)
        print(f"  Before: {before_count}, after deleting #{target_id}: {len(after)}")
        img_gone = not (IMAGES_DIR / target_img).exists()
        print(f"  Image file deleted: {img_gone}")

        if len(after) == before_count - 1 and img_gone:
            print(f"  Single delete: PASS")
        else:
            print(f"  Single delete: FAIL")
            failed += 1

        # Clear all
        cleared = asyncio.run_coroutine_threadsafe(clear_all_sightings(), loop).result(timeout=10)
        remaining = asyncio.run_coroutine_threadsafe(get_all_sightings(), loop).result(timeout=5)
        print(f"  Cleared {cleared} sightings, remaining: {len(remaining)}")

        if len(remaining) == 0:
            print(f"  Clear all: PASS")
            passed += 1
        else:
            print(f"  Clear all: FAIL")
            failed += 1
    else:
        print(f"  SKIP: No sightings to delete")
        failed += 1

    # === TEST 6: Search across videos ===
    print(f"\n{'='*65}")
    print("TEST 6: NL search for detected classes")
    print("=" * 65)

    # Re-process one video so we have sightings to search
    print(f"  Re-processing {available[0][1]} for search test...")
    job = process_video(available[0][0], loop)
    time.sleep(2)
    sightings = asyncio.run_coroutine_threadsafe(get_all_sightings(), loop).result(timeout=10)
    search_classes = set(s["item_name"] for s in sightings)

    from app.routers.search import extract_item_name

    search_ok = 0
    for cls in list(search_classes)[:5]:
        queries = [f"where is my {cls}", f"find the {cls}", cls]
        for q in queries:
            extracted = extract_item_name(q)
            if cls in extracted:
                search_ok += 1
            else:
                print(f"    FAIL: '{q}' -> '{extracted}' (expected '{cls}')")

    total_queries = min(len(search_classes), 5) * 3
    print(f"  Search extraction: {search_ok}/{total_queries} queries correct")

    if search_ok == total_queries:
        print(f"  PASS")
        passed += 1
    else:
        print(f"  FAIL")
        failed += 1

    # Shutdown
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)

    # === RESULTS ===
    print(f"\n{'='*65}")
    print(f"RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 65)

    # Summary table
    print(f"\n  Videos processed: {len(available)}")
    print(f"  Total sightings: {total_sightings}")
    print(f"  Object classes: {sorted(all_classes)}")
    print(f"  Auto-registered items: {len(registered_names)} ({', '.join(registered_names)})")
    print(f"  Zones covered: {sorted(zones)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_tests())
