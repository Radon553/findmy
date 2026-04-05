"""
End-to-end test for auto-detection pipeline.

Tests that YOLO auto-detects objects from a real video and saves them
as sightings with source='auto', without any registered items.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DB_PATH, IMAGES_DIR
from app.database import init_db, get_db

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "test_video.mp4")


def fresh_db():
    """Wipe and recreate DB."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    asyncio.run(init_db())


async def count_auto_sightings():
    db = await get_db()
    cursor = await db.execute("SELECT * FROM sightings WHERE source='auto' ORDER BY timestamp DESC")
    rows = await cursor.fetchall()
    await db.close()
    return rows


def test_1_yolo_detects_objects_in_video():
    """YOLO should detect objects in the test video frames."""
    import cv2
    from ultralytics import YOLO
    from app.config import YOLO_MODEL, YOLO_CONFIDENCE

    model = YOLO(YOLO_MODEL)
    cap = cv2.VideoCapture(VIDEO_PATH)

    all_classes = set()
    frames_with_detections = 0
    total_sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_sampled += 1
        if total_sampled % 10 != 0:
            continue
        results = model(frame, verbose=False, conf=YOLO_CONFIDENCE)[0]
        if len(results.boxes) > 0:
            frames_with_detections += 1
            for box in results.boxes:
                cls_id = int(box.cls[0])
                all_classes.add(model.names.get(cls_id, f"object_{cls_id}"))

    cap.release()

    print(f"\n  Sampled: {total_sampled // 10} frames")
    print(f"  Frames with detections: {frames_with_detections}")
    print(f"  Classes: {sorted(all_classes)}")

    assert frames_with_detections > 0, "YOLO found no objects"
    assert len(all_classes) > 0, "No classes detected"
    print(f"  PASS: {len(all_classes)} classes in {frames_with_detections} frames")
    return all_classes


def test_2_auto_detect_pipeline():
    """Process video through auto-detect and verify sightings in DB."""
    import threading
    from app.services.video_processor import video_processor

    fresh_db()

    # Create a persistent event loop in a background thread
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    # Init DB on this loop
    future = asyncio.run_coroutine_threadsafe(init_db(), loop)
    future.result(timeout=10)

    # Start video processing
    job_id = video_processor.start_processing(VIDEO_PATH, loop)
    job = video_processor.get_job(job_id)

    # Wait for completion
    start = time.time()
    while job.status == "processing" and time.time() - start < 120:
        time.sleep(1)
        job = video_processor.get_job(job_id)

    print(f"\n  Status: {job.status}")
    print(f"  Frames: {job.processed_frames}/{job.total_frames}")
    print(f"  Sightings reported: {job.sightings_found}")
    if job.error:
        print(f"  Error: {job.error}")

    assert job.status == "completed", f"Failed: {job.error}"
    assert job.sightings_found > 0, "No sightings reported"

    # Wait for async DB writes to flush, then query
    time.sleep(3)

    # Query DB from the same event loop to ensure consistency
    future = asyncio.run_coroutine_threadsafe(count_auto_sightings(), loop)
    sightings = future.result(timeout=10)

    # Shutdown loop
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=5)

    print(f"  Auto sightings in DB: {len(sightings)}")
    assert len(sightings) > 0, f"Video found {job.sightings_found} sightings but DB has 0"

    # Validate schema
    s = sightings[0]
    print(f"\n  Sample sighting:")
    print(f"    item_name: {s['item_name']}")
    print(f"    item_id: {s['item_id']} (None = auto)")
    print(f"    source: {s['source']}")
    print(f"    zone: {s['zone']}")
    print(f"    confidence: {s['similarity']:.2f}")
    print(f"    bbox: ({s['bbox_x']:.2f}, {s['bbox_y']:.2f})")
    nearby = json.loads(s['nearby_objects']) if s['nearby_objects'] else []
    print(f"    nearby: {nearby}")

    assert s['item_id'] is None, f"item_id should be None, got {s['item_id']}"
    assert s['source'] == 'auto'
    assert s['zone'] in [
        "top-left", "top-center", "top-right",
        "middle-left", "center", "middle-right",
        "bottom-left", "bottom-center", "bottom-right",
    ]
    assert 0.0 <= s['bbox_x'] <= 1.0
    assert 0.0 <= s['bbox_y'] <= 1.0
    assert s['similarity'] > 0.0

    classes = sorted(set(x['item_name'] for x in sightings))
    zones = sorted(set(x['zone'] for x in sightings))
    print(f"\n  All classes detected: {classes}")
    print(f"  Zones covered: {zones}")
    print(f"  PASS: {len(sightings)} auto sightings, {len(classes)} classes, {len(zones)} zones")

    return sightings


def test_3_search_finds_auto_sightings(sightings):
    """NL search should match auto-detected items."""
    from app.routers.search import extract_item_name

    if not sightings:
        print("  SKIP: No sightings from previous test")
        return

    target = sightings[0]['item_name']

    queries = [
        f"where is my {target}",
        f"where are the {target}",
        f"find the {target}",
        f"show me the {target}",
        target,
    ]

    print(f"\n  Target class: '{target}'")
    all_ok = True
    for q in queries:
        extracted = extract_item_name(q)
        ok = target in extracted
        print(f"    '{q}' -> '{extracted}' {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    assert all_ok, "NL extraction failed for some queries"
    print(f"  PASS: All search queries extract '{target}'")


def test_4_no_registration_needed():
    """System must work with zero registered items."""
    count = asyncio.run(_count_registered())
    sightings = asyncio.run(count_auto_sightings())

    print(f"\n  Registered items: {count}")
    print(f"  Auto sightings: {len(sightings)}")

    assert count == 0, f"Expected 0 registered items, got {count}"
    assert len(sightings) > 0, "Should have sightings with 0 registered items"
    print(f"  PASS: {len(sightings)} sightings with 0 registrations")


async def _count_registered():
    db = await get_db()
    cursor = await db.execute("SELECT COUNT(*) as cnt FROM registered_items")
    row = await cursor.fetchone()
    await db.close()
    return row['cnt']


if __name__ == "__main__":
    print("=" * 60)
    print("AUTO-DETECT END-TO-END TEST")
    print("=" * 60)

    passed = 0
    failed = 0
    sightings_for_search = []

    tests = [
        ("1. YOLO detects objects in video", lambda: test_1_yolo_detects_objects_in_video()),
        ("2. Auto-detect pipeline (video -> DB)", lambda: test_2_auto_detect_pipeline()),
        ("3. NL search finds auto-detected items", None),  # needs sightings
        ("4. No item registration required", lambda: test_4_no_registration_needed()),
    ]

    # Test 1
    print(f"\nTest 1. YOLO detects objects in video:")
    try:
        test_1_yolo_detects_objects_in_video()
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # Test 2 — feeds results to test 3
    print(f"\nTest 2. Auto-detect pipeline (video -> DB):")
    try:
        sightings_for_search = test_2_auto_detect_pipeline()
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # Test 3
    print(f"\nTest 3. NL search finds auto-detected items:")
    try:
        test_3_search_finds_auto_sightings(sightings_for_search)
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # Test 4
    print(f"\nTest 4. No item registration required:")
    try:
        test_4_no_registration_needed()
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
