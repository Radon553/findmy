"""
End-to-end test for FindThis detection pipeline.

1. Extract reference crops from a video frame using YOLO
2. Register them as items (generate CLIP embeddings)
3. Feed the video through the camera service
4. Run detection loop
5. Verify sightings are recorded in the database
"""

import asyncio
import cv2
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from app.config import IMAGES_DIR, REGISTERED_DIR, DB_PATH
from app.database import init_db, get_db, serialize_embedding
from app.services.camera import camera_service
from app.services.clip_service import clip_service
from app.services.detector import detection_service

TEST_VIDEO = Path(__file__).parent / "test_video.mp4"

# Items we expect to detect (YOLO COCO class names)
TARGET_CLASSES = {"laptop", "cup", "book"}


def extract_reference_crops(video_path: str, target_classes: set[str]) -> dict[str, Image.Image]:
    """Use YOLO to extract one crop per target class from the video."""
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)

    crops: dict[str, Image.Image] = {}

    # Scan frames to find best crop (highest confidence) for each target
    best: dict[str, tuple[float, Image.Image]] = {}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for f_idx in range(0, frame_count, 5):  # sample every 5 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)[0]
        h, w = frame.shape[:2]

        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls_name not in target_classes:
                continue
            if cls_name in best and best[cls_name][0] >= conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            best[cls_name] = (conf, Image.fromarray(crop_rgb))

    cap.release()

    for name, (conf, img) in best.items():
        print(f"  Reference crop: {name} (conf={conf:.2f}, size={img.size})")
        crops[name] = img

    return crops


async def register_items(crops: dict[str, Image.Image]):
    """Register each crop as an item in the database."""
    clip_service.load()

    for name, image in crops.items():
        embedding = clip_service.get_image_embedding(image)

        # Save reference image
        filename = f"{name}_ref.jpg"
        filepath = REGISTERED_DIR / filename
        image.save(str(filepath))

        db = await get_db()
        try:
            await db.execute(
                """INSERT OR REPLACE INTO registered_items (name, image_path, embedding)
                   VALUES (?, ?, ?)""",
                (name, filename, serialize_embedding(embedding)),
            )
            await db.commit()
        finally:
            await db.close()
        print(f"  Registered: {name}")


async def run_detection_test(video_path: str, duration: float = 15.0):
    """Start camera with video, run detection, wait, then check results."""
    loop = asyncio.get_event_loop()

    # Start camera with video file (looping so detection has time to process)
    camera_service.start(video_path=video_path, loop=True)
    print(f"  Camera started (video: {video_path})")

    # Give camera a moment to buffer a frame
    await asyncio.sleep(1.0)

    # Start detection
    detection_service.start(loop)
    print(f"  Detection started, running for {duration}s...")

    await asyncio.sleep(duration)

    # Stop
    detection_service.stop()
    camera_service.stop()
    print("  Detection stopped")


async def check_results() -> dict[str, list[dict]]:
    """Query the database for sightings."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT item_name, image_path, similarity, timestamp
               FROM sightings ORDER BY timestamp DESC"""
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    results: dict[str, list[dict]] = {}
    for row in rows:
        name = row["item_name"]
        results.setdefault(name, []).append({
            "image": row["image_path"],
            "similarity": row["similarity"],
            "time": row["timestamp"],
        })
    return results


async def main():
    print("=" * 60)
    print("FindThis End-to-End Test")
    print("=" * 60)

    # Clean slate
    if DB_PATH.exists():
        DB_PATH.unlink()
    for f in IMAGES_DIR.glob("*.jpg"):
        f.unlink()
    for f in REGISTERED_DIR.glob("*_ref.jpg"):
        f.unlink()

    await init_db()
    print("\n[1/4] Database initialized")

    # Extract reference crops
    print("\n[2/4] Extracting reference crops from video...")
    crops = extract_reference_crops(str(TEST_VIDEO), TARGET_CLASSES)
    if not crops:
        print("FAIL: No crops extracted!")
        return False

    # Register items
    print("\n[3/4] Registering items with CLIP embeddings...")
    await register_items(crops)

    # Verify registration
    db = await get_db()
    cursor = await db.execute("SELECT COUNT(*) as c FROM registered_items")
    row = await cursor.fetchone()
    await db.close()
    print(f"  {row['c']} items registered in database")

    # Run detection
    print("\n[4/4] Running detection on video...")
    await run_detection_test(str(TEST_VIDEO), duration=20.0)

    # Check results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    results = await check_results()

    if not results:
        print("\nFAIL: No sightings recorded!")
        # Debug: check what's in the images dir
        saved = list(IMAGES_DIR.glob("*.jpg"))
        print(f"  Saved images in data/images: {len(saved)}")
        return False

    all_passed = True
    for item_name in TARGET_CLASSES:
        sightings = results.get(item_name, [])
        if sightings:
            best = max(sightings, key=lambda s: s["similarity"])
            print(f"  {item_name}: {len(sightings)} sightings "
                  f"(best similarity: {best['similarity']:.3f})")
        else:
            print(f"  {item_name}: NO SIGHTINGS")
            all_passed = False

    total = sum(len(v) for v in results.values())
    print(f"\nTotal sightings: {total}")
    print(f"Items detected: {list(results.keys())}")

    if all_passed:
        print("\nPASS: All target items detected!")
    else:
        print("\nPARTIAL: Some items not detected (may need threshold tuning)")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
