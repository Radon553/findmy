from __future__ import annotations

import asyncio
import cv2
import logging
import numpy as np
import time
import threading
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import supervision as sv

from app.config import (
    YOLO_MODEL,
    SIMILARITY_THRESHOLD,
    DETECTION_INTERVAL,
    COOLDOWN_SECONDS,
    IMAGES_DIR,
)
from app.database import get_db, deserialize_embedding
from app.services.camera import camera_service
from app.services.clip_service import clip_service

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Background detection loop:
    1. YOLO detects objects in the frame
    2. ByteTrack tracks objects across frames
    3. CLIP matches cropped detections against registered item embeddings
    4. Saves sightings above the similarity threshold
    """

    def __init__(self):
        self._yolo: YOLO | None = None
        self._tracker = sv.ByteTrack()
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # track_id -> {item_name: last_save_timestamp}
        self._cooldowns: dict[int, dict[str, float]] = {}

    def load(self):
        if self._yolo is None:
            self._yolo = YOLO(YOLO_MODEL)
        clip_service.load()

    def start(self, loop: asyncio.AbstractEventLoop):
        if self._running:
            return
        self.load()
        self._running = True
        self._loop = loop
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        logger.info("Detection service started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._cooldowns.clear()
        logger.info("Detection service stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def _detection_loop(self):
        while self._running:
            start = time.time()
            try:
                self._process_frame()
            except Exception:
                logger.exception("Error in detection loop")
            elapsed = time.time() - start
            sleep_time = max(0, DETECTION_INTERVAL - elapsed)
            time.sleep(sleep_time)

    def _process_frame(self):
        frame = camera_service.get_frame()
        if frame is None:
            return

        # Run YOLO detection
        results = self._yolo(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            return

        # Track objects across frames
        detections = self._tracker.update_with_detections(detections)

        # Load registered items (run async db call from sync thread)
        future = asyncio.run_coroutine_threadsafe(
            self._load_registered_items(), self._loop
        )
        registered_items = future.result(timeout=5.0)
        if not registered_items:
            return

        # Crop detected objects and generate CLIP embeddings
        crops = []
        valid_indices = []
        h, w = frame.shape[:2]
        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            crop = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(crop_rgb))
            valid_indices.append(i)

        if not crops:
            return

        crop_embeddings = clip_service.get_image_embeddings_batch(crops)

        # Match each crop against registered items
        now = time.time()
        for crop_idx, emb in enumerate(crop_embeddings):
            det_idx = valid_indices[crop_idx]
            tracker_id = int(detections.tracker_id[det_idx]) if detections.tracker_id is not None else crop_idx

            for item in registered_items:
                similarity = clip_service.cosine_similarity(emb, item["embedding"])
                if similarity < SIMILARITY_THRESHOLD:
                    continue

                # Check cooldown
                if tracker_id in self._cooldowns:
                    last = self._cooldowns[tracker_id].get(item["name"], 0)
                    if now - last < COOLDOWN_SECONDS:
                        continue

                # Save sighting
                image_path = self._save_frame(frame)
                asyncio.run_coroutine_threadsafe(
                    self._save_sighting(
                        item["id"], item["name"], image_path, similarity
                    ),
                    self._loop,
                )

                # Update cooldown
                self._cooldowns.setdefault(tracker_id, {})[item["name"]] = now
                logger.info(
                    f"Spotted '{item['name']}' (similarity={similarity:.3f})"
                )

    def _save_frame(self, frame: np.ndarray) -> str:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = IMAGES_DIR / filename
        cv2.imwrite(str(filepath), frame)
        return filename

    @staticmethod
    async def _load_registered_items() -> list[dict]:
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT id, name, embedding FROM registered_items"
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "embedding": deserialize_embedding(row["embedding"]),
                }
                for row in rows
            ]
        finally:
            await db.close()

    @staticmethod
    async def _save_sighting(
        item_id: int, item_name: str, image_path: str, similarity: float
    ):
        db = await get_db()
        try:
            await db.execute(
                """INSERT INTO sightings (item_id, item_name, image_path, similarity)
                   VALUES (?, ?, ?, ?)""",
                (item_id, item_name, image_path, similarity),
            )
            await db.commit()
        finally:
            await db.close()


# Singleton
detection_service = DetectionService()
