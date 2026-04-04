from __future__ import annotations

import asyncio
import cv2
import logging
import numpy as np
import time
import threading
import uuid
from dataclasses import dataclass, field
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
    CONFIRMATION_SECONDS,
    CONFIRMATION_MAX_GAP,
    GRID_CROP_ENABLED,
    GRID_CROP_SCALES,
    GRID_CROP_STRIDE,
    GRID_CROP_MAX,
    GRID_CROP_SIMILARITY_THRESHOLD,
)
from app.database import get_db, deserialize_embedding
from app.services.camera import camera_service
from app.services.clip_service import clip_service

logger = logging.getLogger(__name__)

MAX_PENDING = 50


@dataclass
class PendingSighting:
    item_id: int
    item_name: str
    first_seen: float
    last_seen: float
    best_similarity: float
    best_frame: np.ndarray


class DetectionService:
    """
    Background detection loop:
    1. YOLO detects objects in the frame
    2. ByteTrack tracks objects across frames
    3. Grid-crop fallback finds objects YOLO misses
    4. CLIP matches cropped detections against registered item embeddings
    5. Pending sighting system confirms visibility for >= 1 second
    6. Saves sightings with the best frame above the similarity threshold
    """

    def __init__(self):
        self._yolo: YOLO | None = None
        self._tracker = sv.ByteTrack()
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # track_id -> {item_name: last_save_timestamp}
        self._cooldowns: dict[int, dict[str, float]] = {}
        # (tracker_id, item_name) -> PendingSighting
        self._pending_sightings: dict[tuple[int, str], PendingSighting] = {}
        self._frame_counter = 0

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
        self._frame_counter = 0
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        logger.info("Detection service started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._cooldowns.clear()
        self._pending_sightings.clear()
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

        self._frame_counter += 1
        h, w = frame.shape[:2]

        # --- YOLO detection + tracking ---
        results = self._yolo(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        crops: list[Image.Image] = []
        crop_tracker_ids: list[int] = []

        if len(detections) > 0:
            detections = self._tracker.update_with_detections(detections)
            for i, bbox in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                crop = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(Image.fromarray(crop_rgb))
                tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else i
                crop_tracker_ids.append(tid)

        # --- Grid crop fallback for custom items ---
        yolo_boxes = detections.xyxy if len(detections) > 0 else np.empty((0, 4))
        if GRID_CROP_ENABLED:
            grid_crops, grid_ids = self._generate_grid_crops(frame, yolo_boxes)
            for gc, gid in zip(grid_crops, grid_ids):
                crops.append(gc)
                crop_tracker_ids.append(gid)

        if not crops:
            self._expire_pending_sightings()
            return

        # --- Load registered items ---
        future = asyncio.run_coroutine_threadsafe(
            self._load_registered_items(), self._loop
        )
        registered_items = future.result(timeout=5.0)
        if not registered_items:
            return

        # --- CLIP match ---
        crop_embeddings = clip_service.get_image_embeddings_batch(crops)
        now = time.time()

        for crop_idx, emb in enumerate(crop_embeddings):
            tracker_id = crop_tracker_ids[crop_idx]
            # Grid crops use a higher threshold
            threshold = (
                GRID_CROP_SIMILARITY_THRESHOLD if tracker_id < 0 else SIMILARITY_THRESHOLD
            )

            for item in registered_items:
                similarity = clip_service.cosine_similarity(emb, item["embedding"])
                if similarity < threshold:
                    continue

                # Check cooldown
                if tracker_id in self._cooldowns:
                    last = self._cooldowns[tracker_id].get(item["name"], 0)
                    if now - last < COOLDOWN_SECONDS:
                        continue

                # Phase A: update or create pending sighting
                key = (tracker_id, item["name"])
                if key in self._pending_sightings:
                    ps = self._pending_sightings[key]
                    ps.last_seen = now
                    if similarity > ps.best_similarity:
                        ps.best_similarity = similarity
                        ps.best_frame = frame.copy()
                else:
                    if len(self._pending_sightings) >= MAX_PENDING:
                        # Evict oldest
                        oldest_key = min(
                            self._pending_sightings,
                            key=lambda k: self._pending_sightings[k].first_seen,
                        )
                        del self._pending_sightings[oldest_key]
                    self._pending_sightings[key] = PendingSighting(
                        item_id=item["id"],
                        item_name=item["name"],
                        first_seen=now,
                        last_seen=now,
                        best_similarity=similarity,
                        best_frame=frame.copy(),
                    )

        # Phase B: confirm or expire pending sightings
        self._confirm_pending_sightings(now)

    def _confirm_pending_sightings(self, now: float):
        """Confirm sightings that have been visible long enough, expire stale ones."""
        to_remove = []
        for key, ps in self._pending_sightings.items():
            tracker_id, item_name = key
            # Expire if object disappeared
            if now - ps.last_seen > CONFIRMATION_MAX_GAP:
                to_remove.append(key)
                continue
            # Confirm if visible long enough
            if now - ps.first_seen >= CONFIRMATION_SECONDS:
                image_path = self._save_frame(ps.best_frame)
                asyncio.run_coroutine_threadsafe(
                    self._save_sighting(
                        ps.item_id, ps.item_name, image_path, ps.best_similarity
                    ),
                    self._loop,
                )
                self._cooldowns.setdefault(tracker_id, {})[item_name] = now
                logger.info(
                    f"Confirmed '{ps.item_name}' (similarity={ps.best_similarity:.3f}, "
                    f"visible for {now - ps.first_seen:.1f}s)"
                )
                to_remove.append(key)

        for key in to_remove:
            del self._pending_sightings[key]

    def _expire_pending_sightings(self):
        """Expire pending sightings when no crops were found at all."""
        now = time.time()
        to_remove = [
            key
            for key, ps in self._pending_sightings.items()
            if now - ps.last_seen > CONFIRMATION_MAX_GAP
        ]
        for key in to_remove:
            del self._pending_sightings[key]

    def _generate_grid_crops(
        self, frame: np.ndarray, existing_boxes: np.ndarray
    ) -> tuple[list[Image.Image], list[int]]:
        """Generate overlapping grid crops, skipping areas covered by YOLO detections."""
        h, w = frame.shape[:2]
        crops: list[Image.Image] = []
        ids: list[int] = []
        count = 0

        for sw, sh in GRID_CROP_SCALES:
            stride_x = int(sw * GRID_CROP_STRIDE)
            stride_y = int(sh * GRID_CROP_STRIDE)
            for y in range(0, h - sh + 1, stride_y):
                for x in range(0, w - sw + 1, stride_x):
                    if count >= GRID_CROP_MAX:
                        return crops, ids
                    box = np.array([x, y, x + sw, y + sh])
                    # Skip if overlapping with a YOLO detection
                    if len(existing_boxes) > 0 and self._max_iou(box, existing_boxes) > 0.5:
                        continue
                    crop = frame[y : y + sh, x : x + sw]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crops.append(Image.fromarray(crop_rgb))
                    # Stable negative ID based on grid position
                    grid_id = -(hash((x, y, sw, sh)) % 1_000_000 + 1)
                    ids.append(grid_id)
                    count += 1

        return crops, ids

    @staticmethod
    def _max_iou(box: np.ndarray, boxes: np.ndarray) -> float:
        """Compute max IoU of a single box against an array of boxes."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter
        iou = inter / np.maximum(union, 1e-6)
        return float(iou.max())

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
