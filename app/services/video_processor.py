from __future__ import annotations

import asyncio
import cv2
import json
import logging
import numpy as np
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

from app.config import (
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD,
    SIMILARITY_THRESHOLD, IMAGES_DIR, VIDEO_SAMPLE_INTERVAL, ZONE_NAMES,
    AUTO_DETECT_ENABLED, AUTO_DETECT_COOLDOWN, AUTO_DETECT_MIN_CONFIDENCE,
)
from app.database import get_db, deserialize_embedding, serialize_embedding
from app.services.clip_service import clip_service

logger = logging.getLogger(__name__)


def _compute_zone(cx: float, cy: float) -> str:
    col = min(int(cx * 3), 2)
    row = min(int(cy * 3), 2)
    return ZONE_NAMES[row * 3 + col]


def _draw_bbox(frame, label, cx_norm, cy_norm, w, h, bbox_frac=0.15, color=(99, 102, 241)):
    """Draw a bounding box + label on a frame copy."""
    out = frame.copy()
    bw, bh = int(bbox_frac * w / 2), int(bbox_frac * h / 2)
    px, py = int(cx_norm * w), int(cy_norm * h)
    x1, y1 = max(0, px - bw), max(0, py - bh)
    x2, y2 = min(w, px + bw), min(h, py + bh)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


@dataclass
class VideoJob:
    job_id: str
    video_path: str
    status: str = "processing"
    total_frames: int = 0
    processed_frames: int = 0
    sightings_found: int = 0
    error: str | None = None

    @property
    def progress(self) -> float:
        return min(self.processed_frames / self.total_frames, 1.0) if self.total_frames else 0.0


class VideoProcessor:
    def __init__(self):
        self._yolo: YOLO | None = None
        self._jobs: dict[str, VideoJob] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def load(self):
        if self._yolo is None:
            self._yolo = YOLO(YOLO_MODEL)
        clip_service.load()

    def get_job(self, job_id: str) -> VideoJob | None:
        return self._jobs.get(job_id)

    def start_processing(self, video_path: str, loop: asyncio.AbstractEventLoop) -> str:
        self.load()
        self._loop = loop
        job_id = uuid.uuid4().hex[:12]
        job = VideoJob(job_id=job_id, video_path=video_path)
        self._jobs[job_id] = job
        threading.Thread(target=self._process_video, args=(job,), daemon=True).start()
        return job_id

    def _process_video(self, job: VideoJob):
        try:
            cap = cv2.VideoCapture(job.video_path)
            if not cap.isOpened():
                job.status = "error"
                job.error = "Could not open video file"
                return

            job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            try:
                registered = asyncio.run_coroutine_threadsafe(
                    self._load_registered_items(), self._loop
                ).result(timeout=10.0) or []
            except Exception:
                registered = []
                logger.info("No registered items found — auto-detect only mode")

            auto_cooldowns: dict[tuple[str, str], int] = {}  # (class, zone) -> last frame

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                job.processed_frames = frame_idx
                if frame_idx % VIDEO_SAMPLE_INTERVAL != 0:
                    continue
                self._process_frame(frame, registered, job, auto_cooldowns, frame_idx)

            cap.release()
            job.status = "completed"
            logger.info(f"Video {job.job_id}: {job.sightings_found} sightings in {job.total_frames} frames")
        except Exception as e:
            logger.exception(f"Video processing error {job.job_id}")
            job.status = "error"
            job.error = str(e)

    def _process_frame(self, frame: np.ndarray, registered: list[dict], job: VideoJob,
                       auto_cooldowns: dict, frame_idx: int):
        h, w = frame.shape[:2]
        results = self._yolo(frame, verbose=False, conf=YOLO_CONFIDENCE, iou=YOLO_IOU_THRESHOLD)[0]
        if len(results.boxes) == 0:
            return

        yolo_names = self._yolo.names
        crops, crop_meta = [], []
        for bi, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            crop = frame[y1:y2, x1:x2]
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo_names.get(cls_id, f"object_{cls_id}")
            # Find nearby objects
            nearby = []
            tcx, tcy = (x1 + x2) / 2, (y1 + y2) / 2
            for oi, obox in enumerate(results.boxes):
                if oi == bi:
                    continue
                ox1, oy1, ox2, oy2 = map(int, obox.xyxy[0])
                ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                if ((tcx - ocx)**2 + (tcy - ocy)**2)**0.5 < 300:
                    oid = int(obox.cls[0])
                    name = yolo_names.get(oid, f"object_{oid}")
                    if name not in nearby:
                        nearby.append(name)
            crop_meta.append({"cx": cx, "cy": cy, "nearby": nearby[:5],
                              "class_name": class_name, "conf": conf})

        if not crops:
            return

        # Auto-log all YOLO detections with bounding boxes
        if AUTO_DETECT_ENABLED:
            for ci, meta in enumerate(crop_meta):
                if meta["conf"] < AUTO_DETECT_MIN_CONFIDENCE:
                    continue
                zone = _compute_zone(meta["cx"], meta["cy"])
                key = (meta["class_name"], zone)
                cooldown_frames = int(AUTO_DETECT_COOLDOWN / max(VIDEO_SAMPLE_INTERVAL / 30, 0.1))
                if frame_idx - auto_cooldowns.get(key, -cooldown_frames - 1) < cooldown_frames:
                    continue
                auto_cooldowns[key] = frame_idx
                # Draw bbox on frame copy and save
                label = f"{meta['class_name']} {meta['conf']:.0%}"
                annotated = _draw_bbox(frame, label, meta["cx"], meta["cy"], w, h)
                fn = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
                cv2.imwrite(str(IMAGES_DIR / fn), annotated)

                # Auto-register item
                crop_pil = crops[ci] if ci < len(crops) else None
                if crop_pil:
                    emb = clip_service.get_image_embedding(crop_pil)
                    asyncio.run_coroutine_threadsafe(
                        self._auto_register_item(meta["class_name"], fn, emb), self._loop
                    )

                asyncio.run_coroutine_threadsafe(
                    self._save_sighting(
                        None, meta["class_name"], fn, meta["conf"],
                        zone, meta["cx"], meta["cy"], meta["nearby"], "auto",
                    ),
                    self._loop,
                )
                job.sightings_found += 1

        # Also match against registered items (if any)
        if registered:
            embeddings = clip_service.get_image_embeddings_batch(crops)
            for emb, meta in zip(embeddings, crop_meta):
                for item in registered:
                    sim = max(clip_service.cosine_similarity(emb, ie) for ie in item["embeddings"])
                    if sim >= SIMILARITY_THRESHOLD:
                        label = f"{item['name']} {sim:.0%}"
                        annotated = _draw_bbox(frame, label, meta["cx"], meta["cy"], w, h)
                        fn = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
                        cv2.imwrite(str(IMAGES_DIR / fn), annotated)
                        zone = _compute_zone(meta["cx"], meta["cy"])
                        asyncio.run_coroutine_threadsafe(
                            self._save_sighting(
                                item["id"], item["name"], fn, sim,
                                zone, meta["cx"], meta["cy"], meta["nearby"],
                            ),
                            self._loop,
                        )
                        job.sightings_found += 1
                        break

    @staticmethod
    async def _load_registered_items():
        db = await get_db()
        try:
            cursor = await db.execute("SELECT id, name FROM registered_items")
            items = await cursor.fetchall()
            result = []
            for item in items:
                cursor = await db.execute("SELECT embedding FROM item_embeddings WHERE item_id = ?", (item["id"],))
                rows = await cursor.fetchall()
                embeddings = [deserialize_embedding(r["embedding"]) for r in rows]
                if not embeddings:
                    cursor = await db.execute("SELECT embedding FROM registered_items WHERE id = ?", (item["id"],))
                    lr = await cursor.fetchone()
                    if lr and lr["embedding"]:
                        embeddings = [deserialize_embedding(lr["embedding"])]
                if embeddings:
                    result.append({"id": item["id"], "name": item["name"], "embeddings": embeddings})
            return result
        finally:
            await db.close()

    @staticmethod
    async def _save_sighting(item_id, item_name, image_path, similarity, zone, bbox_x, bbox_y, nearby, source="video"):
        db = await get_db()
        try:
            await db.execute(
                "INSERT INTO sightings (item_id, item_name, image_path, similarity, zone, bbox_x, bbox_y, nearby_objects, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (item_id, item_name, image_path, similarity, zone, bbox_x, bbox_y, json.dumps(nearby), source),
            )
            await db.commit()
        finally:
            await db.close()

    @staticmethod
    async def _auto_register_item(class_name, image_path, embedding):
        """Auto-register a detected class as a tracked item (idempotent)."""
        db = await get_db()
        try:
            cursor = await db.execute("SELECT id FROM registered_items WHERE name=?", (class_name,))
            if await cursor.fetchone():
                return
            emb_json = serialize_embedding(embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding))
            await db.execute(
                "INSERT INTO registered_items (name, image_path, embedding) VALUES (?, ?, ?)",
                (class_name, image_path, emb_json),
            )
            cursor = await db.execute("SELECT id FROM registered_items WHERE name=?", (class_name,))
            row = await cursor.fetchone()
            if row:
                await db.execute(
                    "INSERT INTO item_embeddings (item_id, embedding, source) VALUES (?, ?, ?)",
                    (row["id"], emb_json, "auto"),
                )
            await db.commit()
            logger.info(f"[auto-register] Created item '{class_name}'")
        except Exception:
            logger.exception(f"Failed to auto-register '{class_name}'")
        finally:
            await db.close()


video_processor = VideoProcessor()
