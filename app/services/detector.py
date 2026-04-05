from __future__ import annotations

import asyncio
import csv
import cv2
import json
import logging
import numpy as np
import time
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
import supervision as sv

from app.config import (
    YOLO_MODEL,
    YOLO_CONFIDENCE,
    YOLO_IOU_THRESHOLD,
    SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD_SINGLE,
    SIMILARITY_THRESHOLD_MULTI,
    DETECTION_INTERVAL,
    COOLDOWN_SECONDS,
    IMAGES_DIR,
    BLUR_THRESHOLD,
    MIN_BBOX_AREA_RATIO,
    CONFIRMATION_SECONDS,
    CONFIRMATION_MAX_GAP,
    MIN_MATCH_COUNT,
    GRID_CROP_ENABLED,
    GRID_CROP_SCALES,
    GRID_CROP_STRIDE,
    GRID_CROP_MAX,
    GRID_CROP_SIMILARITY_THRESHOLD,
    MATCH_LOG_PATH,
    ZONE_NAMES,
)
from app.database import get_db, deserialize_embedding
from app.services.camera import camera_service
from app.services.clip_service import clip_service

logger = logging.getLogger(__name__)

MAX_PENDING = 50
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def compute_zone(cx: float, cy: float) -> str:
    col = min(int(cx * 3), 2)
    row = min(int(cy * 3), 2)
    return ZONE_NAMES[row * 3 + col]


@dataclass
class PendingSighting:
    item_id: int
    item_name: str
    first_seen: float
    last_seen: float
    best_similarity: float
    best_frame: np.ndarray
    match_count: int = 1
    bbox_cx: float = 0.5
    bbox_cy: float = 0.5
    zone: str = "center"
    nearby_objects: list[str] = field(default_factory=list)


class DetectionService:
    """Background detection with spatial awareness and movement tracking."""

    def __init__(self):
        self._yolo: YOLO | None = None
        self._tracker = sv.ByteTrack()
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cooldowns: dict[int, dict[str, float]] = {}
        self._pending_sightings: dict[tuple[int, str], PendingSighting] = {}
        self._frame_counter = 0
        self._last_zones: dict[str, str] = {}
        self._match_log_file = None
        self._match_log_writer = None

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
        self._last_zones.clear()
        self._start_match_log()
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        logger.info("Detection service started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        for name, zone in self._last_zones.items():
            self._fire_event(name, "disappeared", zone, {})
        self._cooldowns.clear()
        self._pending_sightings.clear()
        self._last_zones.clear()
        self._stop_match_log()
        logger.info("Detection service stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # --- Spatial ---

    def _get_nearby(self, idx, detections, names):
        if len(detections) < 2:
            return []
        tb = detections.xyxy[idx]
        tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
        nearby = []
        for i, box in enumerate(detections.xyxy):
            if i == idx:
                continue
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if ((tcx - cx)**2 + (tcy - cy)**2)**0.5 < 300:
                cid = int(detections.class_id[i]) if detections.class_id is not None else -1
                n = names.get(cid, f"object_{cid}")
                if n not in nearby:
                    nearby.append(n)
        return nearby[:5]

    # --- Events ---

    def _fire_event(self, item_name, event_type, zone, details):
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._save_event(item_name, event_type, zone, details), self._loop
            )

    @staticmethod
    async def _save_event(item_name, event_type, zone, details):
        db = await get_db()
        try:
            c = await db.execute("SELECT id FROM registered_items WHERE name=?", (item_name,))
            row = await c.fetchone()
            if not row:
                return
            await db.execute(
                "INSERT INTO events (item_id,item_name,event_type,zone,details) VALUES(?,?,?,?,?)",
                (row["id"], item_name, event_type, zone, json.dumps(details)),
            )
            await db.commit()
        finally:
            await db.close()

    # --- Match logging ---

    def _start_match_log(self):
        try:
            hdr = not MATCH_LOG_PATH.exists() or MATCH_LOG_PATH.stat().st_size == 0
            self._match_log_file = open(MATCH_LOG_PATH, "a", newline="", buffering=1)
            self._match_log_writer = csv.writer(self._match_log_file)
            if hdr:
                self._match_log_writer.writerow(["timestamp", "item_name", "tracker_id", "similarity", "threshold", "matched", "source"])
        except Exception:
            logger.exception("Failed to open match log")

    def _stop_match_log(self):
        if self._match_log_file:
            try:
                self._match_log_file.close()
            except Exception:
                pass
            self._match_log_file = self._match_log_writer = None

    def _log_match(self, name, tid, sim, thr, matched, src):
        if self._match_log_writer:
            try:
                self._match_log_writer.writerow([datetime.now().isoformat(timespec="milliseconds"), name, tid, f"{sim:.4f}", f"{thr:.2f}", "yes" if matched else "no", src])
            except Exception:
                pass

    # --- Preprocessing ---

    @staticmethod
    def _enhance_crop(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _clahe.apply(l)
        return cv2.filter2D(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR), -1, _sharpen_kernel)

    @staticmethod
    def _get_threshold(emb_count, is_grid):
        if is_grid:
            return GRID_CROP_SIMILARITY_THRESHOLD
        if emb_count >= 3:
            return SIMILARITY_THRESHOLD_MULTI
        if emb_count == 1:
            return SIMILARITY_THRESHOLD_SINGLE
        return SIMILARITY_THRESHOLD

    # --- Main loop ---

    def _detection_loop(self):
        while self._running:
            t0 = time.time()
            try:
                self._process_frame()
            except Exception:
                logger.exception("Error in detection loop")
            time.sleep(max(0, DETECTION_INTERVAL - (time.time() - t0)))

    def _process_frame(self):
        frame = camera_service.get_frame()
        if frame is None:
            return

        # Blur detection — skip blurry frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD:
            self._expire_pending()
            return

        self._frame_counter += 1
        h, w = frame.shape[:2]
        frame_area = w * h

        results = self._yolo(frame, verbose=False, conf=YOLO_CONFIDENCE, iou=YOLO_IOU_THRESHOLD)[0]
        detections = sv.Detections.from_ultralytics(results)
        yolo_names = self._yolo.names

        crops, tids, sources, bboxes, det_idxs = [], [], [], [], []

        if len(detections) > 0:
            detections = self._tracker.update_with_detections(detections)
            for i, bbox in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) * (y2 - y1) < frame_area * MIN_BBOX_AREA_RATIO:
                    continue
                crops.append(Image.fromarray(cv2.cvtColor(self._enhance_crop(frame[y1:y2, x1:x2]), cv2.COLOR_BGR2RGB)))
                tids.append(int(detections.tracker_id[i]) if detections.tracker_id is not None else i)
                sources.append("yolo")
                bboxes.append(((x1 + x2) / 2 / w, (y1 + y2) / 2 / h))
                det_idxs.append(i)

        yolo_boxes = detections.xyxy if len(detections) > 0 else np.empty((0, 4))
        if GRID_CROP_ENABLED:
            gc, gi, gctr = self._grid_crops(frame, yolo_boxes)
            for c, i, ct in zip(gc, gi, gctr):
                crops.append(c); tids.append(i); sources.append("grid"); bboxes.append(ct); det_idxs.append(-1)

        if not crops:
            self._expire_pending()
            return

        registered = asyncio.run_coroutine_threadsafe(self._load_items(), self._loop).result(timeout=5.0)
        if not registered:
            return

        embs = clip_service.get_image_embeddings_batch(crops)
        now = time.time()

        for ci, emb in enumerate(embs):
            tid, src, is_grid = tids[ci], sources[ci], tids[ci] < 0
            cx, cy = bboxes[ci]
            di = det_idxs[ci]
            zone = compute_zone(cx, cy)
            nearby = self._get_nearby(di, detections, yolo_names) if di >= 0 else []

            for item in registered:
                best = max(clip_service.cosine_similarity(emb, ie) for ie in item["embeddings"])
                thr = self._get_threshold(item["embedding_count"], is_grid)
                matched = best >= thr
                self._log_match(item["name"], tid, best, thr, matched, src)
                if not matched:
                    continue
                if tid in self._cooldowns and now - self._cooldowns[tid].get(item["name"], 0) < COOLDOWN_SECONDS:
                    continue

                key = (tid, item["name"])
                if key in self._pending_sightings:
                    ps = self._pending_sightings[key]
                    ps.last_seen = now
                    ps.match_count += 1
                    ps.zone = zone
                    ps.bbox_cx = cx
                    ps.bbox_cy = cy
                    ps.nearby_objects = nearby
                    if best > ps.best_similarity:
                        ps.best_similarity = best
                        ps.best_frame = frame.copy()
                else:
                    if len(self._pending_sightings) >= MAX_PENDING:
                        del self._pending_sightings[min(self._pending_sightings, key=lambda k: self._pending_sightings[k].first_seen)]
                    self._pending_sightings[key] = PendingSighting(
                        item_id=item["id"], item_name=item["name"],
                        first_seen=now, last_seen=now, best_similarity=best,
                        best_frame=frame.copy(), match_count=1,
                        bbox_cx=cx, bbox_cy=cy, zone=zone, nearby_objects=nearby,
                    )

        self._confirm_pending(now)

    def _confirm_pending(self, now):
        to_rm = []
        for key, ps in self._pending_sightings.items():
            tid, name = key
            if now - ps.last_seen > CONFIRMATION_MAX_GAP:
                to_rm.append(key); continue
            if now - ps.first_seen >= CONFIRMATION_SECONDS and ps.match_count >= MIN_MATCH_COUNT:
                path = self._save_frame(ps.best_frame)
                asyncio.run_coroutine_threadsafe(
                    self._save_sighting(ps.item_id, ps.item_name, path, ps.best_similarity, ps.zone, ps.bbox_cx, ps.bbox_cy, ps.nearby_objects, "camera"),
                    self._loop,
                )
                self._cooldowns.setdefault(tid, {})[name] = now
                prev = self._last_zones.get(name)
                if prev is None:
                    self._fire_event(name, "appeared", ps.zone, {"similarity": round(ps.best_similarity, 3)})
                elif prev != ps.zone:
                    self._fire_event(name, "moved", ps.zone, {"from": prev, "to": ps.zone})
                self._last_zones[name] = ps.zone
                logger.info(f"Confirmed '{ps.item_name}' in {ps.zone} (sim={ps.best_similarity:.3f}, matched {ps.match_count} frames, nearby={ps.nearby_objects})")
                to_rm.append(key)
        for k in to_rm:
            del self._pending_sightings[k]

    def _expire_pending(self):
        now = time.time()
        for k in [k for k, ps in self._pending_sightings.items() if now - ps.last_seen > CONFIRMATION_MAX_GAP]:
            del self._pending_sightings[k]

    def _grid_crops(self, frame, existing):
        h, w = frame.shape[:2]
        crops, ids, ctrs = [], [], []
        n = 0
        for sw, sh in GRID_CROP_SCALES:
            sx, sy = int(sw * GRID_CROP_STRIDE), int(sh * GRID_CROP_STRIDE)
            for y in range(0, h - sh + 1, sy):
                for x in range(0, w - sw + 1, sx):
                    if n >= GRID_CROP_MAX:
                        return crops, ids, ctrs
                    box = np.array([x, y, x + sw, y + sh])
                    if len(existing) > 0 and self._max_iou(box, existing) > 0.5:
                        continue
                    crops.append(Image.fromarray(cv2.cvtColor(self._enhance_crop(frame[y:y+sh, x:x+sw]), cv2.COLOR_BGR2RGB)))
                    ids.append(-(hash((x, y, sw, sh)) % 1_000_000 + 1))
                    ctrs.append(((x + sw/2) / w, (y + sh/2) / h))
                    n += 1
        return crops, ids, ctrs

    @staticmethod
    def _max_iou(box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        return float((inter / np.maximum((box[2]-box[0])*(box[3]-box[1]) + (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]) - inter, 1e-6)).max())

    def _save_frame(self, frame):
        fn = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(IMAGES_DIR / fn), frame)
        return fn

    @staticmethod
    async def _load_items():
        db = await get_db()
        try:
            c = await db.execute("SELECT id, name FROM registered_items")
            items = await c.fetchall()
            if not items:
                return []
            result = []
            for item in items:
                c = await db.execute("SELECT embedding FROM item_embeddings WHERE item_id=?", (item["id"],))
                rows = await c.fetchall()
                embs = [deserialize_embedding(r["embedding"]) for r in rows]
                if not embs:
                    c = await db.execute("SELECT embedding FROM registered_items WHERE id=?", (item["id"],))
                    lr = await c.fetchone()
                    if lr and lr["embedding"]:
                        embs = [deserialize_embedding(lr["embedding"])]
                if embs:
                    result.append({"id": item["id"], "name": item["name"], "embeddings": embs, "embedding_count": len(embs)})
            return result
        finally:
            await db.close()

    @staticmethod
    async def _save_sighting(item_id, item_name, image_path, similarity, zone, bbox_x, bbox_y, nearby, source):
        db = await get_db()
        try:
            await db.execute(
                "INSERT INTO sightings (item_id,item_name,image_path,similarity,zone,bbox_x,bbox_y,nearby_objects,source) VALUES(?,?,?,?,?,?,?,?,?)",
                (item_id, item_name, image_path, similarity, zone, bbox_x, bbox_y, json.dumps(nearby), source),
            )
            await db.commit()
        finally:
            await db.close()


detection_service = DetectionService()
