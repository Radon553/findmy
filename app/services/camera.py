from __future__ import annotations

import asyncio
import cv2
import numpy as np
import threading
import time
from app.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class CameraService:
    """Manages the webcam (Camo virtual camera) capture in a background thread."""

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._video_path: str | None = None
        self._video_fps: float = 25.0
        self._looping = False

    def start(self, video_path: str | None = None, loop: bool = False):
        if self._running:
            return
        self._video_path = video_path
        self._looping = loop
        if video_path:
            self._cap = cv2.VideoCapture(video_path)
            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        else:
            self._cap = cv2.VideoCapture(CAMERA_INDEX)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self._cap.isOpened():
            source = video_path or f"camera index {CAMERA_INDEX}"
            raise RuntimeError(f"Cannot open {source}")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None
        self._video_path = None

    def _capture_loop(self):
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                if self._video_path:
                    time.sleep(1.0 / self._video_fps)
            else:
                if self._video_path and self._looping:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                elif self._video_path:
                    self._running = False
                    break
                else:
                    time.sleep(0.01)

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_jpeg(self, quality: int = 80) -> bytes | None:
        frame = self.get_frame()
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    async def stream_jpeg(self):
        """Async generator yielding JPEG frames for MJPEG streaming."""
        while self._running:
            jpeg = self.get_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.033)  # ~30fps

    @property
    def is_running(self) -> bool:
        return self._running


# Singleton
camera_service = CameraService()
