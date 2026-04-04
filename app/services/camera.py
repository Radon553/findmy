from __future__ import annotations

import asyncio
import cv2
import logging
import numpy as np
import threading
import time
from app.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

logger = logging.getLogger(__name__)

# Backends to probe — order matters: DSHOW picks up OBS Virtual Camera on Windows,
# MSMF is the modern Windows default, then generic fallback.
_BACKENDS = [
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_MSMF, "Media Foundation"),
    (cv2.CAP_ANY, "Default"),
]


class CameraService:
    """Manages the webcam / virtual camera capture in a background thread."""

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._video_path: str | None = None
        self._video_fps: float = 25.0
        self._looping = False
        self._camera_index: int = CAMERA_INDEX
        # index -> backend id that worked during enumeration
        self._camera_backends: dict[int, int] = {}

    def enumerate_cameras(self, max_index: int = 10) -> list[dict]:
        """Probe camera indices across multiple backends (DSHOW, MSMF, default).

        Returns a deduplicated list sorted by index. Virtual cameras like
        OBS Virtual Camera are discovered even if the default backend misses them.
        """
        seen: dict[int, dict] = {}
        self._camera_backends.clear()

        for backend_id, backend_name in _BACKENDS:
            for i in range(max_index):
                if i in seen:
                    continue
                try:
                    cap = cv2.VideoCapture(i, backend_id)
                    if cap.isOpened():
                        # Try to read a test frame to confirm it actually works
                        ret, _ = cap.read()
                        cap.release()
                        if ret:
                            seen[i] = {
                                "index": i,
                                "name": f"Camera {i} ({backend_name})",
                                "backend": backend_id,
                            }
                            self._camera_backends[i] = backend_id
                            logger.info(
                                f"Found camera {i} via {backend_name}"
                            )
                    else:
                        cap.release()
                except Exception:
                    pass

        return sorted(seen.values(), key=lambda c: c["index"])

    def start(
        self,
        video_path: str | None = None,
        loop: bool = False,
        camera_index: int | None = None,
    ):
        if self._running:
            return
        self._video_path = video_path
        self._looping = loop
        if video_path:
            self._cap = cv2.VideoCapture(video_path)
            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        else:
            idx = camera_index if camera_index is not None else CAMERA_INDEX
            self._camera_index = idx
            # Use the backend discovered during enumeration, or try all
            backend = self._camera_backends.get(idx)
            if backend is not None:
                self._cap = cv2.VideoCapture(idx, backend)
            else:
                self._cap = self._open_with_fallback(idx)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self._cap.isOpened():
            source = video_path or f"camera index {self._camera_index}"
            raise RuntimeError(f"Cannot open {source}")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _open_with_fallback(index: int) -> cv2.VideoCapture:
        """Try each backend until one opens the camera successfully."""
        for backend_id, backend_name in _BACKENDS:
            try:
                cap = cv2.VideoCapture(index, backend_id)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        logger.info(
                            f"Opened camera {index} via {backend_name}"
                        )
                        return cap
                    cap.release()
            except Exception:
                pass
        # Last resort: default backend without test frame
        return cv2.VideoCapture(index)

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
