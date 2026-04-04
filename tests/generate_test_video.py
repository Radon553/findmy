"""
Generate a synthetic test video by transforming frames from the original test video.
This creates a video with different brightness, slight crop offsets, and frame order
to simulate a different viewing angle — tests detection robustness.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

SRC = Path(__file__).parent / "test_video.mp4"
DST = Path(__file__).parent / "test_video2.mp4"


def generate():
    cap = cv2.VideoCapture(str(SRC))
    if not cap.isOpened():
        print("Cannot open source video")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Slight crop to simulate different angle
    crop_x, crop_y = 40, 20
    new_w, new_h = w - 2 * crop_x, h - 2 * crop_y

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(DST), fourcc, fps, (new_w, new_h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop
        frame = frame[crop_y:h - crop_y, crop_x:w - crop_x]

        # Vary brightness per frame to simulate lighting changes
        brightness = 0.8 + 0.4 * np.sin(frame_count * 0.1)
        frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)

        # Slight blur on some frames
        if frame_count % 3 == 0:
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Generated {DST.name}: {frame_count} frames, {new_w}x{new_h}")
    return True


if __name__ == "__main__":
    success = generate()
    sys.exit(0 if success else 1)
