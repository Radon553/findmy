from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
REGISTERED_DIR = DATA_DIR / "registered"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "findthis.db"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
REGISTERED_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Detection
YOLO_MODEL = "yolov8n.pt"
CLIP_MODEL = "openai/clip-vit-base-patch32"
YOLO_CONFIDENCE = 0.4
YOLO_IOU_THRESHOLD = 0.45
DETECTION_INTERVAL = 0.5
COOLDOWN_SECONDS = 30

# Blur detection — skip frames below this Laplacian variance
BLUR_THRESHOLD = 100.0

# Minimum bounding box area as fraction of frame area
MIN_BBOX_AREA_RATIO = 0.01

# Dynamic similarity thresholds
SIMILARITY_THRESHOLD = 0.75
SIMILARITY_THRESHOLD_SINGLE = 0.80
SIMILARITY_THRESHOLD_MULTI = 0.70

# Visibility confirmation
CONFIRMATION_SECONDS = 1.0
CONFIRMATION_MAX_GAP = 1.0
MIN_MATCH_COUNT = 2

# Grid crop fallback
GRID_CROP_ENABLED = True
GRID_CROP_SCALES = [(400, 300), (240, 180), (160, 120), (120, 120)]
GRID_CROP_STRIDE = 0.35
GRID_CROP_MAX = 32
GRID_CROP_SIMILARITY_THRESHOLD = 0.76

# Auto-detection — log all YOLO objects, no registration needed
AUTO_DETECT_ENABLED = True
AUTO_DETECT_COOLDOWN = 15  # seconds before re-logging same class in same zone
AUTO_DETECT_MIN_CONFIDENCE = 0.5  # YOLO confidence floor for auto-logging

# Video processing
VIDEO_SAMPLE_INTERVAL = 5

# Spatial context — frame zones (3x3 grid)
ZONE_NAMES = [
    "top-left",    "top-center",    "top-right",
    "middle-left", "center",        "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]

# Confidence logging
MATCH_LOG_PATH = DATA_DIR / "match_log.csv"
