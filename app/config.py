from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
REGISTERED_DIR = DATA_DIR / "registered"
DB_PATH = DATA_DIR / "findthis.db"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
REGISTERED_DIR.mkdir(parents=True, exist_ok=True)

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Detection
YOLO_MODEL = "yolov8n.pt"
CLIP_MODEL = "google/siglip-so400m-patch14-384"
YOLO_CONFIDENCE = 0.4  # min confidence for YOLO detections
YOLO_IOU_THRESHOLD = 0.45  # NMS IoU threshold to suppress duplicate boxes
DETECTION_INTERVAL = 0.5  # seconds between detection cycles
COOLDOWN_SECONDS = 30  # min seconds between saving same item

# Blur detection — skip frames below this Laplacian variance
BLUR_THRESHOLD = 100.0

# Minimum bounding box area as fraction of frame area (0.5% ≈ 68x68px on 720p)
MIN_BBOX_AREA_RATIO = 0.005

# Dynamic similarity thresholds (based on number of registered embeddings)
SIMILARITY_THRESHOLD = 0.70  # baseline (2 photos)
SIMILARITY_THRESHOLD_SINGLE = 0.72  # items with only 1 photo — stricter
SIMILARITY_THRESHOLD_MULTI = 0.68  # items with 3+ photos — more reliable

# Visibility confirmation
CONFIRMATION_SECONDS = 1.0  # object must be visible this long before saving
CONFIRMATION_MAX_GAP = 1.0  # discard pending sighting after this gap
MIN_MATCH_COUNT = 2  # must be matched in at least this many detection cycles

# Grid crop fallback for custom items
GRID_CROP_ENABLED = True
GRID_CROP_SCALES = [(400, 300), (240, 180), (160, 120), (120, 120), (80, 80), (60, 60)]
GRID_CROP_STRIDE = 0.35  # tighter overlap so boundary objects are fully captured
GRID_CROP_MAX = 48  # enough crops for multi-scale coverage
GRID_CROP_SIMILARITY_THRESHOLD = 0.70  # tunable starting point for SigLIP

# Text-based registration
TEXT_PROMPT_TEMPLATES = [
    "a photo of {}",
    "a photo of {} on a table",
    "a photo of {} on a desk",
    "{} in a room",
    "a close-up photo of {}",
    "{}",
]

# Text-image similarity thresholds (SigLIP text-image scores are in a similar range to image-image)
TEXT_SIMILARITY_THRESHOLD = 0.70
TEXT_SIMILARITY_THRESHOLD_GRID = 0.68

# Confidence logging
MATCH_LOG_PATH = DATA_DIR / "match_log.csv"
