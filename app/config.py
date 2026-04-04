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
CLIP_MODEL = "openai/clip-vit-base-patch32"
SIMILARITY_THRESHOLD = 0.75
DETECTION_INTERVAL = 0.5  # seconds between detection cycles
COOLDOWN_SECONDS = 30  # min seconds between saving same item

# Visibility confirmation
CONFIRMATION_SECONDS = 1.0  # object must be visible this long before saving
CONFIRMATION_MAX_GAP = 1.0  # discard pending sighting after this gap

# Grid crop fallback for custom items
GRID_CROP_ENABLED = True
GRID_CROP_SCALES = [(400, 300), (240, 180), (160, 120), (120, 120)]
GRID_CROP_STRIDE = 0.35  # tighter overlap so boundary objects are fully captured
GRID_CROP_MAX = 32  # enough crops for multi-scale coverage
GRID_CROP_SIMILARITY_THRESHOLD = 0.76  # slightly relaxed for distant/small objects
