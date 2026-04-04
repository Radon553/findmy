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
GRID_CROP_SCALES = [(320, 240), (200, 150)]
GRID_CROP_STRIDE = 0.5  # stride as fraction of window size
GRID_CROP_MAX = 16  # max grid crops per frame
GRID_CROP_SIMILARITY_THRESHOLD = 0.80  # higher threshold for grid crops
