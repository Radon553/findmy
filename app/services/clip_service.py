from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from app.config import CLIP_MODEL


class CLIPService:
    """Handles CLIP embedding generation for item registration and matching."""

    def __init__(self):
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"

    def load(self):
        if self._model is not None:
            return
        self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self._model = CLIPModel.from_pretrained(CLIP_MODEL).to(self._device)
        self._model.eval()

    def get_image_embedding(self, image: Image.Image) -> list[float]:
        """Generate a normalized CLIP embedding for a PIL image."""
        self.load()
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            embedding = self._model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze().cpu().tolist()

    def get_image_embeddings_batch(self, images: list[Image.Image]) -> list[list[float]]:
        """Generate normalized CLIP embeddings for a batch of PIL images."""
        if not images:
            return []
        self.load()
        inputs = self._processor(images=images, return_tensors="pt", padding=True).to(
            self._device
        )
        with torch.no_grad():
            embeddings = self._model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().tolist()

    def classify_crop(self, image: Image.Image, candidates: list[str]) -> tuple[str, float]:
        """Zero-shot classify a crop against candidate labels. Returns (best_label, score)."""
        self.load()
        prompts = [f"a photo of {c}" for c in candidates]
        inputs = self._processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits_per_image.squeeze()
        probs = logits.softmax(dim=-1)
        best_idx = int(probs.argmax())
        return candidates[best_idx], float(probs[best_idx])

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)
        return float(np.dot(a_np, b_np))


# Singleton
clip_service = CLIPService()

# Extended vocabulary for CLIP zero-shot classification — common household objects
CLIP_VOCAB = [
    "airpods", "airpods case", "earbuds", "headphones",
    "phone", "iphone", "smartphone", "cell phone",
    "laptop", "macbook", "computer", "tablet", "ipad",
    "keyboard", "mouse", "monitor", "screen",
    "charger", "cable", "usb cable", "power bank",
    "wallet", "purse", "handbag", "backpack", "bag", "tote bag",
    "keys", "keychain", "key ring",
    "glasses", "sunglasses", "reading glasses",
    "watch", "apple watch", "smartwatch",
    "water bottle", "cup", "mug", "coffee cup", "tumbler", "glass",
    "book", "notebook", "journal", "planner",
    "pen", "pencil", "marker", "highlighter",
    "remote control", "tv remote",
    "hat", "cap", "beanie",
    "jacket", "coat", "hoodie", "sweater",
    "shoe", "sneaker", "sandal", "boot",
    "umbrella", "scarf", "gloves",
    "medicine bottle", "pill bottle",
    "camera", "go pro",
    "speaker", "bluetooth speaker",
    "toy", "stuffed animal", "figurine",
    "plant", "potted plant", "flower vase", "vase",
    "plate", "bowl", "fork", "knife", "spoon",
    "scissors", "tape", "stapler",
    "tissue box", "hand sanitizer",
    "flashlight", "lighter",
    "ring", "necklace", "bracelet", "jewelry",
    "card", "credit card", "id card",
    "person", "dog", "cat",
    "chair", "stool", "couch", "sofa",
    "pillow", "blanket", "towel",
    "clock", "alarm clock",
    "lamp", "desk lamp",
    "box", "container", "basket",
    "tie", "belt", "scarf",
]
