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

    def get_text_embeddings(self, prompts: list[str]) -> list[list[float]]:
        """Generate normalized CLIP embeddings for a list of text prompts."""
        if not prompts:
            return []
        self.load()
        inputs = self._processor(text=prompts, return_tensors="pt", padding=True).to(
            self._device
        )
        with torch.no_grad():
            embeddings = self._model.get_text_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().tolist()

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)
        return float(np.dot(a_np, b_np))


# Singleton
clip_service = CLIPService()
