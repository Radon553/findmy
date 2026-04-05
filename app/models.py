from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel


class ItemRegister(BaseModel):
    name: str


class ItemResponse(BaseModel):
    id: int
    name: str
    image_path: str
    created_at: str
    photo_count: int = 1


class SightingResponse(BaseModel):
    id: int
    item_id: Optional[int]
    item_name: str
    image_path: str
    similarity: float
    zone: str
    bbox_x: float
    bbox_y: float
    nearby_objects: List[str]
    source: str
    timestamp: str


class SearchResult(BaseModel):
    item_name: str
    last_seen: str
    image_url: str
    similarity: float
    zone: str
    nearby_objects: List[str]
    sighting_count: int
    first_seen: str


class EventResponse(BaseModel):
    id: int
    item_name: str
    event_type: str
    zone: Optional[str]
    details: Dict
    timestamp: str


class VideoProcessingStatus(BaseModel):
    job_id: str
    status: str
    total_frames: int
    processed_frames: int
    sightings_found: int
    progress: float
    error: Optional[str] = None


class DetectionStatus(BaseModel):
    running: bool
