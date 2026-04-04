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
    item_id: int
    item_name: str
    image_path: str
    similarity: float
    timestamp: str


class SearchResult(BaseModel):
    item_name: str
    last_seen: str
    image_url: str
    similarity: float


class DetectionStatus(BaseModel):
    running: bool
