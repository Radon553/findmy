import asyncio
from fastapi import APIRouter, HTTPException

from app.models import DetectionStatus
from app.services.camera import camera_service
from app.services.detector import detection_service

router = APIRouter(prefix="/api/detection", tags=["detection"])


@router.post("/start")
async def start_detection():
    """Start the background detection loop."""
    if not camera_service.is_running:
        raise HTTPException(503, "Camera must be running before starting detection")
    loop = asyncio.get_event_loop()
    detection_service.start(loop)
    return {"status": "started"}


@router.post("/stop")
async def stop_detection():
    """Stop the background detection loop."""
    detection_service.stop()
    return {"status": "stopped"}


@router.get("/status", response_model=DetectionStatus)
async def detection_status():
    return DetectionStatus(running=detection_service.is_running)
