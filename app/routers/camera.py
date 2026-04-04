from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response

from app.services.camera import camera_service

router = APIRouter(prefix="/api/camera", tags=["camera"])


@router.post("/start")
async def start_camera():
    """Start the camera capture."""
    try:
        camera_service.start()
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"status": "started"}


@router.post("/stop")
async def stop_camera():
    """Stop the camera capture."""
    camera_service.stop()
    return {"status": "stopped"}


@router.get("/status")
async def camera_status():
    return {"running": camera_service.is_running}


@router.get("/feed")
async def camera_feed():
    """MJPEG stream for live camera preview."""
    if not camera_service.is_running:
        raise HTTPException(503, "Camera is not running. Start it first.")
    return StreamingResponse(
        camera_service.stream_jpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/snapshot")
async def camera_snapshot():
    """Single JPEG snapshot from the camera."""
    if not camera_service.is_running:
        raise HTTPException(503, "Camera is not running")
    jpeg = camera_service.get_jpeg(quality=90)
    if jpeg is None:
        raise HTTPException(503, "No frame available")
    return Response(content=jpeg, media_type="image/jpeg")
