from __future__ import annotations

import asyncio
import shutil
import uuid
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import UPLOADS_DIR
from app.models import VideoProcessingStatus
from app.services.video_processor import video_processor

router = APIRouter(prefix="/api/video", tags=["video"])


@router.post("/upload", response_model=VideoProcessingStatus)
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")

    ext = file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "mp4"
    filename = f"{uuid.uuid4().hex[:12]}.{ext}"
    filepath = UPLOADS_DIR / filename

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loop = asyncio.get_event_loop()
    job_id = video_processor.start_processing(str(filepath), loop)
    job = video_processor.get_job(job_id)

    return VideoProcessingStatus(
        job_id=job.job_id, status=job.status,
        total_frames=job.total_frames, processed_frames=job.processed_frames,
        sightings_found=job.sightings_found, progress=job.progress,
    )


@router.get("/status/{job_id}", response_model=VideoProcessingStatus)
async def video_status(job_id: str):
    job = video_processor.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return VideoProcessingStatus(
        job_id=job.job_id, status=job.status,
        total_frames=job.total_frames, processed_frames=job.processed_frames,
        sightings_found=job.sightings_found, progress=job.progress,
        error=job.error,
    )
