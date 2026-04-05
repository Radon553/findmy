import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import IMAGES_DIR, REGISTERED_DIR
from app.database import init_db
from app.routers import camera, detection, items, search, video
from app.services.camera import camera_service
from app.services.detector import detection_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logging.info("Database initialized")
    yield
    detection_service.stop()
    camera_service.stop()
    logging.info("Services stopped")


app = FastAPI(title="FindThis", version="2.0.0", lifespan=lifespan)

app.include_router(items.router)
app.include_router(camera.router)
app.include_router(detection.router)
app.include_router(search.router)
app.include_router(video.router)

app.mount("/data/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
app.mount("/data/registered", StaticFiles(directory=str(REGISTERED_DIR)), name="registered")
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
