"""FastAPI demo server for D405 captures.

Run (from project root, inside uv venv):
    uvicorn server.main:app --reload
Then open http://localhost:8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import repository
from .config import CAPTURES_DIR
from .database import pool
from .models import Capture, Stats

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.open()
    pool.wait()  # 確認 DB 連得到先 serve
    yield
    pool.close()


app = FastAPI(title="D405 3D Scanner — Demo Server", lifespan=lifespan)

# 直接 serve capture 文件（圖片 + ply）
if CAPTURES_DIR.is_dir():
    app.mount("/files", StaticFiles(directory=str(CAPTURES_DIR)), name="files")


@app.get("/api/captures", response_model=list[Capture])
def api_captures():
    return repository.list_captures()


@app.get("/api/captures/{capture_id}", response_model=Capture)
def api_capture(capture_id: int):
    row = repository.get_capture(capture_id)
    if row is None:
        raise HTTPException(status_code=404, detail="capture not found")
    return row


@app.get("/api/stats", response_model=Stats)
def api_stats():
    return repository.get_stats()


@app.get("/")
def dashboard():
    return FileResponse(TEMPLATES_DIR / "dashboard.html")
