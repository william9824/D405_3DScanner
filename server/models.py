"""Pydantic models — API response shapes (mirror the `captures` table)."""
from datetime import datetime

from pydantic import BaseModel


class Capture(BaseModel):
    id: int
    session_id: str | None = None
    captured_at: datetime | None = None

    color_path: str | None = None
    depth_mm_path: str | None = None
    depth_vis_path: str | None = None
    ply_path: str | None = None
    clean_ply_path: str | None = None

    obj_width_mm: float | None = None
    obj_height_mm: float | None = None
    obj_depth_mm: float | None = None

    s3_ply_key: str | None = None
    s3_clean_ply_key: str | None = None

    sync_status: str
    uploaded_at: datetime | None = None
    last_sync_attempt_at: datetime | None = None
    retry_count: int = 0
    last_error: str | None = None


class Stats(BaseModel):
    total: int
    synced: int
    local_only: int
    uploading: int
    failed: int
    with_pointcloud: int
    with_clean: int
