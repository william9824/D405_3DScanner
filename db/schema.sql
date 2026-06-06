-- ============================================================
-- D405 3D Scanner — Local PostgreSQL schema
-- Stage: <learning> local server
-- ============================================================

CREATE TABLE IF NOT EXISTS captures (
    id              SERIAL PRIMARY KEY,

    -- natural key：from capture timestamp，format: "20260605_183959"
    session_id      VARCHAR(30) UNIQUE,
    captured_at     TIMESTAMP,

    -- local file path
    color_path      TEXT,
    depth_mm_path   TEXT,         -- raw depth map (mm) PNG
    depth_vis_path  TEXT,         -- visible depth PNG
    ply_path        TEXT,
    clean_ply_path  TEXT,         -- postprocessing point cloud，nullable

    -- Scanner measurements (bounding box, mm)
    obj_width_mm    REAL,
    obj_height_mm   REAL,
    obj_depth_mm    REAL,

    -- AWS (S3) object key，sync checker
    s3_ply_key          TEXT,
    s3_clean_ply_key    TEXT,

    -- Sync state machine
    sync_status     VARCHAR(30) DEFAULT 'local_only'
                    CHECK (sync_status IN (
                        'local_only',   
                        'uploading',    
                        'synced',       
                        'failed'        --
                    )),
    uploaded_at         TIMESTAMP,
    last_sync_attempt_at TIMESTAMP,
    retry_count         INTEGER DEFAULT 0,
    last_error          TEXT
);

-- Upload agent scan records
CREATE INDEX IF NOT EXISTS idx_captures_sync_status
    ON captures (sync_status);
