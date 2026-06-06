# D405 3D Scanner — Demo Server

Draft demo: serve existing `captures/` records through a FastAPI + Postgres stack.
D405 / Open3D capture (`main_capture.py`) is **not** part of this yet — runs on the host later.

## Architecture

```
captures/ ─(ingest)─► Postgres (Docker) ◄─(query)─ FastAPI (host, uv venv) ─► Dashboard
                           │
                           └─(uploader)─► S3 / MinIO (Docker)   point clouds
```

- **Postgres** runs in Docker Compose (`localhost:5432`), schema auto-loaded from `db/schema.sql`.
- **App** runs on host Windows in a `uv` venv, connects via `DATABASE_URL`.
- **S3 (Phase 2)**: uploader pushes `.ply` point clouds to S3. Locally this is **MinIO**
  (S3-compatible, in Docker); switch to real AWS by changing `.env` (see below).

## Prerequisites

- Docker Desktop (running)
- [uv](https://docs.astral.sh/uv/)
- Python 3.12+

## Setup

```powershell
# 1. config
copy .env.example .env

# 2. start Postgres (auto-runs db/schema.sql on first boot)
docker compose up -d

# 3. host venv + deps
uv venv
uv sync

# 4. backfill existing captures into the DB
uv run python -m scripts.ingest_captures

# 5. run the server
uv run uvicorn server.main:app --reload

# 6. (Phase 2) upload point clouds to S3 / MinIO
uv run python -m uploader.s3_sync --dry-run   # preview, changes nothing
uv run python -m uploader.s3_sync             # actually upload
```

Open <http://localhost:8000> for the dashboard. MinIO console: <http://localhost:9001>
(login `minioadmin` / `minioadmin`).

## Phase 2 — cloud upload

`uploader/s3_sync.py` drives the `captures.sync_status` state machine:

```
local_only ─┐
failed ─────┴─► uploading ─► synced   (all ply in S3)
                          └─► failed   (error; retry_count++, last_error set)
```

- Only captures **with a `.ply` / clean `.ply`** are uploaded; ones without point clouds
  stay `local_only`. Keys land in `s3_ply_key` / `s3_clean_ply_key`.
- Re-running is safe: already-uploaded files are skipped, `synced` rows ignored.
- Failures increment `retry_count` (capped by `MAX_UPLOAD_RETRIES`) and record `last_error`.

### Switch MinIO → real AWS S3

Edit `.env`:

```ini
S3_ENDPOINT_URL=            # leave empty for real AWS
S3_BUCKET=your-bucket
AWS_REGION=ap-east-1
AWS_ACCESS_KEY_ID=...       # your real IAM keys
AWS_SECRET_ACCESS_KEY=...
```

No code changes — `boto3` talks to AWS instead of MinIO.

## API

| Method | Path                  | Description              |
|--------|-----------------------|-------------------------|
| GET    | `/`                   | Dashboard (HTML)        |
| GET    | `/api/captures`       | All captures            |
| GET    | `/api/captures/{id}`  | One capture             |
| GET    | `/api/stats`          | Sync / file counts      |
| GET    | `/files/{name}`       | Raw capture file        |
| GET    | `/docs`               | Swagger UI              |

## Ports

| Port | Service              |
|------|----------------------|
| 5432 | Postgres             |
| 8000 | FastAPI / dashboard  |
| 9000 | MinIO S3 API         |
| 9001 | MinIO web console    |

## Notes

- `db/schema.sql` only auto-runs when the Postgres volume is first created.
  To re-apply after schema changes: `docker compose down -v && docker compose up -d`.
- Re-running ingest is safe (idempotent upsert on `session_id`).
- `docker compose down -v` also wipes the MinIO volume (uploaded objects). Re-run the
  uploader after a reset; it will re-upload because `s3_*` keys were wiped with the DB too.
