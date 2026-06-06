# D405 3D Scanner — Demo Server

Draft demo: serve existing `captures/` records through a FastAPI + Postgres stack.
D405 / Open3D capture (`main_capture.py`) is **not** part of this yet — runs on the host later.

## Architecture

```
captures/  ──(ingest)──►  Postgres (Docker)  ◄──(query)──  FastAPI (host, uv venv)  ──►  Dashboard
```

- **Postgres** runs in Docker Compose (`localhost:5432`), schema auto-loaded from `db/schema.sql`.
- **App** runs on host Windows in a `uv` venv, connects via `DATABASE_URL`.

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
```

Open <http://localhost:8000> for the dashboard.

## API

| Method | Path                  | Description              |
|--------|-----------------------|-------------------------|
| GET    | `/`                   | Dashboard (HTML)        |
| GET    | `/api/captures`       | All captures            |
| GET    | `/api/captures/{id}`  | One capture             |
| GET    | `/api/stats`          | Sync / file counts      |
| GET    | `/files/{name}`       | Raw capture file        |
| GET    | `/docs`               | Swagger UI              |

## Notes

- `db/schema.sql` only auto-runs when the Postgres volume is first created.
  To re-apply after schema changes: `docker compose down -v && docker compose up -d`.
- Re-running ingest is safe (idempotent upsert on `session_id`).
- Cloud (AWS S3 / RDS) sync is the next phase — the `sync_status` / `s3_*` columns are placeholders for it.
