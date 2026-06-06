"""Central config — loaded from .env (host runs app, connects to Docker Postgres)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://scanner:scanner@localhost:5432/scanner_db",
)


def _resolve_captures_dir() -> Path:
    raw = os.getenv("CAPTURES_DIR", "captures")
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


CAPTURES_DIR = _resolve_captures_dir()


# ------------------------------------------------------------------
# S3 / cloud upload 
# Local cloud MinIO（S3-compatible）；edit endpoint + credentials -> real AWS S3。
#   - MinIO 本地:  S3_ENDPOINT_URL=http://localhost:9000
#   - AWS    :  S3_ENDPOINT_URL blank（boto3 use AWS official endpoint）
# ------------------------------------------------------------------
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000") or None
S3_BUCKET = os.getenv("S3_BUCKET", "scanner-pointclouds")
S3_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

# Retry count up to 3 when uploading to S3（state machine）
MAX_UPLOAD_RETRIES = int(os.getenv("MAX_UPLOAD_RETRIES", "3"))
