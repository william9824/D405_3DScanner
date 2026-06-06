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
