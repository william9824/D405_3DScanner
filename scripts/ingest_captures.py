"""Backfill the `captures` table from existing files in the captures folder.

Idempotent: re-run any time. Groups files by session_id (the timestamp),
maps each file kind to its column, and upserts on session_id.

Usage (from project root, inside uv venv):
    python -m scripts.ingest_captures
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import psycopg

# 容許獨立執行（python -m scripts.ingest_captures）攞到 server 設定
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import CAPTURES_DIR, DATABASE_URL  # noqa: E402

# 注意 alternation 次序：pointcloud_clean 要喺 pointcloud 前面
FILENAME_RE = re.compile(
    r"^(?P<kind>color|depth_mm|depth_vis|pointcloud_clean|pointcloud)"
    r"_(?P<ts>\d{8}_\d{6})\.(?:png|ply)$"
)

# file kind -> DB column
KIND_TO_COLUMN = {
    "color": "color_path",
    "depth_mm": "depth_mm_path",
    "depth_vis": "depth_vis_path",
    "pointcloud": "ply_path",
    "pointcloud_clean": "clean_ply_path",
}

COLUMNS = (
    "color_path", "depth_mm_path", "depth_vis_path", "ply_path", "clean_ply_path",
)

UPSERT = """
INSERT INTO captures
    (session_id, captured_at, color_path, depth_mm_path, depth_vis_path, ply_path, clean_ply_path)
VALUES
    (%(session_id)s, %(captured_at)s, %(color_path)s, %(depth_mm_path)s,
     %(depth_vis_path)s, %(ply_path)s, %(clean_ply_path)s)
ON CONFLICT (session_id) DO UPDATE SET
    captured_at    = EXCLUDED.captured_at,
    color_path     = EXCLUDED.color_path,
    depth_mm_path  = EXCLUDED.depth_mm_path,
    depth_vis_path = EXCLUDED.depth_vis_path,
    ply_path       = EXCLUDED.ply_path,
    clean_ply_path = EXCLUDED.clean_ply_path
RETURNING (xmax = 0) AS inserted
"""


def scan(captures_dir: Path) -> dict[str, dict]:
    """Group capture files into one record per session_id."""
    sessions: dict[str, dict] = {}
    for f in sorted(captures_dir.iterdir()):
        if not f.is_file():
            continue
        m = FILENAME_RE.match(f.name)
        if not m:
            print(f"  skip (unrecognised): {f.name}")
            continue
        sid = m.group("ts")
        rec = sessions.setdefault(
            sid,
            {
                "session_id": sid,
                "captured_at": datetime.strptime(sid, "%Y%m%d_%H%M%S"),
                **{c: None for c in COLUMNS},
            },
        )
        rec[KIND_TO_COLUMN[m.group("kind")]] = f.name
    return sessions


def main() -> None:
    if not CAPTURES_DIR.is_dir():
        sys.exit(f"captures dir not found: {CAPTURES_DIR}")

    print(f"Scanning {CAPTURES_DIR} ...")
    sessions = scan(CAPTURES_DIR)
    if not sessions:
        sys.exit("No recognisable capture files found.")

    inserted = updated = 0
    with psycopg.connect(DATABASE_URL) as conn:
        for sid, rec in sorted(sessions.items()):
            row = conn.execute(UPSERT, rec).fetchone()
            if row[0]:
                inserted += 1
            else:
                updated += 1
            print(f"  {'INSERT' if row[0] else 'update'}  {sid}")
        conn.commit()

    print(f"\nDone. {len(sessions)} sessions  →  {inserted} inserted, {updated} updated.")


if __name__ == "__main__":
    main()
