"""Data access — raw SQL via the psycopg3 pool (learning-friendly, no ORM)."""
from .database import pool


def list_captures() -> list[dict]:
    with pool.connection() as conn:
        return conn.execute(
            "SELECT * FROM captures ORDER BY captured_at DESC NULLS LAST, id DESC"
        ).fetchall()


def get_capture(capture_id: int) -> dict | None:
    with pool.connection() as conn:
        return conn.execute(
            "SELECT * FROM captures WHERE id = %s", (capture_id,)
        ).fetchone()


def get_stats() -> dict:
    with pool.connection() as conn:
        return conn.execute(
            """
            SELECT
                COUNT(*)                                          AS total,
                COUNT(*) FILTER (WHERE sync_status = 'synced')     AS synced,
                COUNT(*) FILTER (WHERE sync_status = 'local_only') AS local_only,
                COUNT(*) FILTER (WHERE sync_status = 'uploading')  AS uploading,
                COUNT(*) FILTER (WHERE sync_status = 'failed')     AS failed,
                COUNT(ply_path)                                    AS with_pointcloud,
                COUNT(clean_ply_path)                              AS with_clean
            FROM captures
            """
        ).fetchone()
