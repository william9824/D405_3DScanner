"""Phase 2 — upload point clouds to S3 (MinIO locally / real AWS later).

Drives the `captures.sync_status` state machine:

    local_only ─┐
    failed ─────┴─► uploading ─► synced      (all ply uploaded OK)
                              └─► failed      (error; retry_count++, last_error set)

Only captures that have a ply / clean_ply file are considered "uploadable".
Captures with no point cloud stay `local_only` (nothing destined for cloud).

Usage (from project root, inside uv venv):
    python -m uploader.s3_sync            # upload everything pending
    python -m uploader.s3_sync --dry-run  # show what would upload, touch nothing
"""
import argparse
import sys
from pathlib import Path

import boto3
import psycopg
from botocore.client import Config
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import (  # noqa: E402
    CAPTURES_DIR,
    DATABASE_URL,
    MAX_UPLOAD_RETRIES,
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_REGION,
    S3_SECRET_KEY,
)

# (DB column holding the file, DB column holding its S3 key)
PLY_COLUMNS = [
    ("ply_path", "s3_ply_key"),
    ("clean_ply_path", "s3_clean_ply_key"),
]

SELECT_PENDING = """
    SELECT id, session_id, ply_path, clean_ply_path, s3_ply_key, s3_clean_ply_key
    FROM captures
    WHERE sync_status IN ('local_only', 'failed')
      AND retry_count < %(max_retries)s
      AND (ply_path IS NOT NULL OR clean_ply_path IS NOT NULL)
    ORDER BY captured_at NULLS LAST, id
"""


def make_s3_client():
    """boto3 S3 client — works for both MinIO (endpoint_url set) and AWS (unset)."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket(s3) -> None:
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except ClientError:
        print(f"  creating bucket: {S3_BUCKET}")
        s3.create_bucket(Bucket=S3_BUCKET)


def s3_key(session_id: str | None, filename: str) -> str:
    return f"pointclouds/{session_id or 'unknown'}/{filename}"


def upload_one(s3, cap: dict, dry_run: bool) -> dict[str, str]:
    """Upload this capture's ply files. Returns {key_column: s3_key} for uploaded files."""
    new_keys: dict[str, str] = {}
    for path_col, key_col in PLY_COLUMNS:
        filename = cap[path_col]
        if not filename or cap[key_col]: # filename is None if no file exists, or if already uploaded → skip
            continue
        local_path = CAPTURES_DIR / filename
        if not local_path.is_file():
            raise FileNotFoundError(f"missing local file: {local_path}")
        key = s3_key(cap["session_id"], filename)
        size_kb = local_path.stat().st_size / 1024
        if dry_run:
            print(f"    [dry-run] would upload {filename} ({size_kb:,.0f} KB) -> s3://{S3_BUCKET}/{key}")
        else:
            s3.upload_file(str(local_path), S3_BUCKET, key)
            print(f"    uploaded {filename} ({size_kb:,.0f} KB) -> s3://{S3_BUCKET}/{key}")
        new_keys[key_col] = key
    return new_keys


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload pending point clouds to S3.")
    ap.add_argument("--dry-run", action="store_true", help="show plan, change nothing")
    args = ap.parse_args()

    s3 = make_s3_client()
    endpoint = S3_ENDPOINT_URL or "AWS default"
    print(f"S3 endpoint: {endpoint}  bucket: {S3_BUCKET}")

    if not args.dry_run:
        ensure_bucket(s3)

    synced = failed = skipped = 0
    with psycopg.connect(DATABASE_URL, row_factory=psycopg.rows.dict_row) as conn:
        pending = conn.execute(
            SELECT_PENDING, {"max_retries": MAX_UPLOAD_RETRIES}
        ).fetchall()
        print(f"{len(pending)} capture(s) pending upload.\n")

        for cap in pending:
            print(f"• {cap['session_id']} (id={cap['id']})")
            if args.dry_run:
                upload_one(s3, cap, dry_run=True)
                skipped += 1
                continue

            # → uploading
            conn.execute(
                "UPDATE captures SET sync_status='uploading', last_sync_attempt_at=now() WHERE id=%s",
                (cap["id"],),
            )
            conn.commit()

            try:
                new_keys = upload_one(s3, cap, dry_run=False)
                # merge existing S3 keys with newly uploaded ones (in case some files were already uploaded in previous attempts)
                ply_key = new_keys.get("s3_ply_key", cap["s3_ply_key"])
                clean_key = new_keys.get("s3_clean_ply_key", cap["s3_clean_ply_key"])
                conn.execute(
                    """
                    UPDATE captures
                    SET sync_status='synced', uploaded_at=now(),
                        s3_ply_key=%s, s3_clean_ply_key=%s, last_error=NULL
                    WHERE id=%s
                    """,
                    (ply_key, clean_key, cap["id"]),
                )
                conn.commit()
                synced += 1
                print("    -> synced\n")
            except Exception as e:  # noqa: BLE001
                conn.execute(
                    """
                    UPDATE captures
                    SET sync_status='failed', retry_count=retry_count+1, last_error=%s
                    WHERE id=%s
                    """,
                    (str(e), cap["id"]),
                )
                conn.commit()
                failed += 1
                print(f"    -> FAILED: {e}\n")

    if args.dry_run:
        print(f"Dry run done. {skipped} capture(s) would be processed.")
    else:
        print(f"Done. {synced} synced, {failed} failed.")


if __name__ == "__main__":
    main()
