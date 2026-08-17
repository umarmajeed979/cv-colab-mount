"""
Wraps wherever files/results actually live -- local disk today,
swap the internals for S3/GCS later without touching callers.
"""
from pathlib import Path


def save_upload(file_bytes: bytes, filename: str, dest_dir: str = "data/uploads") -> str:
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    path = Path(dest_dir) / filename
    path.write_bytes(file_bytes)
    return str(path)
