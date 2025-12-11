import tempfile
import uuid
from pathlib import Path
from typing import Tuple

import boto3
import polars as pl

from .config import get_settings


def _ensure_bucket(path: str) -> Tuple[str, str]:
    """
    Split s3://bucket/key into parts.
    """
    if not path.startswith("s3://"):
        raise ValueError("Path must start with s3://bucket/key")
    parts = path.replace("s3://", "", 1).split("/", 1)
    if len(parts) != 2:
        raise ValueError("Invalid S3 path")
    return parts[0], parts[1]


def download_to_local(s3_path: str) -> Path:
    """
    Download an S3 object to a temporary local file.
    """
    bucket_env = get_settings().s3_bucket
    bucket, key = _ensure_bucket(s3_path)
    # Optional override: if s3_bucket env is set, force that bucket.
    bucket = bucket_env or bucket

    s3 = boto3.client("s3", region_name=get_settings().aws_region)
    suffix = Path(key).suffix or ".bin"
    tmp_file = Path(tempfile.gettempdir()) / f"s3obj-{uuid.uuid4().hex}{suffix}"
    with open(tmp_file, "wb") as f:
        s3.download_fileobj(bucket, key, f)
    return tmp_file


def convert_to_parquet_if_needed(local_path: Path) -> Path:
    """
    Convert CSV/Excel to Parquet for faster subsequent reads.
    """
    suffix = local_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return local_path

    lf: pl.LazyFrame
    if suffix in {".csv"}:
        lf = pl.scan_csv(local_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pl.read_excel(local_path)
        lf = df.lazy()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    parquet_path = local_path.with_suffix(".parquet")
    lf.sink_parquet(parquet_path)
    return parquet_path

