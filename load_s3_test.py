"""
Quick CLI to verify S3 loading with Polars.

Usage:
  set env vars: AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or role/profile),
  optional S3_BUCKET override.

  python load_s3_test.py s3://training-data-kg/100mb.xlsx
"""
import sys
from pathlib import Path

import polars as pl

from app.s3_io import convert_to_parquet_if_needed, download_to_local


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python load_s3_test.py s3://bucket/key")
        sys.exit(1)
    s3_path = sys.argv[1]
    local_file = download_to_local(s3_path)
    parquet_path = convert_to_parquet_if_needed(local_file)
    lf = pl.scan_parquet(parquet_path)
    print(f"Loaded: {s3_path}")
    print("Schema:", lf.schema)
    print("Row count (approx):", lf.select(pl.len()).collect()[0, 0])
    print("Preview:")
    print(lf.limit(5).collect())


if __name__ == "__main__":
    main()

