import json
import time
import uuid
from pathlib import Path
from typing import Dict, Tuple

import polars as pl
from filelock import FileLock

from .config import get_settings

ROW_ID_COL = "__row_id"
_METADATA_FILENAME = "datasets.json"


def _metadata_path() -> Path:
    settings = get_settings()
    base = Path(settings.download_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / _METADATA_FILENAME


def _read_metadata() -> Dict[str, dict]:
    path = _metadata_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_metadata(data: Dict[str, dict]) -> None:
    path = _metadata_path()
    path.write_text(json.dumps(data, indent=2))


def register_dataset(parquet_path: Path, source_path: str, schema: dict, row_count: int) -> str:
    """
    Create a dataset_id entry pointing to the canonical Parquet path.
    """
    dataset_id = uuid.uuid4().hex
    data = _read_metadata()
    data[dataset_id] = {
        "parquet_path": str(parquet_path),
        "source_path": source_path,
        "schema": schema,
        "row_count": row_count,
        "updated_at": time.time(),
    }
    _write_metadata(data)
    return dataset_id


def update_dataset_metadata(dataset_id: str, schema: dict, row_count: int, parquet_path: Path) -> None:
    data = _read_metadata()
    if dataset_id not in data:
        raise ValueError("dataset_id not found")
    data[dataset_id].update(
        {
            "parquet_path": str(parquet_path),
            "schema": schema,
            "row_count": row_count,
            "updated_at": time.time(),
        }
    )
    _write_metadata(data)


def get_dataset_entry(dataset_id: str) -> dict:
    data = _read_metadata()
    entry = data.get(dataset_id)
    if not entry:
        raise ValueError("dataset_id not found")
    return entry


def ensure_row_id_column(parquet_path: Path) -> Path:
    """
    Guarantee the canonical Parquet has a stable row id column.
    Rewrites the file in-place (atomic replace).
    """
    lf = pl.scan_parquet(parquet_path)
    if ROW_ID_COL in lf.columns:
        return parquet_path
    df = lf.collect()
    df = df.with_row_count(ROW_ID_COL)
    _write_parquet_atomic(df, parquet_path, suffix=".rowid.tmp.parquet")
    return parquet_path


def load_lazyframe(dataset_id: str) -> Tuple[pl.LazyFrame, dict, int, Path]:
    entry = get_dataset_entry(dataset_id)
    parquet_path = Path(entry["parquet_path"])
    lf = pl.scan_parquet(parquet_path)
    schema = {name: str(dtype) for name, dtype in lf.collect_schema().items()}
    row_count = entry.get("row_count") or lf.select(pl.len()).collect()[0, 0]
    return lf, schema, row_count, parquet_path


def _write_parquet_atomic(df: pl.DataFrame, target_path: Path, suffix: str = ".tmp.parquet") -> None:
    """
    Write to a temp file and atomically replace the target. Uses a file lock to avoid
    concurrent writers stepping on each other.
    """
    lock = FileLock(str(target_path) + ".lock")
    with lock:
        tmp_path = target_path.with_suffix(suffix)
        df.write_parquet(tmp_path)
        # Path.replace overwrites on both POSIX/Windows.
        tmp_path.replace(target_path)


def persist_lazyframe(dataset_id: str, lf: pl.LazyFrame, parquet_path: Path) -> Tuple[dict, int]:
    """
    Persist a LazyFrame to the canonical Parquet, preserving/adding ROW_ID_COL.
    Returns (schema, row_count).
    """
    df = lf.collect()
    if ROW_ID_COL not in df.columns:
        df = df.with_row_count(ROW_ID_COL)
    _write_parquet_atomic(df, parquet_path)
    schema = {name: str(dtype) for name, dtype in df.schema.items()}
    row_count = df.height
    update_dataset_metadata(dataset_id, schema=schema, row_count=row_count, parquet_path=parquet_path)
    return schema, row_count


