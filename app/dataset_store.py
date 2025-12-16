import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from filelock import FileLock

from .config import get_settings

ROW_ID_COL = "__row_id"
_METADATA_FILENAME = "datasets.json"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Level Backup/Restore functions for Undo functionality (up to 10 levels)
# ─────────────────────────────────────────────────────────────────────────────

MAX_UNDO_LEVELS = 10


def _backup_path_versioned(parquet_path: Path, version: int) -> Path:
    """Get versioned backup file path: file.v1.parquet, file.v2.parquet, etc."""
    stem = parquet_path.stem
    return parquet_path.with_name(f"{stem}.v{version}.parquet")


def _list_backup_versions(parquet_path: Path) -> List[int]:
    """
    Return sorted list of backup version numbers for this parquet file.
    Versions are monotonic and we keep only the latest MAX_UNDO_LEVELS entries.
    """
    stem = parquet_path.stem
    pattern = re.compile(rf"^{re.escape(stem)}\.v(\d+)\.parquet$")
    versions: List[int] = []
    for candidate in parquet_path.parent.glob(f"{stem}.v*.parquet"):
        match = pattern.match(candidate.name)
        if match:
            versions.append(int(match.group(1)))
    return sorted(versions)


def get_backup_count(parquet_path: Path) -> int:
    """Count how many backup versions exist for this parquet file."""
    return len(_list_backup_versions(parquet_path))


def push_backup(parquet_path: Path) -> int:
    """
    Push current state onto backup stack before mutation.
    Uses a sliding window of MAX_UNDO_LEVELS with monotonic version numbers.
    Example after many updates: file.v11.parquet … file.v20.parquet (max 10 files).
    Returns the new backup count (clamped to MAX_UNDO_LEVELS).
    """
    lock = FileLock(str(parquet_path) + ".lock")
    with lock:
        versions = _list_backup_versions(parquet_path)
        next_version = (versions[-1] + 1) if versions else 1

        # Write new backup
        target = _backup_path_versioned(parquet_path, next_version)
        shutil.copy2(parquet_path, target)

        # Track and prune to sliding window
        versions.append(next_version)
        versions_sorted = sorted(versions)
        if len(versions_sorted) > MAX_UNDO_LEVELS:
            to_delete = versions_sorted[: len(versions_sorted) - MAX_UNDO_LEVELS]
            for ver in to_delete:
                candidate = _backup_path_versioned(parquet_path, ver)
                if candidate.exists():
                    candidate.unlink()
            versions_sorted = versions_sorted[len(to_delete) :]

        remaining_count = len(versions_sorted)

    return remaining_count


def pop_backup(parquet_path: Path) -> int:
    """
    Pop from backup stack (undo): restore latest version and remove it.
    Uses monotonic version numbers with sliding window.
    Returns the number of remaining backups (0 if no backups existed).
    """
    lock = FileLock(str(parquet_path) + ".lock")
    with lock:
        versions = _list_backup_versions(parquet_path)
        if not versions:
            return 0

        latest_version = versions[-1]
        latest_path = _backup_path_versioned(parquet_path, latest_version)
        if not latest_path.exists():
            return len(versions) - 1

        shutil.copy2(latest_path, parquet_path)
        latest_path.unlink()

        remaining_count = len(versions) - 1

    return remaining_count


def has_backups(parquet_path: Path) -> bool:
    """Check if any backup (undo) is available for this parquet file."""
    return len(_list_backup_versions(parquet_path)) > 0


def clear_all_backups(parquet_path: Path) -> None:
    """Delete all backup files for this parquet file."""
    for ver in _list_backup_versions(parquet_path):
        backup = _backup_path_versioned(parquet_path, ver)
        if backup.exists():
            backup.unlink()


# Legacy single-backup functions (kept for backward compatibility)
def _backup_path(parquet_path: Path) -> Path:
    """Get the backup file path for a given parquet file (legacy)."""
    return parquet_path.with_suffix(".prev.parquet")


def create_backup(parquet_path: Path) -> Path:
    """
    Legacy: Use push_backup instead.
    Returns the path of the latest backup created.
    """
    push_backup(parquet_path)
    versions = _list_backup_versions(parquet_path)
    if not versions:
        raise RuntimeError("Backup creation failed.")
    return _backup_path_versioned(parquet_path, versions[-1])


def restore_backup(parquet_path: Path) -> bool:
    """Legacy: Use pop_backup instead. Returns True if undo was performed."""
    # Check if backup exists before attempting
    if not has_backups(parquet_path):
        return False
    pop_backup(parquet_path)
    return True


def has_backup(parquet_path: Path) -> bool:
    """Legacy: Use has_backups instead."""
    return has_backups(parquet_path)


def delete_backup(parquet_path: Path) -> None:
    """Legacy: No longer needed with stack-based approach."""
    pass  # No-op, stack manages itself


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


def set_undo_state(
    dataset_id: str,
    undo_count: int,
    updated_cells: Optional[List[dict]] = None
) -> None:
    """
    Update undo-related metadata for a dataset.
    undo_count: number of available undo levels (0 to MAX_UNDO_LEVELS)
    """
    data = _read_metadata()
    if dataset_id not in data:
        raise ValueError("dataset_id not found")
    data[dataset_id]["undo_count"] = undo_count
    data[dataset_id]["has_undo"] = undo_count > 0  # Legacy compatibility
    if updated_cells is not None:
        data[dataset_id]["last_updated_cells"] = updated_cells
    elif undo_count == 0:
        # Clear updated cells when no undo is available
        data[dataset_id].pop("last_updated_cells", None)
    _write_metadata(data)


def get_undo_state(dataset_id: str) -> Tuple[int, Optional[List[dict]]]:
    """
    Get undo state for a dataset.
    Returns (undo_count, last_updated_cells).
    """
    entry = get_dataset_entry(dataset_id)
    undo_count = entry.get("undo_count", 0)
    # Fallback for legacy has_undo
    if undo_count == 0 and entry.get("has_undo", False):
        undo_count = 1
    updated_cells = entry.get("last_updated_cells")
    return undo_count, updated_cells


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


