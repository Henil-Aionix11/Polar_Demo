import time
from dataclasses import dataclass
from typing import Dict, Optional

import polars as pl

from .config import get_settings


@dataclass
class CachedFrame:
    lf: pl.LazyFrame
    schema: Dict[str, str]
    row_count: int
    parquet_path: str
    created_at: float


class FrameCache:
    """
    In-memory cache keyed by session/dataset id.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, CachedFrame] = {}
        self._settings = get_settings()

    def set(self, key: str, value: CachedFrame) -> None:
        self._cache[key] = value

    def get(self, key: str) -> Optional[CachedFrame]:
        self._evict_expired()
        return self._cache.get(key)

    def _evict_expired(self) -> None:
        ttl = self._settings.cache_ttl_seconds
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now - v.created_at > ttl]
        for k in expired_keys:
            self._cache.pop(k, None)


frame_cache = FrameCache()

