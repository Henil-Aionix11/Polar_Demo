import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from project root and app/.env if present
_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
_APP_ENV = Path(__file__).resolve().parent / ".env"
load_dotenv(_ROOT_ENV)
load_dotenv(_APP_ENV)


class Settings:
    """
    Centralized settings with environment overrides.
    """

    # Data/location
    download_dir: str = os.getenv(
        "DOWNLOAD_DIR", str(Path(__file__).resolve().parent.parent / "tmp")
    )
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: Optional[str] = os.getenv("S3_BUCKET")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    model: str = os.getenv("OPENAI_MODEL", "gpt-5.1")
    preview_limit: int = int(os.getenv("PREVIEW_LIMIT", "100"))
    max_preview_rows: int = int(os.getenv("MAX_PREVIEW_ROWS", "500"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    max_sql_attempts: int = int(os.getenv("MAX_SQL_ATTEMPTS", "5"))


@lru_cache()
def get_settings() -> Settings:
    return Settings()

