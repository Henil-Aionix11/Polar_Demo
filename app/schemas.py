from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LoadRequest(BaseModel):
    path: str = Field(..., description="S3 path, e.g., s3://bucket/key.parquet")


class LoadResponse(BaseModel):
    dataset_id: str
    session: str
    columns: dict
    row_count: int
    preview: list


class NLExprRequest(BaseModel):
    session: str
    question: str


class NLExprResponse(BaseModel):
    code: str
    preview: list
    error: Optional[str] = None


class PageRequest(BaseModel):
    session: str
    offset: int = Field(0, ge=0)
    limit: int = Field(100, gt=0)
    code: Optional[str] = Field(
        None,
        description="Optional Polars expression code to apply before paging; when provided, paging is over the transformed result.",
    )


class PageResponse(BaseModel):
    rows: list
    total: int


class OpenRequest(BaseModel):
    dataset_id: str = Field(..., description="Existing dataset identifier returned from load.")


class UpdateRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset identifier to update.")
    row_id: int = Field(..., description="Row identifier added as __row_id during ingest.")
    updates: Dict[str, Any] = Field(..., description="Column -> new value map.")
    session: Optional[str] = Field(
        None,
        description="Optional session to refresh; otherwise a new session will be created.",
    )


class UpdateResponse(BaseModel):
    dataset_id: str
    session: str
    columns: dict
    row_count: int
    preview: list


class DownloadRequest(BaseModel):
    dataset_id: Optional[str] = Field(
        None, description="Dataset identifier; required if session does not resolve to a dataset."
    )
    session: Optional[str] = Field(None, description="Optional session to derive dataset_id and cached lf.")
    code: Optional[str] = Field(
        None,
        description="Optional Polars expression to apply before download; same format as /nlq/expr results.",
    )
    format: str = Field(
        "csv",
        description="Export format: csv or xlsx.",
        pattern="^(csv|xlsx)$",
    )

