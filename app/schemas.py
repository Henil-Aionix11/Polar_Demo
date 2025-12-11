from typing import Optional

from pydantic import BaseModel, Field


class LoadRequest(BaseModel):
    path: str = Field(..., description="S3 path, e.g., s3://bucket/key.parquet")


class LoadResponse(BaseModel):
    session: str
    columns: dict
    row_count: int
    preview: list


class NLQRequest(BaseModel):
    session: str
    question: str


class NLQResponse(BaseModel):
    sql: str
    preview: list
    error: Optional[str] = None


class ApplyRequest(BaseModel):
    session: str
    sql: str
    output_path: Optional[str] = Field(
        None, description="Optional target s3 path for materialized result"
    )


class ApplyResponse(BaseModel):
    output_path: str
    preview: list
    row_count: int


class PageRequest(BaseModel):
    session: str
    offset: int = Field(0, ge=0)
    limit: int = Field(100, gt=0)
    sql: Optional[str] = Field(
        None,
        description="Optional SQL to apply before paging; when provided, paging is over the transformed result.",
    )


class PageResponse(BaseModel):
    rows: list
    total: int

