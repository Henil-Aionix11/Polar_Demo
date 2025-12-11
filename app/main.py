import time
import uuid
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .data_cache import CachedFrame, frame_cache
from .nlq import generate_sql
from .s3_io import convert_to_parquet_if_needed, download_to_local
from .schemas import (
    ApplyRequest,
    ApplyResponse,
    LoadRequest,
    LoadResponse,
    NLQRequest,
    NLQResponse,
    PageRequest,
    PageResponse,
)

app = FastAPI(title="Polars NLQ Demo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_lazyframe(parquet_path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(parquet_path)


def _schema_dict(lf: pl.LazyFrame) -> dict:
    return {name: str(dtype) for name, dtype in lf.schema.items()}


@app.post("/dataset/load", response_model=LoadResponse)
def load_dataset(req: LoadRequest) -> LoadResponse:
    """
    Load from S3, convert to Parquet if needed, cache, and return preview.
    """
    settings = get_settings()
    try:
        local_file = download_to_local(req.path)
        parquet_path = convert_to_parquet_if_needed(local_file)
        lf = _load_lazyframe(parquet_path)
        schema = _schema_dict(lf)
        row_count = lf.select(pl.len()).collect()[0, 0]
        preview_rows = lf.limit(settings.preview_limit).collect().to_dicts()

        session = uuid.uuid4().hex
        frame_cache.set(
            session,
            CachedFrame(
                lf=lf,
                schema=schema,
                row_count=row_count,
                parquet_path=str(parquet_path),
                created_at=time.time(),
            ),
        )
        return LoadResponse(
            session=session,
            columns=schema,
            row_count=row_count,
            preview=preview_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/nlq", response_model=NLQResponse)
def nlq(req: NLQRequest) -> NLQResponse:
    """
    Generate SQL and return a preview of the result.
    """
    settings = get_settings()
    cached = frame_cache.get(req.session)
    if not cached:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    try:
        sql = generate_sql(req.question, cached.lf, cached.schema)
        result_preview = (
            cached.lf.sql(sql).limit(settings.preview_limit).collect().to_dicts()
        )
        return NLQResponse(sql=sql, preview=result_preview, error=None)
    except Exception as exc:
        return NLQResponse(sql="", preview=[], error=str(exc))


@app.post("/apply", response_model=ApplyResponse)
def apply(req: ApplyRequest) -> ApplyResponse:
    """
    Execute user-provided SQL, persist to parquet, and return a preview.
    """
    settings = get_settings()
    cached = frame_cache.get(req.session)
    if not cached:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    try:
        lf_result = cached.lf.sql(req.sql)
        df = lf_result.collect()
        row_count = df.height

        output_path = req.output_path
        if not output_path:
            output_path = Path(cached.parquet_path).with_name(
                f"result-{uuid.uuid4().hex}.parquet"
            )
            df.write_parquet(output_path)
            output_path = str(output_path)
        else:
            df.write_parquet(output_path)

        preview_rows = df.head(settings.preview_limit).to_dicts()
        return ApplyResponse(
            output_path=str(output_path),
            preview=preview_rows,
            row_count=row_count,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/dataset/page", response_model=PageResponse)
def dataset_page(req: PageRequest) -> PageResponse:
    """
    Return a paginated slice of the current dataset.
    """
    cached = frame_cache.get(req.session)
    if not cached:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    settings = get_settings()
    try:
        effective_limit = min(req.limit, settings.max_preview_rows)
        lf_target = cached.lf.sql(req.sql) if req.sql else cached.lf
        total_rows = (
            lf_target.select(pl.len()).collect()[0, 0]
            if req.sql
            else cached.row_count
        )
        lf_slice = lf_target.slice(req.offset, effective_limit)
        rows = lf_slice.collect().to_dicts()
        return PageResponse(rows=rows, total=total_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {
        "message": "Polars NLQ API. UI is served at /ui. Health at /healthz.",
        "endpoints": ["/dataset/load", "/nlq", "/apply", "/healthz", "/ui"],
    }

