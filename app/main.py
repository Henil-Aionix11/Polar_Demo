import io
import logging
import time
import uuid
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import get_settings
from .data_cache import CachedFrame, frame_cache
from .dataset_store import (
    ROW_ID_COL,
    ensure_row_id_column,
    get_dataset_entry,
    load_lazyframe,
    persist_lazyframe,
    register_dataset,
    update_dataset_metadata,
)
from .nlq import _execute_polars_expr, generate_polars_expr, should_persist
from .s3_io import convert_to_parquet_if_needed, download_to_local
from .schemas import (
    LoadRequest,
    LoadResponse,
    NLExprRequest,
    NLExprResponse,
    OpenRequest,
    PageRequest,
    PageResponse,
    UpdateRequest,
    UpdateResponse,
    DownloadRequest,
)

app = FastAPI(title="Polars NLQ Demo", version="0.1.0")
logger = logging.getLogger(__name__)

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
        parquet_path = ensure_row_id_column(parquet_path)
        lf = _load_lazyframe(parquet_path)
        schema = _schema_dict(lf)
        row_count = lf.select(pl.len()).collect()[0, 0]
        preview_rows = lf.limit(settings.preview_limit).collect().to_dicts()
        dataset_id = register_dataset(parquet_path, req.path, schema, row_count)
        session = uuid.uuid4().hex
        frame_cache.set(
            session,
            CachedFrame(
                lf=lf,
                schema=schema,
                row_count=row_count,
                parquet_path=str(parquet_path),
                dataset_id=dataset_id,
                created_at=time.time(),
            ),
        )
        return LoadResponse(
            dataset_id=dataset_id,
            session=session,
            columns=schema,
            row_count=row_count,
            preview=preview_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _download_response(dataset_id: str, code: str | None, fmt: str, session: str | None):
    cached = frame_cache.get(session) if session else None
    logger.info(
        "download_request dataset_id=%s session=%s fmt=%s code_present=%s",
        dataset_id,
        session,
        fmt,
        bool(code),
    )
    try:
        lf, schema, _, parquet_path = load_lazyframe(dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        lf_target = _execute_polars_expr(lf, code) if code else lf
        df = lf_target.collect()
        if ROW_ID_COL in df.columns:
            df = df.drop(ROW_ID_COL)
        logger.info(
            "download_build dataframe rows=%s cols=%s fmt=%s dataset_id=%s",
            df.height,
            len(df.columns),
            fmt,
            dataset_id,
        )
        if fmt == "xlsx":
            import pandas as pd

            buf = io.BytesIO()
            df.to_pandas().to_excel(buf, index=False)
            buf.seek(0)
            logger.info(
                "download_stream size_bytes=%s fmt=xlsx dataset_id=%s",
                buf.getbuffer().nbytes,
                dataset_id,
            )
            return StreamingResponse(
                buf,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": 'attachment; filename="dataset.xlsx"'},
            )
        # default csv
        buf = io.StringIO()
        df.write_csv(buf)
        buf.seek(0)
        logger.info(
            "download_stream size_bytes=%s fmt=csv dataset_id=%s",
            len(buf.getvalue().encode("utf-8")),
            dataset_id,
        )
        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="dataset.csv"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/dataset/download")
def download_dataset(req: DownloadRequest):
    """
    Download the current dataset (optionally with a provided code transformation) as CSV or XLSX.
    """
    cached = frame_cache.get(req.session) if req.session else None
    dataset_id = req.dataset_id or (cached.dataset_id if cached else None)
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id or session required")
    return _download_response(dataset_id, req.code, req.format, req.session)


@app.get("/dataset/download")
def download_dataset_get(
    dataset_id: str | None = None,
    session: str | None = None,
    code: str | None = None,
    format: str = "xlsx",
):
    """
    GET-friendly download endpoint to simplify browser-triggered downloads.
    """
    cached = frame_cache.get(session) if session else None
    resolved_dataset = dataset_id or (cached.dataset_id if cached else None)
    if not resolved_dataset:
        raise HTTPException(status_code=400, detail="dataset_id or session required")
    fmt = "xlsx" if format == "xlsx" else "csv"
    return _download_response(resolved_dataset, code, fmt, session)


@app.post("/nlq/expr", response_model=NLExprResponse)
def nlq_expr(req: NLExprRequest) -> NLExprResponse:
    """
    Generate a Polars expression and return a preview of the result.
    """
    settings = get_settings()
    cached = frame_cache.get(req.session)
    if not cached:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    try:
        code, lf_result = generate_polars_expr(req.question, cached.lf, cached.schema)
        persist_error = None
        updated_cells = None

        # Heuristic: only persist when the code appears to mutate data.
        if cached.dataset_id and should_persist(code):
            try:
                # Detect updated cells before persisting
                old_df = cached.lf.collect()
                new_df = lf_result.collect()
                
                # Track cell changes using vectorized comparison (efficient for large datasets)
                updated_cells = []
                if ROW_ID_COL in old_df.columns and ROW_ID_COL in new_df.columns:
                    # Get common columns (excluding __row_id)
                    common_cols = [c for c in old_df.columns if c in new_df.columns and c != ROW_ID_COL]
                    
                    # For each column, find rows where values changed
                    for col in common_cols:
                        # Join old and new dataframes on __row_id
                        joined = old_df.select([ROW_ID_COL, pl.col(col).alias("old_val")]).join(
                            new_df.select([ROW_ID_COL, pl.col(col).alias("new_val")]),
                            on=ROW_ID_COL,
                            how="inner"
                        )
                        
                        # Find rows where old_val != new_val (comparing as strings to handle nulls)
                        changed = joined.filter(
                            pl.col("old_val").cast(pl.Utf8).fill_null("__NULL__") != 
                            pl.col("new_val").cast(pl.Utf8).fill_null("__NULL__")
                        )
                        
                        # Add to updated_cells list
                        for row in changed.iter_rows(named=True):
                            updated_cells.append({
                                "row_id": row[ROW_ID_COL],
                                "column": col,
                                "old_value": str(row["old_val"]) if row["old_val"] is not None else None,
                                "new_value": str(row["new_val"]) if row["new_val"] is not None else None,
                            })
                
                parquet_path = ensure_row_id_column(Path(cached.parquet_path))
                schema, row_count = persist_lazyframe(cached.dataset_id, lf_result, parquet_path)
                lf_result = _load_lazyframe(parquet_path)
                frame_cache.set(
                    req.session,
                    CachedFrame(
                        lf=lf_result,
                        schema=schema,
                        row_count=row_count,
                        parquet_path=str(parquet_path),
                        dataset_id=cached.dataset_id,
                        created_at=time.time(),
                    ),
                )
            except Exception as exc:  # keep preview but report persistence issue
                persist_error = f"Persist failed: {exc}"

        # Get total count of result rows
        total_count = lf_result.select(pl.len()).collect()[0, 0]
        
        # Get preview (limited rows for chat display)
        result_preview = lf_result.limit(settings.preview_limit).collect().to_dicts()
        
        return NLExprResponse(
            code=code, 
            preview=result_preview, 
            total_count=total_count,
            updated_cells=updated_cells if updated_cells else None,
            error=persist_error
        )
    except Exception as exc:
        return NLExprResponse(code="", preview=[], total_count=0, error=str(exc))


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
        lf_target = _execute_polars_expr(cached.lf, req.code) if req.code else cached.lf
        total_rows = (
            lf_target.select(pl.len()).collect()[0, 0]
            if req.code
            else cached.row_count
        )
        lf_slice = lf_target.slice(req.offset, effective_limit)
        rows = lf_slice.collect().to_dicts()
        return PageResponse(rows=rows, total=total_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/dataset/open", response_model=LoadResponse)
def open_dataset(req: OpenRequest) -> LoadResponse:
    """
    Rehydrate a session from an existing dataset_id without re-uploading.
    """
    settings = get_settings()
    try:
        entry = get_dataset_entry(req.dataset_id)
        parquet_path = ensure_row_id_column(Path(entry["parquet_path"]))
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
                dataset_id=req.dataset_id,
                created_at=time.time(),
            ),
        )
        return LoadResponse(
            dataset_id=req.dataset_id,
            session=session,
            columns=schema,
            row_count=row_count,
            preview=preview_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/dataset/update", response_model=UpdateResponse)
def update_dataset(req: UpdateRequest) -> UpdateResponse:
    """
    Persist an in-place update to the canonical Parquet and refresh the session.
    """
    settings = get_settings()
    try:
        lf, schema, _, parquet_path = load_lazyframe(req.dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        if ROW_ID_COL not in lf.columns:
            parquet_path = ensure_row_id_column(parquet_path)
            lf = _load_lazyframe(parquet_path)

        invalid_cols = [c for c in req.updates if c not in lf.columns or c == ROW_ID_COL]
        if invalid_cols:
            raise ValueError(f"Invalid update columns: {invalid_cols}")

        update_exprs = [
            pl.when(pl.col(ROW_ID_COL) == req.row_id)
            .then(pl.lit(val))
            .otherwise(pl.col(col))
            .alias(col)
            for col, val in req.updates.items()
        ]
        lf_updated = lf.with_columns(update_exprs)
        df_updated = lf_updated.collect()

        # Write back atomically
        from .dataset_store import _write_parquet_atomic  # local import to avoid cycle

        _write_parquet_atomic(df_updated, parquet_path)

        schema = {name: str(dtype) for name, dtype in df_updated.schema.items()}
        row_count = df_updated.height
        update_dataset_metadata(req.dataset_id, schema=schema, row_count=row_count, parquet_path=parquet_path)

        # Refresh session cache (reuse provided session if any)
        session = req.session or uuid.uuid4().hex
        lf_new = _load_lazyframe(parquet_path)
        preview_rows = lf_new.limit(settings.preview_limit).collect().to_dicts()
        frame_cache.set(
            session,
            CachedFrame(
                lf=lf_new,
                schema=schema,
                row_count=row_count,
                parquet_path=str(parquet_path),
                dataset_id=req.dataset_id,
                created_at=time.time(),
            ),
        )
        return UpdateResponse(
            dataset_id=req.dataset_id,
            session=session,
            columns=schema,
            row_count=row_count,
            preview=preview_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {
        "message": "Polars NLQ API. UI is served at /ui. Health at /healthz.",
        "endpoints": [
            "/dataset/load",
            "/nlq/expr",
            "/dataset/page",
            "/healthz",
            "/ui",
        ],
    }

