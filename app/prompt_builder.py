from typing import List, Tuple

import polars as pl

from .config import get_settings


def sample_rows(lf: pl.LazyFrame, n: int = 15) -> List[dict]:
    """
    Collect a small sample for prompt grounding.
    """
    try:
        return lf.limit(n).collect().to_dicts()
    except Exception:
        return []


def build_expr_prompts(question: str, lf: pl.LazyFrame, schema: dict) -> Tuple[str, str]:
    """
    Build system/user prompts for NL -> Polars expression (LazyFrame pipeline).
    """
    settings = get_settings()
    samples = sample_rows(lf, n=15)
    schema_lines = [f"- {k}: {v}" for k, v in schema.items()]

    system_prompt = f"""
You are a senior data engineer generating a Polars LazyFrame transformation.
Rules:
- Use only the provided columns and Polars API (lazy mode).
- Do not import anything, do not read/write files, do not mutate globals.
- Start from the provided LazyFrame named lf.
- Produce a single assignment: result = <LazyFrame expression>.
- Allowed libraries: pl (Polars) only; no other modules.
- Avoid division by zero; prefer safe casts and coalesce for null math.
- No markdown fences, comments, or prose—just Python code.
- String ops: cast to Utf8 before str functions.
- Filters must use boolean expressions; use bitwise &, |, ~ (not Python and/or/not).
- Never rely on truthiness of DataFrame/LazyFrame; never wrap LazyFrame in if/while.
- Do not call collect, sink, write, save, or any IO. Do not change schema unless asked.
""".strip()

    user_prompt = f"""
Schema:
{chr(10).join(schema_lines)}

Sample rows (truncated):
{samples}

User question:
{question}

Output format (Python code only):
result = lf.<transformations>.select(...).filter(...).group_by(...).agg(...)  # as needed
""".strip()

    return system_prompt, user_prompt
