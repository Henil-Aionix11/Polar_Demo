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


def build_prompts(question: str, lf: pl.LazyFrame, schema: dict) -> Tuple[str, str]:
    """
    Build system/user prompts for NL -> Polars SQL.
    System prompt holds all rules; user prompt only carries context + question.
    """
    settings = get_settings()
    samples = sample_rows(lf, n=15)
    schema_lines = [f"- {k}: {v}" for k, v in schema.items()]

    system_prompt = f"""
You are a senior data engineer generating Polars SQL against a single table named self.
Rules:
- Use only the provided columns.
- No DDL, INSERT, UPDATE, DELETE; return a single SELECT query only.
- Avoid SELECT * on wide tables; project only needed columns.
- Prefer safe casts and COALESCE for null math; avoid division by zero.
- Do not add LIMIT; the backend will apply a preview limit when collecting results.
- One statement only; no explanations or prose.
- Return only the SQL string; no markdown fences, no comments, no prose.
""".strip()

    user_prompt = f"""
Schema:
{chr(10).join(schema_lines)}

Sample rows (truncated):
{samples}

User question:
{question}

Output format (SQL only, no markdown fences):
SELECT ... FROM self ...;
""".strip()

    return system_prompt, user_prompt

