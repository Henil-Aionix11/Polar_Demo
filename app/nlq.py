import logging
import re
from typing import Optional

import openai
import polars as pl

from .config import get_settings
from .prompt_builder import build_prompts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def sanitize_sql(sql: str) -> str:
    """
    Reject dangerous tokens and multiple statements.
    """
    lowered = sql.lower()
    # Disallow DDL/DML keywords only when they appear as standalone words.
    if re.search(r"\b(drop|alter|insert|update|delete|create)\b", lowered):
        raise ValueError("Query contains banned tokens.")
    # Strip trailing semicolons but reject embedded semicolons (multiple statements)
    stripped = sql.strip()
    if ";" in stripped[:-1]:
        raise ValueError("Multiple statements are not allowed.")
    return stripped.rstrip(";")


def generate_sql(question: str, lf: pl.LazyFrame, schema: dict) -> str:
    """
    Generate a Polars SQL string via OpenAI with guardrails.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = openai.OpenAI(api_key=settings.openai_api_key)
    system_prompt, user_prompt = build_prompts(question, lf, schema)

    logger.info("NLQ question: %s", question)
    completion = client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    text = completion.choices[0].message.content or ""
    print("text: ", text)
    # Extract SQL from possible formatting.
    sql = ""
    fence = re.search(r"```sql(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        sql = fence.group(1)
    else:
        sql_match = re.search(r"(select|with)\s.+", text, flags=re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql = sql_match.group(0)
    if not sql.strip():
        raise ValueError("Could not parse SQL from model response.")
    sql = sql.strip().strip(";")
    logger.info("NLQ generated SQL: %s", sql.replace("\n", " "))
    return sanitize_sql(sql)

