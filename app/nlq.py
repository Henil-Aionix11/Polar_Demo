import logging
import re
from typing import Optional, Tuple

import openai
import polars as pl

from .config import get_settings
from .prompt_builder import build_expr_prompts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _extract_code(text: str) -> str:
    """
    Extract Python code block or raw code from the model response.
    """
    fence = re.search(r"```python(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    fence_any = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if fence_any:
        return fence_any.group(1).strip()
    return text.strip()


def _sanitize_expr_code(code: str) -> str:
    """
    Basic guardrails to prevent dangerous operations in exec.
    """
    lowered = code.lower()
    banned = [
        "import ",
        "__",
        "os.",
        "sys.",
        "subprocess",
        "eval(",
        "exec(",
        "open(",
        "path(",
        "shutil",
        "pickle",
        "boto3",
        "pandas",
    ]
    if any(tok in lowered for tok in banned):
        raise ValueError("Expression contains banned tokens.")
    return code.strip()


def _execute_polars_expr(lf: pl.LazyFrame, code: str) -> pl.LazyFrame:
    """
    Execute a Polars LazyFrame expression safely-ish by restricting globals.
    Expects code to assign `result`.
    """
    sanitized = _sanitize_expr_code(code)
    globals_safe = {"pl": pl}
    locals_safe = {"lf": lf}
    exec(sanitized, globals_safe, locals_safe)
    result = locals_safe.get("result")
    if result is None:
        result = globals_safe.get("result")
    if isinstance(result, pl.DataFrame):
        result = result.lazy()
    if not isinstance(result, pl.LazyFrame):
        raise ValueError("Expression must assign a LazyFrame to `result`.")
    # Tiny validation
    result.limit(1).collect()
    return result


def should_persist(code: str) -> bool:
    """
    Heuristic: persist when the code appears to mutate data (with_columns/rename/drop).
    Simple filters/select-only expressions will not be persisted.
    """
    lowered = code.lower()
    mutation_markers = ["with_columns", "with_column", ".rename", ".drop", ".replace"]
    return any(tok in lowered for tok in mutation_markers)


def generate_sql(question: str, lf: pl.LazyFrame, schema: dict) -> str:
    """
    Deprecated: SQL generation removed in favor of Polars expression generation.
    """
    raise RuntimeError("SQL generation is disabled. Use generate_polars_expr instead.")


def generate_polars_expr(question: str, lf: pl.LazyFrame, schema: dict) -> Tuple[str, pl.LazyFrame]:
    """
    Generate and execute a Polars LazyFrame expression via OpenAI with retries.
    Returns the code string and the resulting LazyFrame.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = openai.OpenAI(api_key=settings.openai_api_key)
    system_prompt, user_prompt = build_expr_prompts(question, lf, schema)

    logger.info("NLQ (expr) question: %s", question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: Optional[str] = None

    for attempt in range(1, settings.max_sql_attempts + 1):
        candidate_code = ""
        completion = client.chat.completions.create(
            model=settings.model,
            messages=messages,
            temperature=0.1,
        )
        text = completion.choices[0].message.content or ""
        logger.info("NLQ expr raw response (attempt %s): %s", attempt, text.replace("\n", " "))
        try:
            candidate_code = _extract_code(text)
            expr_lf = _execute_polars_expr(lf, candidate_code)
            logger.info(
                "NLQ expr succeeded on attempt %s; code: %s",
                attempt,
                candidate_code.replace("\n", " "),
            )
            return candidate_code, expr_lf
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Expr attempt %s failed: %s | code: %s",
                attempt,
                last_error,
                candidate_code.replace("\n", " "),
            )
            if attempt >= settings.max_sql_attempts:
                break
            feedback_code = candidate_code or text
            messages.append({"role": "assistant", "content": feedback_code})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous code failed with error: {last_error}. "
                        "Return corrected Python code that assigns a Polars LazyFrame to `result` using the provided lf. "
                        "No imports, no IO, no markdown. Avoid Python and/or/not; use bitwise &, |, ~ on boolean expressions."
                    ),
                }
            )

    raise ValueError(f"Failed to generate valid Polars expression after retries. Last error: {last_error}")
