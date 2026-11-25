"""Groq chat model factory + shared prompt helpers."""
from __future__ import annotations

from functools import lru_cache

import config


class LLMConfigError(RuntimeError):
    """Raised when the LLM cannot be configured (e.g. missing API key)."""


@lru_cache(maxsize=1)
def get_llm():
    key = config.GROQ_API_KEY
    if not key or key.lower().startswith(("your", "gsk_xxx")):
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Open the .env file in the project folder, paste your "
            "key after GROQ_API_KEY= (get one free at https://console.groq.com/keys), "
            "then restart the app."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=config.GROQ_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=key,
        max_retries=3,
        timeout=120,
    )


def _is_tool_use_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "tool_use_failed" in text or "did not call a tool" in text or "failed to call a function" in text


def structured(schema, attempts: int = 3):
    """``llm.with_structured_output(schema)`` via tool calling, retried when the model answers
    in plain text instead of calling the tool (Groq 400 'tool_use_failed')."""
    from langchain_core.runnables import RunnableLambda

    runnable = get_llm().with_structured_output(schema, method="function_calling")

    def invoke_with_retry(prompt_value):
        last: BaseException | None = None
        for _ in range(max(1, attempts)):
            try:
                return runnable.invoke(prompt_value)
            except Exception as exc:  # noqa: BLE001
                if not _is_tool_use_failure(exc):
                    raise
                last = exc
        raise last  # type: ignore[misc]

    return RunnableLambda(invoke_with_retry, name=f"structured_{schema.__name__}")


GRADE_LEVEL_GUIDE = {
    1: "Grade 1 (age 6-7): concrete objects, pictures, oral answers, numbers up to 20, 1-step instructions, very short sentences.",
    2: "Grade 2 (age 7-8): objects and drawings, numbers up to 100, simple written answers, 1-2 step instructions.",
    3: "Grade 3 (age 8-9): drawings plus symbols, short written answers, simple word problems.",
    4: "Grade 4 (age 9-10): symbols and number lines, 2-step word problems, explain in one sentence.",
    5: "Grade 5 (age 10-11): multi-step problems, written reasoning, real-life applications, can act as peer helpers.",
}


def grade_guide(grades: list[int]) -> str:
    return "\n".join(f"- {GRADE_LEVEL_GUIDE.get(g, f'Grade {g}')}" for g in grades)


def language_instruction(language: str) -> str:
    if language == "Kannada":
        return (
            "OUTPUT LANGUAGE: Write every human-readable value in simple Kannada using Kannada "
            "script (ಕನ್ನಡ). Keep JSON field names in English. Use digits 0-9 and fraction "
            "notation like 1/2. Source labels like [S1] stay unchanged."
        )
    if language == "Hindi":
        return (
            "OUTPUT LANGUAGE: Write every human-readable value in simple Hindi using Devanagari "
            "script (हिन्दी). Keep JSON field names in English. Use digits 0-9 and fraction "
            "notation like 1/2. Source labels like [S1] stay unchanged."
        )
    return "OUTPUT LANGUAGE: Simple, clear English suitable for rural primary classrooms."


def friendly_llm_error(exc: Exception) -> str:
    """Turn provider exceptions into a message a teacher/demo audience can understand."""
    if isinstance(exc, LLMConfigError):
        return str(exc)
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "401" in low or "invalid_api_key" in low or "authentication" in low:
        return "Groq rejected the API key. Check GROQ_API_KEY in .env and restart the app."
    if "429" in low or "rate limit" in low or "rate_limit" in low:
        return ("Groq rate limit reached. Wait about a minute and try again, or set "
                "MAX_CONCURRENCY=1 in .env.")
    if "model" in low and ("not found" in low or "decommissioned" in low or "404" in low):
        return f"The Groq model '{config.GROQ_MODEL}' is unavailable. Change GROQ_MODEL in .env."
    if "timeout" in low or "connect" in low:
        return "Could not reach Groq (network timeout). Check your internet connection."
    return f"The AI model returned an error: {text[:300]}"
