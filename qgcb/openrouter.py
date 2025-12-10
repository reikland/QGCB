"""Helpers for OpenRouter API calls."""

import json as _json
import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from qgcb.config import DEFAULT_JUDGE_MODEL, DEFAULT_MAIN_MODEL, OPENROUTER_URL, REFERER, TITLE, USER_AGENT


def get_openrouter_key() -> str:
    """Fetch the OpenRouter API key from override, env or secrets."""

    try:
        v = st.session_state.get("OPENROUTER_API_KEY_OVERRIDE", "").strip()
    except Exception:
        v = ""

    if not v and hasattr(st, "secrets"):
        v = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
    if not v:
        v = os.environ.get("OPENROUTER_API_KEY", "").strip()

    return v


def ascii_safe(s: str) -> str:
    try:
        return s.encode("latin-1", "ignore").decode("latin-1")
    except Exception:
        return "".join(ch for ch in s if ord(ch) < 256)


def or_headers() -> Dict[str, str]:
    key = get_openrouter_key()
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": ascii_safe(REFERER),
        "X-Title": ascii_safe(TITLE),
        "User-Agent": ascii_safe(USER_AGENT),
    }


def call_openrouter_raw(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 2000,
    temperature: float = 0.4,
    retries: int = 3,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """Lightweight wrapper around OpenRouter chat completions."""

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
    }

    if response_format:
        payload["response_format"] = response_format

    last_error: Optional[Exception] = None

    for k in range(retries):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=or_headers(),
                json=payload,
                timeout=120,
            )
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", "2") or 2)
                time.sleep(min(retry_after, 10))
                continue

            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("No choices in OpenRouter response")

            content = choices[0].get("message", {}).get("content", "")
            return content or ""
        except Exception as e:
            last_error = e
            time.sleep(0.8 * (k + 1))

    raise RuntimeError(f"[openrouter] giving up after retries: {repr(last_error)}")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    if stripped.endswith("```"):
        stripped = stripped[: -3]
    return stripped.strip()


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def call_openrouter_structured(
    system_prompt: str,
    user_prompt: str,
    model: str,
    schema_hint: str,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    retries: int = 3,
) -> Dict[str, Any]:
    """Minimal structured output extraction with retries."""

    last_error: Optional[Exception] = None

    system_with_schema = (
        system_prompt.strip()
        + "\n\nYou MUST respond with **pure JSON only**, no markdown, no comments.\n"
        + f"Target JSON schema (informal): {schema_hint}\n"
        + "Do not wrap the JSON in backticks. Do not add any text before or after it."
    )

    for _ in range(retries):
        try:
            raw = call_openrouter_raw(
                messages=[
                    {"role": "system", "content": system_with_schema},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            candidates = [raw, _strip_code_fences(raw), _extract_json_block(raw)]
            last_exc: Optional[Exception] = None
            for cand in candidates:
                try:
                    data = _json.loads(cand)
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    last_exc = exc
                    continue
            else:
                raise last_exc or RuntimeError("No JSON candidate parsed")
            if isinstance(data, dict):
                return data
            return {"data": data}
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"[openrouter_structured] Invalid JSON after retries: {repr(last_error)}")


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MAIN_MODEL",
    "ascii_safe",
    "call_openrouter_raw",
    "call_openrouter_structured",
    "get_openrouter_key",
]
