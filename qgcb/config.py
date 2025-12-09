"""Shared configuration constants for the proto-question generator."""

import os

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_ENV = os.environ.get("OPENROUTER_MODEL", "").strip()

REFERER = "https://localhost"
TITLE = "Metaculus – Evolutionary Proto Question Generator"
USER_AGENT = "metaculus-evo-qgen/0.2"

# Primary and judge models (can be overridden via OPENROUTER_MODEL env or UI input)
DEFAULT_MAIN_MODEL = OPENROUTER_MODEL_ENV or "openai/gpt-5.1"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MAIN_MODEL",
    "OPENROUTER_MODEL_ENV",
    "OPENROUTER_URL",
    "REFERER",
    "TITLE",
    "USER_AGENT",
]
