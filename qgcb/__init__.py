"""Core package for the Metaculus proto-question generator."""

from qgcb.config import DEFAULT_JUDGE_MODEL, DEFAULT_MAIN_MODEL
from qgcb.models import JudgeKeepResult, ProtoQuestion, model_to_dict
from qgcb.openrouter import call_openrouter_raw, call_openrouter_structured, get_openrouter_key
from qgcb.pipeline import (
    find_resolution_sources_for_prompt,
    generate_initial_questions,
    generate_resolution_card,
    judge_initial_questions,
    judge_one_question_keep,
    mock_proto_questions,
    mutate_seed_prompt,
    select_top_k,
)
from qgcb import prompts
"""Core helpers for the Metaculus proto-question generator."""

from .core import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAIN_MODEL,
    JudgeKeepResult,
    ProtoQuestion,
    call_openrouter_raw,
    find_resolution_sources_for_prompt,
    generate_initial_questions,
    get_openrouter_key,
    judge_initial_questions,
    mutate_seed_prompt,
    select_top_k,
)

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MAIN_MODEL",
    "JudgeKeepResult",
    "ProtoQuestion",
    "call_openrouter_raw",
    "call_openrouter_structured",
    "find_resolution_sources_for_prompt",
    "generate_initial_questions",
    "generate_resolution_card",
    "get_openrouter_key",
    "judge_initial_questions",
    "judge_one_question_keep",
    "model_to_dict",
    "mock_proto_questions",
    "mutate_seed_prompt",
    "prompts",
    "mutate_seed_prompt",
    "select_top_k",
]
