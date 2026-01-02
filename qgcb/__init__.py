"""Core package for the Metaculus proto-question generator."""

from qgcb.config import DEFAULT_JUDGE_MODEL, DEFAULT_MAIN_MODEL
from qgcb.models import JudgeKeepResult, ProtoQuestion, model_to_dict
from qgcb.openrouter import call_openrouter_raw, call_openrouter_structured, get_openrouter_key
from qgcb.pipeline import (
    build_kept_questions_payload,
    derive_seed_from_question,
    enforce_type_distribution,
    find_resolution_sources_for_prompt,
    generate_initial_questions,
    generate_resolution_card,
    rebalance_final_entries,
    rebalance_question_types,
    judge_initial_questions,
    judge_one_question_keep,
    mock_proto_questions,
    mutate_seed_prompt,
    normalize_question_entry,
    run_kept_questions_llm_hook,
    serialize_questions_to_csv,
    select_top_k,
)
from qgcb import prompts

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MAIN_MODEL",
    "JudgeKeepResult",
    "ProtoQuestion",
    "call_openrouter_raw",
    "call_openrouter_structured",
    "build_kept_questions_payload",
    "derive_seed_from_question",
    "enforce_type_distribution",
    "find_resolution_sources_for_prompt",
    "generate_initial_questions",
    "generate_resolution_card",
    "rebalance_final_entries",
    "rebalance_question_types",
    "get_openrouter_key",
    "judge_initial_questions",
    "judge_one_question_keep",
    "model_to_dict",
    "mock_proto_questions",
    "mutate_seed_prompt",
    "prompts",
    "normalize_question_entry",
    "run_kept_questions_llm_hook",
    "serialize_questions_to_csv",
    "select_top_k",
]
