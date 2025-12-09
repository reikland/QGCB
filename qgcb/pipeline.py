"""Pipeline helpers orchestrating the prompt/generation/judge steps."""
from typing import Any, Dict, List, Optional
import json as _json
import random

from qgcb.models import (
    JudgeKeepResult,
    ProtoQuestion,
    parse_judge_keep_line,
    parse_proto_questions_from_text,
)
from qgcb.openrouter import call_openrouter_raw, call_openrouter_structured
from qgcb.prompts import (
    GEN_SYS_INITIAL,
    GEN_USER_TMPL_INITIAL,
    JUDGE_SYS_KEEP,
    JUDGE_USER_TMPL_KEEP,
    PROMPT_MUTATOR_SYS,
    PROMPT_MUTATOR_USER_TMPL,
    SOURCE_SYS,
    SOURCE_USER_TMPL,
)


# ---------------------------------------------------------------------------
# Mock helpers (dry_run)
# ---------------------------------------------------------------------------

def mock_proto_questions(seed: str, n: int) -> List[ProtoQuestion]:
    out: List[ProtoQuestion] = []
    prefix = seed.strip().split("\n")[0][:60] or "Example topic"
    for i in range(n):
        role = "CORE" if i < 2 else "VARIANT"
        angle = "anchor question" if i < 2 else f"variant angle #{i}"
        out.append(
            ProtoQuestion(
                role=role,
                angle=angle,
                title=f"[MOCK] {prefix} – Q{i+1}",
                question="Will the mocked event occur before 2035-12-31?",
                candidate_source="Mock dataset / World Bank / Reuters",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Step A – Prompt mutation
# ---------------------------------------------------------------------------

def mutate_seed_prompt(
    seed: str,
    tags: List[str],
    horizon: str,
    m: int,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if m <= 0:
        return {"mutations": [], "raw_output": ""}

    if dry_run:
        mutations = []
        for i in range(1, m + 1):
            mutations.append(
                {
                    "text": f"[MOCK MUTATION {i}] {seed[:60]} – more concrete angle {i}",
                    "focus": f"mock focus #{i}",
                    "rationale": "Dry-run mutation for testing.",
                }
            )
        return {"mutations": mutations, "raw_output": ""}

    user_prompt = PROMPT_MUTATOR_USER_TMPL.format(
        seed=seed.strip(),
        horizon=horizon.strip() or "unspecified",
        tags=", ".join(tags) or "unspecified",
        m=m,
    )

    data = call_openrouter_structured(
        system_prompt=PROMPT_MUTATOR_SYS,
        user_prompt=user_prompt,
        model=model,
        schema_hint='{"mutations":[{"prompt": string, "focus": string, "rationale": string}]}',
        max_tokens=1500,
        temperature=0.4,
    )

    raw_output = _json.dumps(data, ensure_ascii=False, indent=2)
    mutations_raw = data.get("mutations", []) or []

    mutations: List[Dict[str, str]] = []
    for item in mutations_raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("prompt") or item.get("text") or "").strip()
        if not text:
            continue
        focus = str(item.get("focus", "") or "").strip()
        rationale = str(item.get("rationale", "") or "").strip()
        mutations.append(
            {
                "text": text,
                "focus": focus,
                "rationale": rationale,
            }
        )
        if len(mutations) >= m:
            break

    return {
        "mutations": mutations,
        "raw_output": raw_output,
    }


# ---------------------------------------------------------------------------
# Step B – Resolution sources per prompt
# ---------------------------------------------------------------------------

def find_resolution_sources_for_prompt(
    prompt_text: str,
    tags: List[str],
    horizon: str,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        sources = [
            "World Bank – World Development Indicators (WDI)",
            "IMF – World Economic Outlook (WEO) database",
            "Reuters / Bloomberg news wires",
        ]
        return {"sources": sources, "raw_output": ""}

    user_prompt = SOURCE_USER_TMPL.format(
        prompt_text=prompt_text.strip(),
        horizon=horizon.strip() or "unspecified",
        tags=", ".join(tags) or "unspecified",
    )

    data = call_openrouter_structured(
        system_prompt=SOURCE_SYS,
        user_prompt=user_prompt,
        model=model,
        schema_hint='{"sources":[string, ...]}',
        max_tokens=1200,
        temperature=0.2,
    )

    raw_output = _json.dumps(data, ensure_ascii=False, indent=2)
    raw_sources = data.get("sources", []) or []

    sources: List[str] = []
    for s in raw_sources:
        try:
            line = str(s).strip()
        except Exception:
            continue
        if line:
            sources.append(line)

    if not sources:
        sources = ["Generic public statistics and major news wires (fallback)."]

    return {
        "sources": sources,
        "raw_output": raw_output,
    }


# ---------------------------------------------------------------------------
# Step C1 – Question generation
# ---------------------------------------------------------------------------

def generate_initial_questions(
    seed: str,
    tags: List[str],
    horizon: str,
    n: int,
    model: str,
    dry_run: bool = False,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    if dry_run:
        questions = mock_proto_questions(seed, n)
        return {
            "questions": questions,
            "raw_output": "",
            "n_parsed": len(questions),
            "n_requested": n,
            "attempts": 1,
        }

    all_questions: List[ProtoQuestion] = []
    raw_chunks: List[str] = []
    attempts = 0

    while len(all_questions) < n and attempts < max_attempts:
        attempts += 1
        need = n - len(all_questions)

        user_prompt = GEN_USER_TMPL_INITIAL.format(
            n=need,
            seed=seed.strip(),
            prompt_text="",
            resolution_hints="(see seed context above)",
            tags=", ".join(tags) or "unspecified",
            horizon=horizon.strip() or "unspecified",
        )

        raw = call_openrouter_raw(
            messages=[
                {"role": "system", "content": GEN_SYS_INITIAL},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            max_tokens=3500,
            temperature=0.5,
        )
        raw_chunks.append(raw)

        new_questions = parse_proto_questions_from_text(raw)
        if new_questions:
            all_questions.extend(new_questions)

    if not all_questions:
        raise RuntimeError("Generator returned no parsable proto-questions.")

    questions = all_questions[:n]
    raw_output = "\n\n-----\n\n".join(raw_chunks)

    return {
        "questions": questions,
        "raw_output": raw_output,
        "n_parsed": len(questions),
        "n_requested": n,
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# Step C2 – Judge
# ---------------------------------------------------------------------------

def judge_one_question_keep(
    q: ProtoQuestion,
    seed: str,
    tags: List[str],
    horizon: str,
    judge_model: str,
    dry_run: bool = False,
) -> JudgeKeepResult:
    if dry_run:
        resolvability = random.randint(3, 5)
        info = random.randint(2, 5)
        keep = 1 if resolvability >= 4 and info >= 3 else 0
        decision_impact = random.random()
        voi = random.uniform(0.0, 5.0)
        minutes_to_resolve = random.uniform(1.0, 30.0)
        rationale = "Mock judge (dry run)."
        raw_line = (
            f"keep={keep}; resolvability={resolvability}; info={info}; "
            f"decision_impact={decision_impact:.2f}; voi={voi:.2f}; "
            f"minutes_to_resolve={minutes_to_resolve:.2f}; rationale={rationale}"
        )
        return JudgeKeepResult(
            keep=keep,
            resolvability=resolvability,
            info=info,
            decision_impact=decision_impact,
            voi=voi,
            minutes_to_resolve=minutes_to_resolve,
            rationale=rationale,
            raw_line=raw_line,
        )

    user_text = JUDGE_USER_TMPL_KEEP.format(
        seed=seed.strip(),
        horizon=horizon.strip(),
        tags=", ".join(tags) or "unspecified",
        title=q.title,
        question=q.question,
        source=q.candidate_source,
    )

    raw = call_openrouter_raw(
        messages=[
            {"role": "system", "content": JUDGE_SYS_KEEP},
            {"role": "user", "content": user_text},
        ],
        model=judge_model,
        max_tokens=256,
        temperature=0.0,
    )

    first_line = ""
    for ln in raw.splitlines():
        if "keep=" in ln:
            first_line = ln.strip()
            break
    if not first_line:
        raise RuntimeError(f"Judge returned no parsable line: {raw!r}")

    parsed = parse_judge_keep_line(first_line)
    return parsed


def judge_initial_questions(
    questions: List[ProtoQuestion],
    seed: str,
    tags: List[str],
    horizon: str,
    judge_model: str,
    dry_run: bool = False,
    contexts: Optional[List[str]] = None,
) -> List[JudgeKeepResult]:
    results: List[JudgeKeepResult] = []
    for i, q in enumerate(questions):
        ctx_seed = seed
        if contexts is not None and i < len(contexts) and contexts[i].strip():
            ctx_seed = contexts[i]
        res = judge_one_question_keep(
            q=q,
            seed=ctx_seed,
            tags=tags,
            horizon=horizon,
            judge_model=judge_model,
            dry_run=dry_run,
        )
        results.append(res)
    return results


def select_top_k(
    questions: List[ProtoQuestion],
    judge_res: List[JudgeKeepResult],
    k: int,
) -> Dict[str, Any]:
    n = len(questions)
    if n != len(judge_res):
        raise ValueError("questions and judge_res length mismatch")

    indices = list(range(n))

    def score_tuple(i: int):
        jr = judge_res[i]
        return (jr.resolvability, jr.info)

    kept_indices = [i for i in indices if judge_res[i].keep == 1]
    not_kept_indices = [i for i in indices if i not in kept_indices]

    kept_indices_sorted = sorted(kept_indices, key=score_tuple, reverse=True)
    not_kept_sorted = sorted(not_kept_indices, key=score_tuple, reverse=True)

    final_indices: List[int] = []

    for i in kept_indices_sorted:
        if len(final_indices) >= k:
            break
        final_indices.append(i)

    if len(final_indices) < k:
        for i in not_kept_sorted:
            if len(final_indices) >= k:
                break
            final_indices.append(i)

    final_indices = final_indices[: min(k, n)]

    keep_final_flags = [False] * n
    for idx in final_indices:
        keep_final_flags[idx] = True

    return {
        "final_indices": final_indices,
        "keep_final_flags": keep_final_flags,
        "n_kept_initial": len(kept_indices),
        "n_selected_final": len(final_indices),
    }


__all__ = [
    "find_resolution_sources_for_prompt",
    "generate_initial_questions",
    "judge_initial_questions",
    "judge_one_question_keep",
    "mock_proto_questions",
    "mutate_seed_prompt",
    "select_top_k",
]
