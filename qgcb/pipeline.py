"""Pipeline helpers orchestrating the prompt/generation/judge steps."""
from typing import Any, Dict, List, Optional
import textwrap
import json as _json
import random
import csv
import io
import logging

from pydantic import ValidationError

from qgcb.models import (
    JudgeKeepResult,
    ProtoQuestion,
    parse_judge_keep_line,
    parse_proto_questions_from_text,
)
from qgcb.openrouter import call_openrouter_raw, call_openrouter_structured
from qgcb.prompts import (
    CATEGORIES,
    CATEGORIES_DISPLAY,
    GEN_SYS_INITIAL,
    GEN_USER_TMPL_INITIAL,
    JUDGE_SYS_KEEP,
    JUDGE_USER_TMPL_KEEP,
    PROMPT_MUTATOR_SYS,
    PROMPT_MUTATOR_USER_TMPL,
    RESOLUTION_CARD_SYS,
    RESOLUTION_CARD_USER_TMPL,
    SEED_DERIVER_SYS,
    SEED_DERIVER_USER_TMPL,
    SOURCE_SYS,
    SOURCE_USER_TMPL,
    TYPE_REBALANCER_SYS,
    TYPE_REBALANCER_USER_TMPL,
)


# ---------------------------------------------------------------------------
# Step X – Hook on kept / selected questions (pre-export batch)
# ---------------------------------------------------------------------------

KEPT_HOOK_SYS = textwrap.dedent(
    """
    You are a post-selection hook running on the FINAL, KEPT proto-questions
    that passed the judge. You receive a JSON payload that contains ALL kept
    questions and relevant metadata (seed, tags, horizon, prompt id, etc.).

    Your job is to acknowledge the batch and return JSON ONLY with:
    - a `summary` object (total questions, counts by type and category),
    - an `echo` array mirroring the incoming questions with the fields:
      id, title, question, type, category, role, parent_prompt_id,
      question_weight, rating, rating_rationale, candidate_source, horizon,
      tags, seed.

    Do NOT add natural language. Respond with pure JSON, no markdown.
    """
)

KEPT_HOOK_USER_TMPL = textwrap.dedent(
    """
    Kept questions payload (JSON):
    {payload_json}

    Respond with JSON only using the schema described in the system message.
    """
)


# ---------------------------------------------------------------------------
# Mock helpers (dry_run)
# ---------------------------------------------------------------------------

TYPE_DISTRIBUTION_TARGET = {
    "binary": 0.5,
    "numeric": 0.3,
    "multiple_choice": 0.2,
}

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _allocate_question_types(
    n: int, proportions: Dict[str, float] | None = None
) -> Dict[str, int]:
    proportions = proportions or TYPE_DISTRIBUTION_TARGET
    base = {key: int(n * pct) for key, pct in proportions.items()}
    remainder = n - sum(base.values())
    if remainder > 0:
        fractions = {
            key: (n * pct) - base[key]
            for key, pct in proportions.items()
        }
        for key, _ in sorted(fractions.items(), key=lambda item: item[1], reverse=True):
            if remainder <= 0:
                break
            base[key] += 1
            remainder -= 1
    return base


def mock_proto_questions(seed: str, n: int) -> List[ProtoQuestion]:
    out: List[ProtoQuestion] = []
    prefix = seed.strip().split("\n")[0][:60] or "Example topic"
    type_counts = _allocate_question_types(n)
    type_sequence: List[str] = (
        ["binary"] * type_counts["binary"]
        + ["numeric"] * type_counts["numeric"]
        + ["multiple_choice"] * type_counts["multiple_choice"]
    )
    if len(type_sequence) < n:
        type_sequence.extend(["binary"] * (n - len(type_sequence)))
    for i in range(n):
        role = "CORE" if i < 2 else "VARIANT"
        angle = "anchor question" if i < 2 else f"variant angle #{i}"
        q_type = type_sequence[i]
        category = random.choice(CATEGORIES)
        inbound_outcome_count = None
        options = ""
        group_variable = ""
        range_min = None
        range_max = None
        zero_point = None
        open_lower_bound = None
        open_upper_bound = None
        unit = ""
        question_weight = 1.0
        if q_type == "numeric":
            inbound_outcome_count = 200
            range_min = 0.0
            range_max = 100.0
            open_lower_bound = False
            open_upper_bound = False
            unit = "units"
        elif q_type == "multiple_choice":
            options = "Option A|Option B|Option C"
            group_variable = "category"
        out.append(
            ProtoQuestion(
                role=role,
                angle=angle,
                title=f"[MOCK] {prefix} – Q{i+1}",
                question=(
                    "Title: Mock formatted question"\
                    "\nResolution Criteria: This mock resolves if the placeholder event is reported by the specified date."\
                    "\nFine Print: Treat any missing data as non-occurrence."\
                    "\nRating: Publishable"\
                    "\nRationale: Dry-run placeholder for testing formatting."
                ),
                question_weight=question_weight,
                type=q_type,
                inbound_outcome_count=inbound_outcome_count,
                options=options,
                group_variable=group_variable,
                category=category,
                range_min=range_min,
                range_max=range_max,
                zero_point=zero_point,
                open_lower_bound=open_lower_bound,
                open_upper_bound=open_upper_bound,
                unit=unit,
                candidate_source="Mock dataset / World Bank / Reuters",
                rating="Publishable",
                rating_rationale="Dry-run placeholder for testing formatting.",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Step 0 – Derive seed prompt + tags + horizon from a raw question
# ---------------------------------------------------------------------------

def derive_seed_from_question(
    question_text: str,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {
            "seed": f"[MOCK SEED] {question_text.strip()}",
            "tags": ["mock", "forecasting", "general"],
            "horizon": "resolve by 2030-12-31 UTC",
            "raw_output": "",
        }

    user_prompt = SEED_DERIVER_USER_TMPL.format(question=question_text.strip())

    data = call_openrouter_structured(
        system_prompt=SEED_DERIVER_SYS,
        user_prompt=user_prompt,
        model=model,
        schema_hint='{"seed":"string","tags":["string"],"horizon":"string"}',
        max_tokens=600,
        temperature=0.3,
    )

    raw_output = _json.dumps(data, ensure_ascii=False, indent=2)
    seed = str(data.get("seed", "") or "").strip()
    tags_raw = data.get("tags", []) or []
    horizon = str(data.get("horizon", "") or "").strip()

    tags: List[str] = []
    for tag in tags_raw:
        try:
            cleaned = str(tag).strip().lower()
        except Exception:
            continue
        if cleaned:
            tags.append(cleaned)

    if not seed:
        seed = question_text.strip()
    if not tags:
        tags = ["general"]
    if not horizon:
        horizon = "resolve by 2030-12-31 UTC"

    return {
        "seed": seed,
        "tags": tags,
        "horizon": horizon,
        "raw_output": raw_output,
    }


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
        schema_hint='{"mutations":[{"prompt":"string","focus":"string","rationale":"string"}]}',
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
        schema_hint='{"sources":["string", "string"]}',
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

    # Keep the list focused and aligned with the generator expectation of two to three concrete sources
    sources = sources[:3]

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
    prompt_text: str = "",
    resolution_hints: str = "",
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
        type_counts = _allocate_question_types(need)

        user_prompt = GEN_USER_TMPL_INITIAL.format(
            n=need,
            n_binary=type_counts["binary"],
            n_numeric=type_counts["numeric"],
            n_multiple_choice=type_counts["multiple_choice"],
            seed=seed.strip(),
            prompt_text=prompt_text.strip() or "(see seed context above)",
            resolution_hints=resolution_hints.strip() or "(see seed context above)",
            tags=", ".join(tags) or "unspecified",
            horizon=horizon.strip() or "unspecified",
            categories_list=CATEGORIES_DISPLAY,
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

    distribution_check = enforce_type_distribution(questions)

    return {
        "questions": questions,
        "raw_output": raw_output,
        "n_parsed": len(questions),
        "n_requested": n,
        "attempts": attempts,
        "distribution_check": distribution_check,
    }


# ---------------------------------------------------------------------------
# Step C1.b – Type rebalance controller (post-generation)
# ---------------------------------------------------------------------------

def _type_counts(questions: List[ProtoQuestion]) -> Dict[str, int]:
    counts = {key: 0 for key in TYPE_DISTRIBUTION_TARGET}
    counts["unknown"] = 0
    for q in questions:
        t = (q.type or "").strip().lower()
        if t in counts:
            counts[t] += 1
        else:
            counts["unknown"] += 1
    return counts


def _type_proportions(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.get(k, 0) for k in TYPE_DISTRIBUTION_TARGET)
    if total <= 0:
        return {key: 0.0 for key in TYPE_DISTRIBUTION_TARGET}
    return {
        key: round(counts.get(key, 0) / total, 4)
        for key in TYPE_DISTRIBUTION_TARGET
    }


def enforce_type_distribution(
    questions: List[ProtoQuestion],
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """
    Recalculate distribution on generated questions and block if off target.

    Args:
        questions: Generated proto-questions.
        tolerance: Allowed absolute deviation per type (proportion). Defaults to
            0 for a strict match against the rounded target counts.

    Raises:
        ValueError: If the observed distribution diverges beyond tolerance.
    """

    counts = _type_counts(questions)
    proportions = _type_proportions(counts)
    n = sum(counts.get(k, 0) for k in TYPE_DISTRIBUTION_TARGET)
    target_counts = _allocate_question_types(n)

    allowed_diff = max(1, int(round(tolerance * n))) if tolerance > 0 else 0
    mismatches: Dict[str, int] = {}
    for t, expected_count in target_counts.items():
        observed = counts.get(t, 0)
        if abs(observed - expected_count) > allowed_diff:
            mismatches[t] = observed - expected_count

    if counts.get("unknown", 0):
        mismatches["unknown"] = counts["unknown"]

    if mismatches:
        msg = (
            "Type distribution deviates from target; aborting batch. "
            f"counts={counts}, proportions={proportions}, target_counts={target_counts}, "
            f"deviations={mismatches}, tolerance={tolerance} (allowed diff={allowed_diff})"
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        "Type distribution validated (counts=%s, proportions=%s, target_counts=%s)",
        counts,
        proportions,
        target_counts,
    )
    return {
        "counts": counts,
        "proportions": proportions,
        "target_counts": target_counts,
    }


def _format_questions_for_rebalancer(questions: List[ProtoQuestion]) -> str:
    lines: List[str] = []
    for idx, q in enumerate(questions, start=1):
        lines.append(
            textwrap.dedent(
                f"""
                {idx}. ID=g0-q{idx}; Title={q.title}
                Type={q.type}; Question={q.question}
                Options={q.options}; Group-variable={q.group_variable};
                Range=[{q.range_min}, {q.range_max}] (open_lower={q.open_lower_bound}, open_upper={q.open_upper_bound}); Unit={q.unit}; inbound_outcome_count={q.inbound_outcome_count}
                """.strip()
            )
        )
    return "\n".join(lines)


def rebalance_question_types(
    questions: List[ProtoQuestion],
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    n = len(questions)
    if n == 0:
        return {
            "questions": questions,
            "adjusted": False,
            "before_counts": {},
            "after_counts": {},
            "target_counts": {},
            "raw_output": "",
        }

    before_counts = _type_counts(questions)
    target_counts = _allocate_question_types(n)

    target_matches = all(
        before_counts.get(k, 0) == target_counts.get(k, 0) for k in target_counts
    ) and before_counts.get("unknown", 0) == 0

    if target_matches:
        return {
            "questions": questions,
            "adjusted": False,
            "before_counts": before_counts,
            "after_counts": before_counts,
            "target_counts": target_counts,
            "raw_output": "",
        }

    if dry_run:
        return {
            "questions": questions,
            "adjusted": False,
            "before_counts": before_counts,
            "after_counts": before_counts,
            "target_counts": target_counts,
            "raw_output": "",
        }

    questions_block = _format_questions_for_rebalancer(questions)
    user_prompt = TYPE_REBALANCER_USER_TMPL.format(
        current_counts=before_counts,
        target_counts=target_counts,
        n_questions=n,
        questions_block=questions_block,
    )

    data = call_openrouter_structured(
        system_prompt=TYPE_REBALANCER_SYS,
        user_prompt=user_prompt,
        model=model,
        schema_hint='{"questions":[{"index":1,"type":"binary"}],"notes":"string"}',
        max_tokens=2200,
        temperature=0.2,
    )

    raw_output = _json.dumps(data, ensure_ascii=False, indent=2)

    updated_questions: List[ProtoQuestion] = questions.copy()
    adjustments = False

    for item in data.get("questions", []) or []:
        try:
            idx = int(item.get("index", 0)) - 1
        except Exception:
            continue
        if idx < 0 or idx >= len(questions):
            continue

        base = questions[idx].dict()
        updates: Dict[str, Any] = {}
        if item.get("title"):
            updates["title"] = str(item.get("title")).strip()
        if item.get("question"):
            updates["question"] = str(item.get("question")).strip()
        type_val = str(item.get("type", "")).strip().lower()
        if type_val:
            updates["type"] = type_val
        if "options" in item:
            updates["options"] = str(item.get("options") or "").strip()
        if "group_variable" in item:
            updates["group_variable"] = str(item.get("group_variable") or "").strip()
        if "range_min" in item:
            updates["range_min"] = item.get("range_min")
        if "range_max" in item:
            updates["range_max"] = item.get("range_max")
        if "open_lower_bound" in item:
            updates["open_lower_bound"] = item.get("open_lower_bound")
        if "open_upper_bound" in item:
            updates["open_upper_bound"] = item.get("open_upper_bound")
        if "unit" in item:
            updates["unit"] = str(item.get("unit") or "").strip()
        if "inbound_outcome_count" in item:
            updates["inbound_outcome_count"] = item.get("inbound_outcome_count")

        try:
            updated_questions[idx] = ProtoQuestion(**{**base, **updates})
            if type_val and type_val != (questions[idx].type or "").lower():
                adjustments = True
        except ValidationError:
            continue

    after_counts = _type_counts(updated_questions)
    adjustments = adjustments or after_counts != before_counts

    return {
        "questions": updated_questions,
        "adjusted": adjustments,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "target_counts": target_counts,
        "raw_output": raw_output,
    }


CSV_CORE_FIELDS = [
    "id",
    "title",
    "question",
    "type",
    "options",
    "category",
    "question_weight",
    "role",
    "parent_prompt_id",
    "inbound_outcome_count",
    "group_variable",
    "range_min",
    "range_max",
    "zero_point",
    "open_lower_bound",
    "open_upper_bound",
    "unit",
    "candidate_source",
    "rating",
    "rating_rationale",
    "horizon",
    "tags",
]


def _normalize_question_type(val: str) -> str:
    cleaned = (val or "").strip().lower().replace("-", "_")
    mapping = {
        "mc": "multiple_choice",
        "multiple choice": "multiple_choice",
        "multiple_choice": "multiple_choice",
        "numeric": "numeric",
        "number": "numeric",
        "binary": "binary",
        "yes_no": "binary",
    }
    return mapping.get(cleaned, cleaned)


def _normalize_options(options: Any) -> str:
    if options is None:
        return ""
    if isinstance(options, (list, tuple)):
        return "|".join(str(opt).strip() for opt in options if str(opt).strip())
    return "|".join(
        seg.strip() for seg in str(options).split("|") if seg is not None and str(seg).strip()
    )


def _coerce_number(val: Any) -> Any:
    if val is None:
        return ""
    try:
        num = float(val)
        return int(num) if num.is_integer() else num
    except Exception:
        return ""


def normalize_question_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a proto-question dict into a CSV-friendly schema.
    """

    type_norm = _normalize_question_type(entry.get("type", ""))
    tags_val = entry.get("tags", [])
    normalized = {
        "id": str(entry.get("id", "")).strip(),
        "title": str(entry.get("title", "")).strip(),
        "question": str(entry.get("question", "")).strip(),
        "type": type_norm,
        "options": _normalize_options(entry.get("options", "")),
        "category": str(entry.get("category", "")).strip(),
        "question_weight": _coerce_number(entry.get("question_weight")),
        "role": str(entry.get("role", "")).strip(),
        "parent_prompt_id": str(entry.get("parent_prompt_id", "")).strip(),
        "inbound_outcome_count": _coerce_number(entry.get("inbound_outcome_count")),
        "group_variable": str(entry.get("group_variable", "")).strip(),
        "range_min": _coerce_number(entry.get("range_min")),
        "range_max": _coerce_number(entry.get("range_max")),
        "zero_point": _coerce_number(entry.get("zero_point")),
        "open_lower_bound": entry.get("open_lower_bound", ""),
        "open_upper_bound": entry.get("open_upper_bound", ""),
        "unit": str(entry.get("unit", "")).strip(),
        "candidate_source": str(entry.get("candidate_source", "")).strip(),
        "rating": str(entry.get("rating", "")).strip(),
        "rating_rationale": str(entry.get("rating_rationale", "")).strip(),
        "horizon": str(
            entry.get("horizon")
            or entry.get("resolution_horizon")
            or ""
        ).strip(),
        "tags": ", ".join(tags_val) if isinstance(tags_val, list) else str(tags_val or "").strip(),
    }

    if not normalized["question"] and normalized["title"]:
        normalized["question"] = f"Title: {normalized['title']}"

    return normalized


def serialize_questions_to_csv(
    rows: List[Dict[str, Any]], extra_fields: Optional[List[str]] = None
) -> str:
    """Normalize and serialize a list of question dicts to CSV."""

    extra_fields = extra_fields or []
    fieldnames = CSV_CORE_FIELDS.copy()
    for f in extra_fields:
        if f not in fieldnames:
            fieldnames.append(f)

    normalized_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        try:
            base = normalize_question_entry(row)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Skipping row %s due to normalization error: %s", idx, exc)
            continue

        for f in extra_fields:
            val = row.get(f, "")
            if isinstance(val, list):
                val = "; ".join(str(v).strip() for v in val if str(v).strip())
            base[f] = "" if val is None else val
        normalized_rows.append(base)

    if not normalized_rows:
        raise ValueError("No valid questions to serialize.")

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(normalized_rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Step D – Fiches de résolution pour questions conservées
# ---------------------------------------------------------------------------

def generate_resolution_card(
    question_entry: Dict[str, Any],
    seed: str,
    tags: List[str],
    horizon: str,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {
            "question_id": question_entry.get("id", "unknown"),
            "card": (
                "Title: Mock resolution card\n"
                "Resolution Criteria: This dry-run card explains how the question would resolve using the supplied sources.\n"
                "Fine Print: In a real run this would restate the safeguards and fallback behaviour in sentences.\n"
                "Resolution Sources: World Bank WDI and IMF WEO as primary references."
            ),
            "raw_output": "",
        }

    question_block = str(question_entry.get("question", "")).strip()
    user_prompt = RESOLUTION_CARD_USER_TMPL.format(
        seed=seed.strip(),
        tags=", ".join(tags) or "unspecified",
        horizon=horizon.strip() or "unspecified",
        question_id=question_entry.get("id", "unknown"),
        title=question_entry.get("title", "(missing title)"),
        question_block=question_block,
        candidate_source=question_entry.get("candidate_source", ""),
        rating=question_entry.get("rating", ""),
        rating_rationale=question_entry.get("rating_rationale", ""),
    )

    messages = [
        {"role": "system", "content": RESOLUTION_CARD_SYS},
        {"role": "user", "content": user_prompt},
    ]

    raw_output = call_openrouter_raw(
        messages=messages,
        model=model,
        max_tokens=800,
        temperature=0.2,
    )

    return {
        "question_id": question_entry.get("id", "unknown"),
        "card": raw_output.strip(),
        "raw_output": raw_output,
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
        verdict = "Publishable" if keep else "Hard Reject"
        rationale = "Mock judge (dry run)."
        raw_line = (
            f"keep={keep}; rating={verdict}; resolvability={resolvability}; info={info}; "
            f"decision_impact={decision_impact:.2f}; voi={voi:.2f}; "
            f"minutes_to_resolve={minutes_to_resolve:.2f}; rationale={rationale}"
        )
        return JudgeKeepResult(
            keep=keep,
            verdict=verdict,
            verdict_rationale=rationale,
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
    verdict_weight = {"Publishable": 2, "Soft Reject": 1, "Hard Reject": 0}

    def score_tuple(i: int):
        jr = judge_res[i]
        verdict_score = verdict_weight.get(jr.verdict, 0)
        return (verdict_score, jr.resolvability, jr.info, jr.keep)

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


def build_kept_questions_payload(
    kept_entries: List[Dict[str, Any]],
    seed: str,
    tags: List[str],
    horizon: str,
    generated_at_utc: str,
) -> Dict[str, Any]:
    """Assemble the JSON payload shared with the post-selection LLM hook."""

    questions: List[Dict[str, Any]] = []
    for entry in kept_entries:
        questions.append(
            {
                "id": entry.get("id", ""),
                "title": entry.get("title", ""),
                "question": entry.get("question", ""),
                "type": entry.get("type", ""),
                "category": entry.get("category", ""),
                "role": entry.get("role", ""),
                "parent_prompt_id": entry.get("parent_prompt_id", ""),
                "question_weight": entry.get("question_weight", 1.0),
                "rating": entry.get("rating", ""),
                "rating_rationale": entry.get("rating_rationale", ""),
                "candidate_source": entry.get("candidate_source", ""),
                "horizon": horizon,
                "tags": tags,
                "seed": seed,
            }
        )

    return {
        "seed": seed,
        "tags": tags,
        "horizon": horizon,
        "generated_at_utc": generated_at_utc,
        "n_kept": len(questions),
        "questions": questions,
    }


def run_kept_questions_llm_hook(
    payload: Dict[str, Any],
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Trigger an LLM call on the aggregated kept-question payload.

    The hook returns a minimal summary plus an echo of the questions to
    make pre-export auditing explicit. In dry_run mode the function
    returns a stubbed response without calling OpenRouter.
    """

    if dry_run:
        type_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}
        for q in payload.get("questions", []):
            t = str(q.get("type", "")).strip().lower() or "unknown"
            c = str(q.get("category", "")).strip().lower() or "uncategorized"
            type_counts[t] = type_counts.get(t, 0) + 1
            category_counts[c] = category_counts.get(c, 0) + 1
        return {
            "status": "dry_run",
            "summary": {
                "total": len(payload.get("questions", [])),
                "by_type": type_counts,
                "by_category": category_counts,
            },
            "echo": payload.get("questions", []),
            "raw_output": "",
        }

    payload_json = _json.dumps(payload, ensure_ascii=False, indent=2)
    data = call_openrouter_structured(
        system_prompt=KEPT_HOOK_SYS,
        user_prompt=KEPT_HOOK_USER_TMPL.format(payload_json=payload_json),
        model=model,
        schema_hint='{"summary":{"total":0,"by_type":{"string":0},"by_category":{"string":0}},"echo":[{"id":"string"}]}',
        max_tokens=1200,
        temperature=0.0,
    )

    raw_output = _json.dumps(data, ensure_ascii=False, indent=2)
    return {"raw_output": raw_output, "data": data, "request_payload": payload}


__all__ = [
    "derive_seed_from_question",
    "find_resolution_sources_for_prompt",
    "generate_initial_questions",
    "enforce_type_distribution",
    "rebalance_question_types",
    "generate_resolution_card",
    "build_kept_questions_payload",
    "run_kept_questions_llm_hook",
    "judge_initial_questions",
    "judge_one_question_keep",
    "mock_proto_questions",
    "mutate_seed_prompt",
    "normalize_question_entry",
    "serialize_questions_to_csv",
    "select_top_k",
]
