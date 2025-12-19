"""Pydantic models and parsing helpers used across the pipeline."""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError


class ProtoQuestion(BaseModel):
    role: str = "VARIANT"
    angle: str = ""
    title: str
    question: str
    type: str = ""
    inbound_outcome_count: Optional[int] = None
    options: str = ""
    group_variable: str = ""
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    zero_point: Optional[float] = None
    open_lower_bound: Optional[bool] = None
    open_upper_bound: Optional[bool] = None
    unit: str = ""
    candidate_source: str = Field(default="", alias="candidate_source")
    rating: str = ""
    rating_rationale: str = ""
    raw_block: str = ""


class JudgeKeepResult(BaseModel):
    """Structured representation of the judge's single-line output."""

    keep: int = 0
    verdict: str = ""
    verdict_rationale: str = ""
    resolvability: int = 0
    info: int = 0
    decision_impact: float = 0.0
    voi: float = 0.0
    minutes_to_resolve: float = 0.0
    rationale: str = ""
    raw_line: str = ""


def model_to_dict(m: BaseModel) -> Dict[str, Any]:
    """Pydantic v1/v2 compatible conversion to dict."""

    if hasattr(m, "model_dump"):
        return m.model_dump()
    return m.dict()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_proto_questions_from_text(text: str) -> List[ProtoQuestion]:
    """
    Parse QUESTION i / Role / Title / Question / Angle / Candidate-source blocks
    into a list of ProtoQuestion objects.
    """

    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    questions: List[ProtoQuestion] = []
    current: Optional[Dict[str, Any]] = None
    question_lines: List[str] = []
    block_lines: List[str] = []
    capturing_question = False

    header_patterns = [
        re.compile(r"^QUESTION\s+\d+", re.IGNORECASE),
        re.compile(r"^Q\s*\d+\b", re.IGNORECASE),
    ]

    def parse_int(val: str) -> Optional[int]:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def parse_float(val: str) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def parse_bool(val: str) -> Optional[bool]:
        if val is None:
            return None
        lowered = val.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
        return None

    def is_question_header(line: str) -> bool:
        normalized = line.lstrip("#*- \t").strip()
        for pat in header_patterns:
            if pat.match(normalized):
                return True
        return False

    def finalize_question():
        nonlocal question_lines, capturing_question
        if current is None:
            return
        if capturing_question:
            # Remove stray scoring/meta-evaluation lines that occasionally leak from models
            cleaned_lines = []
            for ln in question_lines:
                lower_ln = ln.lower()
                if lower_ln.startswith("resolvability:"):
                    continue
                if lower_ln.startswith("information value") or lower_ln.startswith("info value"):
                    continue
                if lower_ln.startswith("decision impact"):
                    continue
                if lower_ln.startswith("voi:"):
                    continue
                if lower_ln.startswith("minutes to resolve"):
                    continue
                cleaned_lines.append(ln)

            joined = "\n".join([ln.strip() for ln in cleaned_lines if ln.strip() != ""])
            current["question"] = joined
        question_lines = []
        capturing_question = False

    def push_current():
        nonlocal current, block_lines
        if not current:
            return
        if capturing_question:
            finalize_question()
        raw_block_val = "\n".join([ln for ln in block_lines if str(ln).strip()]).strip()
        if raw_block_val:
            current["raw_block"] = raw_block_val
        # If the model forgot to include the question body, fall back to the title to keep the block usable.
        if not current.get("question") and current.get("title"):
            current["question"] = f"Title: {current['title']}"
        # Enforce a default rating if missing to keep downstream displays populated
        rating_val = (current or {}).get("rating", "").strip()
        if rating_val.upper() not in {"PUBLISHABLE", "SOFT REJECT", "HARD REJECT"}:
            current["rating"] = "Hard Reject"
            current["rating_rationale"] = current.get("rating_rationale", "").strip() or (
                "Rating missing from generation; defaulted to Hard Reject for safety."
            )
        if current.get("title") and current.get("question"):
            try:
                q = ProtoQuestion(**current)
                questions.append(q)
            except ValidationError:
                pass
        current = None
        block_lines = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if is_question_header(line):
            push_current()
            current = {
                "role": "VARIANT",
                "angle": "",
                "title": "",
                "question": "",
                "type": "",
                "inbound_outcome_count": None,
                "options": "",
                "group_variable": "",
                "range_min": None,
                "range_max": None,
                "zero_point": None,
                "open_lower_bound": None,
                "open_upper_bound": None,
                "unit": "",
                "candidate_source": "",
                "rating": "",
                "rating_rationale": "",
            }
            block_lines = [raw]
            continue
        if current is None and line.lower().startswith("title:"):
            current = {
                "role": "VARIANT",
                "angle": "",
                "title": line.split(":", 1)[1].strip(),
                "question": "",
                "type": "",
                "inbound_outcome_count": None,
                "options": "",
                "group_variable": "",
                "range_min": None,
                "range_max": None,
                "zero_point": None,
                "open_lower_bound": None,
                "open_upper_bound": None,
                "unit": "",
                "candidate_source": "",
                "rating": "",
                "rating_rationale": "",
            }
            block_lines = [raw]
            continue
        if current is None and line.lower().startswith("title:"):
            current = {
                "role": "VARIANT",
                "angle": "",
                "title": line.split(":", 1)[1].strip(),
                "question": "",
                "type": "",
                "inbound_outcome_count": None,
                "options": "",
                "group_variable": "",
                "range_min": None,
                "range_max": None,
                "zero_point": None,
                "open_lower_bound": None,
                "open_upper_bound": None,
                "unit": "",
                "candidate_source": "",
                "rating": "",
                "rating_rationale": "",
            }
            continue
        if current is None:
            continue
        block_lines.append(raw)
        lower = line.lower()

        # If we are capturing the Question block and encounter a new section label, finalize the question first
        if capturing_question and (
            lower.startswith("angle:")
            or lower.startswith("candidate-source:")
            or lower.startswith("role:")
            or lower.startswith("type:")
            or lower.startswith("inbound-outcome-count:")
            or lower.startswith("options:")
            or lower.startswith("group-variable:")
            or lower.startswith("range-min:")
            or lower.startswith("range-max:")
            or lower.startswith("zero-point:")
            or lower.startswith("open-lower-bound:")
            or lower.startswith("open-upper-bound:")
            or lower.startswith("unit:")
        ):
            finalize_question()

        if lower.startswith("role:"):
            val = line.split(":", 1)[1].strip().upper()
            if val not in {"CORE", "VARIANT"}:
                val = "VARIANT"
            current["role"] = val
        elif lower.startswith("title:"):
            current["title"] = line.split(":", 1)[1].strip()
        elif lower.startswith("question:"):
            finalize_question()
            capturing_question = True
            question_lines.append(line.split(":", 1)[1].strip())
        elif capturing_question:
            if lower.startswith("rating:"):
                finalize_question()
                current["rating"] = line.split(":", 1)[1].strip()
                continue
            elif lower.startswith("rationale:"):
                finalize_question()
                current["rating_rationale"] = line.split(":", 1)[1].strip()
                continue
            question_lines.append(line)
        elif lower.startswith("rating:"):
            current["rating"] = line.split(":", 1)[1].strip()
        elif lower.startswith("rationale:"):
            current["rating_rationale"] = line.split(":", 1)[1].strip()
        elif lower.startswith("angle:"):
            current["angle"] = line.split(":", 1)[1].strip()
        elif lower.startswith("type:"):
            current["type"] = line.split(":", 1)[1].strip()
        elif lower.startswith("inbound-outcome-count:"):
            val = parse_int(line.split(":", 1)[1].strip())
            if val is not None:
                current["inbound_outcome_count"] = val
        elif lower.startswith("options:"):
            current["options"] = line.split(":", 1)[1].strip()
        elif lower.startswith("group-variable:"):
            current["group_variable"] = line.split(":", 1)[1].strip()
        elif lower.startswith("range-min:"):
            val = parse_float(line.split(":", 1)[1].strip())
            if val is not None:
                current["range_min"] = val
        elif lower.startswith("range-max:"):
            val = parse_float(line.split(":", 1)[1].strip())
            if val is not None:
                current["range_max"] = val
        elif lower.startswith("zero-point:"):
            val = parse_float(line.split(":", 1)[1].strip())
            if val is not None:
                current["zero_point"] = val
        elif lower.startswith("open-lower-bound:"):
            val = parse_bool(line.split(":", 1)[1].strip())
            if val is not None:
                current["open_lower_bound"] = val
        elif lower.startswith("open-upper-bound:"):
            val = parse_bool(line.split(":", 1)[1].strip())
            if val is not None:
                current["open_upper_bound"] = val
        elif lower.startswith("unit:"):
            current["unit"] = line.split(":", 1)[1].strip()
        elif lower.startswith("candidate-source:"):
            current["candidate_source"] = line.split(":", 1)[1].strip()

    push_current()
    return questions


def parse_judge_keep_line(line: str) -> JudgeKeepResult:
    """
    Parse:
      keep=0|1; rating=Publishable|Soft Reject|Hard Reject; resolvability=X; info=Y; decision_impact=D; voi=V; minutes_to_resolve=R; rationale=TEXT
    into a JudgeKeepResult.
    """

    line = line.strip()
    parts = [p.strip() for p in line.split(";") if p.strip()]
    mapping: Dict[str, str] = {}
    for seg in parts:
        if "=" not in seg:
            continue
        k, v = seg.split("=", 1)
        mapping[k.strip().lower()] = v.strip()

    def to_int(name: str, default: int = 0) -> int:
        try:
            return int(mapping.get(name, default))
        except Exception:
            return default

    def to_float(name: str, default: float = 0.0) -> float:
        try:
            return float(mapping.get(name, default))
        except Exception:
            return default

    def normalize_verdict(val: str) -> str:
        low = val.strip().lower()
        if low.startswith("pub"):
            return "Publishable"
        if low.startswith("soft"):
            return "Soft Reject"
        if low.startswith("hard"):
            return "Hard Reject"
        return ""

    keep = to_int("keep", 0)
    if keep not in (0, 1):
        keep = 0
    resolvability = to_int("resolvability", 0)
    info = to_int("info", 0)
    decision_impact = to_float("decision_impact", 0.0)
    voi = to_float("voi", 0.0)
    minutes_to_resolve = to_float("minutes_to_resolve", 0.0)

    rationale = mapping.get("rationale", "").replace(";", ",").strip()
    if len(rationale) > 300:
        rationale = rationale[:300]

    verdict_raw = mapping.get("rating") or mapping.get("verdict") or ""
    verdict = normalize_verdict(verdict_raw)
    verdict_rationale = mapping.get("rating_rationale") or mapping.get("verdict_rationale") or rationale
    verdict_rationale = (verdict_rationale or "").replace(";", ",").strip()
    if len(verdict_rationale) > 400:
        verdict_rationale = verdict_rationale[:400]

    if not verdict:
        verdict = "Publishable" if keep == 1 and resolvability >= 4 and info >= 3 else "Hard Reject"

    return JudgeKeepResult(
        keep=keep,
        verdict=verdict,
        verdict_rationale=verdict_rationale,
        resolvability=resolvability,
        info=info,
        decision_impact=decision_impact,
        voi=voi,
        minutes_to_resolve=minutes_to_resolve,
        rationale=rationale,
        raw_line=line,
    )


__all__ = [
    "JudgeKeepResult",
    "ProtoQuestion",
    "model_to_dict",
    "parse_judge_keep_line",
    "parse_proto_questions_from_text",
]
