"""Pydantic models and parsing helpers used across the pipeline."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError


class ProtoQuestion(BaseModel):
    role: str = "VARIANT"
    angle: str = ""
    title: str
    question: str
    candidate_source: str = Field(default="", alias="candidate_source")
    rating: str = ""
    rating_rationale: str = ""


class JudgeKeepResult(BaseModel):
    """Structured representation of the judge's single-line output."""

    keep: int = 0
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
    capturing_question = False

    def finalize_question():
        nonlocal question_lines, capturing_question
        if current is None:
            return
        if capturing_question:
            joined = "\n".join([ln.strip() for ln in question_lines if ln.strip() != ""])
            current["question"] = joined
        question_lines = []
        capturing_question = False

    def push_current():
        nonlocal current
        if not current:
            return
        if capturing_question:
            finalize_question()
        if current.get("title") and current.get("question"):
            try:
                q = ProtoQuestion(**current)
                questions.append(q)
            except ValidationError:
                pass
        current = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("QUESTION "):
            push_current()
            current = {
                "role": "VARIANT",
                "angle": "",
                "title": "",
                "question": "",
                "candidate_source": "",
                "rating": "",
                "rating_rationale": "",
            }
            continue
        if current is None:
            continue
        lower = line.lower()

        # If we are capturing the Question block and encounter a new section label, finalize the question first
        if capturing_question and (
            lower.startswith("angle:") or lower.startswith("candidate-source:") or lower.startswith("role:")
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
                current["rating"] = line.split(":", 1)[1].strip()
            elif lower.startswith("rationale:"):
                current["rating_rationale"] = line.split(":", 1)[1].strip()
            question_lines.append(line)
        elif lower.startswith("angle:"):
            current["angle"] = line.split(":", 1)[1].strip()
        elif lower.startswith("candidate-source:"):
            current["candidate_source"] = line.split(":", 1)[1].strip()

    push_current()
    return questions


def parse_judge_keep_line(line: str) -> JudgeKeepResult:
    """
    Parse:
      keep=0|1; resolvability=X; info=Y; decision_impact=D; voi=V; minutes_to_resolve=R; rationale=TEXT
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

    return JudgeKeepResult(
        keep=keep,
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
