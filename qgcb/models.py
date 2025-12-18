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
    capturing_question = False

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
        nonlocal current
        if not current:
            return
        if capturing_question:
            finalize_question()
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
            lower.startswith("angle:")
            or lower.startswith("candidate-source:")
            or lower.startswith("role:")
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
