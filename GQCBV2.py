#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import time
import textwrap
import random
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st
import json as _json
from pydantic import BaseModel, Field, ValidationError

# ============================================================
# 1. CONFIG / CONSTANTS
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_ENV = os.environ.get("OPENROUTER_MODEL", "").strip()

REFERER = "https://localhost"
TITLE = "Metaculus – Evolutionary Proto Question Generator"

# Modèle principal (génération, mutations de prompts, sources)
DEFAULT_MAIN_MODEL = OPENROUTER_MODEL_ENV or "openai/gpt-5.1"
# Modèle léger (judge résolvabilité + info + "juge bis" des sources)
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"

if "evo_result" not in st.session_state:
    st.session_state["evo_result"] = None


# ============================================================
# 1.b – Pydantic models pour les objets LLM-facing
# ============================================================

class ProtoQuestion(BaseModel):
    """
    Structured representation of a proto forecasting question
    as produced/consumed by the LLM prompts.
    """
    role: str = "VARIANT"
    angle: str = ""
    title: str
    question: str
    candidate_source: str = Field(default="", alias="candidate_source")


class JudgeKeepResult(BaseModel):
    """
    Structured representation of the judge's single-line output.
    """
    keep: int = 0
    resolvability: int = 0
    info: int = 0
    decision_impact: float = 0.0
    voi: float = 0.0
    minutes_to_resolve: float = 0.0
    rationale: str = ""
    raw_line: str = ""


def model_to_dict(m: BaseModel) -> Dict[str, Any]:
    """
    Pydantic v1/v2 compatible conversion to dict.
    """
    if hasattr(m, "model_dump"):
        return m.model_dump()
    return m.dict()


# ============================================================
# 2. OPENROUTER HELPERS (+ mini "Structuring")
# ============================================================

def get_openrouter_key() -> str:
    """Récupère la clé OpenRouter de plusieurs sources possibles."""
    try:
        v = st.session_state.get("OPENROUTER_API_KEY_OVERRIDE", "").strip()
    except Exception:
        v = ""

    if not v:
        v = os.environ.get("OPENROUTER_API_KEY", "").strip()

    if not v:
        try:
            if "OPENROUTER_API_KEY" in st.secrets:
                v = str(st.secrets["OPENROUTER_API_KEY"]).strip()
        except Exception:
            pass

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
        "User-Agent": ascii_safe("metaculus-evo-qgen/0.2"),
    }


def call_openrouter_raw(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 2000,
    temperature: float = 0.4,
    retries: int = 3,
) -> str:
    """Appel brut à OpenRouter, en forçant un format court et strict côté modèle."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
    }

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


def _extract_json_block(text: str) -> str:
    """
    Extrait le bloc JSON principal d'une réponse (équivalent très light
    de ce que fait forecasting-tools.Structuring).
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
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
    """
    Mini équivalent de "structure_output" / "Structuring":

    - Force le modèle à produire du JSON uniquement.
    - Extrait le bloc JSON.
    - Fait un json.loads dessus.
    - Réessaie quelques fois en cas d'échec.
    """
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
            )
            candidate = _extract_json_block(raw)
            data = _json.loads(candidate)
            if isinstance(data, dict):
                return data
            # si ce n'est pas un dict, forcer dans un dict générique
            return {"data": data}
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"[openrouter_structured] Invalid JSON after retries: {repr(last_error)}")


# ============================================================
# 3. PROMPTS – GÉNÉRATION / JUDGE / MUTATION DE PROMPTS / SOURCES
# ============================================================

# ---------------------- MUTATION DE PROMPT (JSON) ----------------------

PROMPT_MUTATOR_SYS = """
You specialise and mutate high-level forecasting SEED PROMPTS.

Goal:
- Take a single general seed prompt about an uncertainty.
- Propose several more CONCRETE, RESOLVABLE prompts (mutations).
- Each mutation should push towards quantifiable, well-scoped questions
  that can be resolved from public data / official statistics / reputable news.

Hard constraints:
- Focus on resolvability: observable outcomes, clear metrics, specific populations, explicit time windows.
- Avoid extremely vague, purely qualitative or opinion-based prompts.
- Avoid already resolved topics (as of your knowledge) if obviously outdated.
- Think like a forecasting site designer who wants good, checkable questions.

You MUST output JSON only, with this informal schema:

{
  "mutations": [
    {
      "prompt": "string, concrete specialised prompt",
      "focus": "short phrase describing the angle / subtheme",
      "rationale": "1–2 sentences on how this increases resolvability"
    },
    ...
  ]
}
""".strip()

PROMPT_MUTATOR_USER_TMPL = textwrap.dedent(
    """
    Seed (central theme):
    {seed}

    Horizon / rough timeline:
    {horizon}

    Domain tags:
    {tags}

    Constraints for mutations:
    - You MUST generate at most {m} distinct mutated prompts.
    - Each mutated prompt must be clearly more specific and resolvable than the seed.
    - Prefer prompts that naturally lead to questions with:
      * explicit indicators or metrics,
      * explicit populations or geographies,
      * explicit time windows or deadlines.

    Output JSON ONLY, matching the schema described in the system message.
    """
)

# ---------------------- SOURCES / "JUGE BIS" (JSON) ----------------------

SOURCE_SYS = """
You design RESOLUTION SOURCES for forecasting questions.

Given a concrete forecasting prompt, you must propose families of public sources
that future forecasters will use to RESOLVE questions derived from this prompt.

Resolution sources can include:
- official national or international statistics (World Bank WDI, IMF WEO, Eurostat, etc.),
- regulatory or legal documents (Official Journal of the EU, US Federal Register, etc.),
- specific agencies (CDC, WHO, EMA, FDA, etc.),
- large, stable datasets (UN Comtrade, OECD databases, etc.),
- major reputable news wires (Reuters, AP, Bloomberg, etc.) if appropriate.

Hard constraints:
- Focus on sources that are realistically usable within 15 minutes by a careful forecaster,
  with web search and a general-purpose chatbot.
- Prefer sources that are stable, well-known, and easy to query.
- Avoid obscure, paywalled or extremely niche sources unless obviously necessary.

You MUST output JSON only, with this informal schema:

{
  "sources": [
    "short name – optional URL or dataset family",
    ...
  ]
}
""".strip()

SOURCE_USER_TMPL = textwrap.dedent(
    """
    Prompt for which we need public resolution sources:

    {prompt_text}

    Horizon:
    {horizon}

    Domain tags:
    {tags}

    Requirements:
    - Propose 3–7 concrete, realistic public source families.
    - These will be given to the question generator and MUST be usable
      to resolve most questions derived from this prompt.
    - Answer with JSON only, matching the schema in the system message.
    """
)

# ---------------------- GÉNÉRATION DE QUESTIONS (N questions) ----------------------

GEN_SYS_INITIAL = """
You generate CLUSTERS of proto forecasting questions for Metaculus.

Your output is parsed by a STRICT machine.
If you deviate from the required format, THE OUTPUT IS DISCARDED.

ABSOLUTE RULES
- You MUST strictly follow the template described below.
- You MUST produce EXACTLY N questions: not fewer, not more.
- You MUST NOT add explanations, comments, headings, or markdown.
- You MUST NOT include internal reasoning, rationale, or examples.
- Your VERY FIRST non-empty line MUST be: "QUESTION 1".
- Your LAST non-empty line MUST start with "Candidate-source:" for QUESTION N.
- No JSON. No code fences. No bullet lists. Plain text only.

CLUSTER BEHAVIOUR
- Interpret the (prompt + resolution hints) as describing ONE central theme for THIS cluster.
- Produce a coherent cluster of N related proto-questions.
- 1–2 questions should be broad "anchor" questions about the central theme (Role=CORE).
- The remaining questions should be narrower "variants" exploring different angles (Role=VARIANT).

Content constraints:
- Questions must be about an uncertain future or an as-yet-unobserved outcome.
- They must be resolvable from PUBLIC sources that match the provided resolution hints.
- Before emitting a question, briefly check whether it is likely ALREADY resolved; if so, DO NOT use that question.
- Target questions that a careful forecaster, with access to a general-purpose chatbot plus web search,
  could fully resolve in <= 15 minutes once the resolution date has passed.
- Avoid questions that require obscure, proprietary, or extremely hard-to-find sources.
- Avoid trivial questions whose probability is obviously ~0% or ~100%.
- Avoid questions that are already resolved.

STRICT FORMAT (LINE-BASED)
For each i = 1..N you output a block with these 5 lines:

QUESTION i
Role: CORE or VARIANT
Title: <short title, <= 100 characters, single line>
Question: <1–3 sentences, single line, ends with '?' or equivalent>
Angle: <short phrase capturing the angle within the cluster>
Candidate-source: <likely family of public resolution sources/datasets, single line>

Between blocks you MAY optionally have a single blank line.
You MUST NOT output anything else before, between, or after the blocks.
""".strip()

GEN_USER_TMPL_INITIAL = textwrap.dedent(
    """
    You must now generate a CLUSTER of proto forecasting questions.

    HARD CONSTRAINT:
    - N_questions = {n}.
    - You MUST output EXACTLY N_questions blocks, labelled QUESTION 1, QUESTION 2, ..., QUESTION {n}.
    - Any extra text or missing block makes the output INVALID.

    Root seed (global theme, do NOT restate it):
    {seed}

    Prompt used for THIS cluster:
    {prompt_text}

    Resolution hints for THIS cluster (you MUST adhere to these source families):
    {resolution_hints}

    Optional context:
    - Domain tags: {tags}
    - Horizon / rough timeline: {horizon}

    Cluster constraints:
    - 1–2 questions are broad anchor questions (Role=CORE).
    - Remaining questions (Role=VARIANT) explore distinct angles: geographies, actors, baselines vs tails,
      policy vs market, distributional effects, etc.
    - Before finalising a question, mentally outline how you would resolve it in practice using PUBLIC sources
      consistent with the resolution hints + a chatbot with web access; if you cannot, do not output it.

    Output ONLY the blocks in the EXACT format specified in the system message.
    Do NOT restate the instructions. Do NOT explain your choices.
    """
)

# ---------------------- JUDGE LIGHT (keep K parmi N) ----------------------

JUDGE_SYS_KEEP = """
You are a FAST, STRICT judge for proto forecasting questions, with an OBSESSION for resolvability.

Your ONLY task is to decide whether to KEEP or DISCARD ONE question, based on:
- resolvability from PUBLIC sources (this is the PRIMARY criterion),
- information value for forecasting and decision-making (secondary),
- practical solvability.

Practical solvability:
- Imagine a careful forecaster working with a general-purpose chatbot + web search.
- Once the resolution date has passed, that team should be able to fully resolve the question in <= 15 minutes.
- If you cannot see a concrete, realistic path to resolution within that constraint, resolvability is LOW.

You MUST output EXACTLY ONE LINE, with this format:

keep=0|1; resolvability=X; info=Y; decision_impact=D; voi=V; minutes_to_resolve=R; rationale=TEXT

Hard constraints:
- X and Y MUST be integers from 1 to 5.
- D MUST be a float between 0 and 1 (0 = no impact on decisions for an average global citizen, 1 = very high impact).
- V and R MUST be floats (V unbounded, higher = higher value of information; R >= 0, in minutes, lower = easier to resolve).
- rationale MUST be <= 200 characters and MUST NOT contain semicolons.
- The very first non-space characters of your reply MUST be "keep=".
- You MUST NOT add any other lines, JSON, markdown, or commentary.
- No bullet lists. No explanations before or after the line.
- If you are unsure, choose a reasonable guess and still follow the format.

Scoring hints:
- First, briefly check whether the question is LIKELY ALREADY RESOLVED as of your knowledge; if yes,
  set keep=0 and resolvability=0 or 1.
- resolvability: 1 = barely or not resolvable in practice; 5 = clearly resolvable from stable public sources
  with a simple, fast procedure.
- info: 1 = almost no useful information for real decisions; 5 = high value-of-information for policies,
  investment, planning, or safety.
- decision_impact: think of a random global citizen; 0 = knowing the answer would not change any behaviour,
  1 = knowing the answer would clearly change important behaviours or choices.
- voi: approximate value of information relative to important metrics (economic, risk, or parent-question
  uncertainty); higher = better.
- minutes_to_resolve: rough estimate of how long it would take a competent human or AI agent with web access
  to resolve the question once the resolution date has passed.

Decision rule (VERY STRICT):
- If resolvability <= 3, you MUST set keep=0, even if information value is high.
- keep=1 ONLY if:
  * the question is clearly resolvable from public sources,
  * a careful forecaster with a chatbot + web could actually resolve it in <= 15 minutes after the resolution date,
  * AND resolvability >= 4,
  * AND the information value is at least moderate (info >= 3).
- Otherwise keep=0.
""".strip()

JUDGE_USER_TMPL_KEEP = textwrap.dedent(
    """
    You are judging the following proto forecasting question.

    Context (root seed, mutated prompt, resolution hints):
    {seed}

    Horizon:
    {horizon}

    Domain tags (optional):
    {tags}

    Proto-question:
    Title: {title}
    Question: {question}
    Candidate-source: {source}

    Decision rule (SUPER STRICT on resolvability):
    - If resolvability <= 3, you MUST set keep=0.
    - keep=1 ONLY if:
      * clearly resolvable from PUBLIC sources,
      * practically solvable in <= 15 minutes with a chatbot + web after the resolution date,
      * resolvability >= 4,
      * info >= 3.

    Additional scoring dimensions (do NOT change the keep rule):
    - decision_impact: 0–1, impact on decisions for a random global citizen if the answer were known.
    - voi: float, overall value of information of this question; higher = better.
    - minutes_to_resolve: float, approximate minutes needed to resolve the question once the resolution date has passed;
      lower = better.

    Now output ONLY the single line:
    keep=0|1; resolvability=X; info=Y; decision_impact=D; voi=V; minutes_to_resolve=R; rationale=TEXT
    """
)


# ============================================================
# 4. PARSING HELPERS (returning Pydantic models)
# ============================================================

def parse_proto_questions_from_text(text: str) -> List[ProtoQuestion]:
    """
    Parse QUESTION i / Role / Title / Question / Angle / Candidate-source blocks
    into a list of ProtoQuestion objects.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    questions: List[ProtoQuestion] = []
    current: Optional[Dict[str, Any]] = None

    def push_current():
        nonlocal current
        if not current:
            return
        if current.get("title") and current.get("question"):
            try:
                q = ProtoQuestion(**current)
                questions.append(q)
            except ValidationError:
                # Ignore invalid blocks silently; downstream will see fewer questions
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
            }
            continue
        if current is None:
            continue
        lower = line.lower()
        if lower.startswith("role:"):
            val = line.split(":", 1)[1].strip().upper()
            if val not in {"CORE", "VARIANT"}:
                val = "VARIANT"
            current["role"] = val
        elif lower.startswith("title:"):
            current["title"] = line.split(":", 1)[1].strip()
        elif lower.startswith("question:"):
            current["question"] = line.split(":", 1)[1].strip()
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


# ============================================================
# 5. PIPELINE FUNCTIONS
# ============================================================

# ---------------------- MOCK HELPERS (dry_run) ----------------------

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


# ---------------------- Étape A – Mutation de prompt ----------------------

def mutate_seed_prompt(
    seed: str,
    tags: List[str],
    horizon: str,
    m: int,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Mutate the high-level seed into up to m more concrete, resolvable prompts.
    Returns:
      {
        "mutations": [{"text": str, "focus": str, "rationale": str}, ...],
        "raw_output": <raw JSON / model text>,
      }
    """
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


# ---------------------- Étape B – Recherche de sources (juge bis) ----------------------

def find_resolution_sources_for_prompt(
    prompt_text: str,
    tags: List[str],
    horizon: str,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    For a given prompt, ask a small LLM to propose concrete resolution source families.
    Returns:
      {
        "sources": [str, ...],
        "raw_output": <raw JSON / model text>,
      }
    """
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

    # sanité
    if not sources:
        sources = ["Generic public statistics and major news wires (fallback)."]

    return {
        "sources": sources,
        "raw_output": raw_output,
    }


# ---------------------- Étape C1 – Génération de questions ----------------------

def generate_initial_questions(
    seed: str,
    tags: List[str],
    horizon: str,
    n: int,
    model: str,
    dry_run: bool = False,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Generate N proto-questions (generation=0) as ProtoQuestion objects."""
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
            prompt_text="",           # seed already inclut le prompt précis + hints en amont
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


# ---------------------- Étape C2 – Judge (résolvabilité + info + critères sup.) ----------------------

def judge_one_question_keep(
    q: ProtoQuestion,
    seed: str,
    tags: List[str],
    horizon: str,
    judge_model: str,
    dry_run: bool = False,
) -> JudgeKeepResult:
    if dry_run:
        # mock strict-ish: favor resolvability
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
    """
    Judge all questions. If `contexts` is provided, it must be a list of
    same length as `questions`, and each entry is used as the "seed"
    context for that question (root seed + mutated prompt + sources).
    """
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
    """
    Selects K questions:
    - priority to keep=1,
    - sort by resolvability then info,
    - fill-up with keep=0 if necessary.
    """
    n = len(questions)
    if n != len(judge_res):
        raise ValueError("questions and judge_res length mismatch")

    indices = list(range(n))

    def score_tuple(i: int):
        jr = judge_res[i]
        # higher resolvability, then info
        return (jr.resolvability, jr.info)

    kept_indices = [i for i in indices if judge_res[i].keep == 1]
    not_kept_indices = [i for i in indices if i not in kept_indices]

    # tri décroissant (resolvability, info)
    kept_indices_sorted = sorted(kept_indices, key=score_tuple, reverse=True)
    not_kept_sorted = sorted(not_kept_indices, key=score_tuple, reverse=True)

    final_indices: List[int] = []

    # d'abord les keep=1
    for i in kept_indices_sorted:
        if len(final_indices) >= k:
            break
        final_indices.append(i)

    # si pas assez, compléter avec les meilleurs "keep=0"
    if len(final_indices) < k:
        for i in not_kept_sorted:
            if len(final_indices) >= k:
                break
            final_indices.append(i)

    # Au final, on ne garde que min(k, n)
    final_indices = final_indices[: min(k, n)]

    # Marquage keep_final
    keep_final_flags = [False] * n
    for idx in final_indices:
        keep_final_flags[idx] = True

    return {
        "final_indices": final_indices,
        "keep_final_flags": keep_final_flags,
        "n_kept_initial": len(kept_indices),
        "n_selected_final": len(final_indices),
    }


# ============================================================
# 6. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Metaculus – Evolutionary Proto Question Generator",
    page_icon="🧬",
    layout="wide",
)

st.title("Metaculus – Evolutionary Proto Question Generator")

st.markdown(
    """
New 3-stage pipeline (PROMPTS → SOURCES → QUESTIONS):

1. **Prompt mutations**: starting from your seed, the main model creates several more concrete,
   resolvable prompts (mutations).
2. **Resolution sources (judge-bis)**: for EACH prompt (seed + mutations), a light model proposes
   concrete public resolution sources (datasets, agencies, news wires, etc.).
3. **Question generation + strict judge**: the main model generates proto-questions around all prompts
   (respecting the resolution sources), then the judge filters them with an obsession for resolvability.

Formatting is strictly controlled (no JSON returned by templates for questions) and the final JSON
is built on the Python side only.
"""
)

# ---------------------- Sidebar ----------------------

with st.sidebar:
    st.header("OpenRouter configuration")

    dry_run = st.checkbox(
        "Dry run (no API calls, mock prompts/questions & scores)",
        value=False,
    )

    api_key_input = st.text_input(
        "OpenRouter API key",
        type="password",
        help="Key is only stored in this session.",
    )
    if api_key_input:
        st.session_state["OPENROUTER_API_KEY_OVERRIDE"] = api_key_input.strip()

    main_model_input = st.text_input(
        "Main model (mutations + sources + generation)",
        value=DEFAULT_MAIN_MODEL,
        help="Ex: openai/gpt-5.1, anthropic/claude-3.5-sonnet, etc.",
    )

    judge_model_input = st.text_input(
        "Judge model (light, resolvability + sources)",
        value=DEFAULT_JUDGE_MODEL,
        help="Ex: openai/gpt-4o-mini (fast, not costly).",
    )

    st.markdown("---")

    n_mutations = st.slider(
        "Number of mutated prompts (in addition to base seed)",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        help="How many specialised prompts to generate from the seed.",
    )

    n_initial = st.slider(
        "Total N proto-questions (across ALL prompts)",
        min_value=5,
        max_value=60,
        value=20,
        step=1,
    )

    k_keep = st.slider(
        "K questions kept by judge",
        min_value=1,
        max_value=n_initial,
        value=min(10, n_initial),
        step=1,
    )

current_key = get_openrouter_key()
if not current_key and not dry_run:
    st.warning(
        "No OPENROUTER_API_KEY detected. Enter it in the sidebar, "
        "or enable dry_run mode for local testing."
    )

# ---------------------- Main inputs ----------------------

st.subheader("Seed prompt")

seed = st.text_area(
    "Seed prompt (1–12 sentences, central theme)",
    height=180,
    placeholder=(
        "Describe the main uncertainty / topic.\n"
        "Example: 'I want questions about global diffusion of frontier AI systems in "
        "education and public administration by 2040, including inequalities, safety, and regulation.'"
    ),
)

tags_str = st.text_input(
    "Domain tags (comma-separated)",
    value="ai,policy,macro",
)

horizon = st.text_input(
    "Horizon / rough timeline",
    value="resolve by 2040-12-31 UTC",
)

run_button = st.button("Run full pipeline (PROMPTS → SOURCES → QUESTIONS)")

# ---------------------- Run pipeline ----------------------

if run_button:
    if not seed.strip():
        st.warning("Please provide a seed prompt.")
    elif not dry_run and not current_key:
        st.error("No OPENROUTER_API_KEY set and dry_run is disabled.")
    else:
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        main_model = (main_model_input or "").strip() or DEFAULT_MAIN_MODEL
        judge_model = (judge_model_input or "").strip() or DEFAULT_JUDGE_MODEL

        st.info(
            f"Using main model (mutations + sources + generation): `{main_model}`\n\n"
            f"Using judge model (strict resolvability): `{judge_model}`"
        )

        # A) Mutations de prompt
        with st.spinner("Step 1/3 – Mutating seed prompt into more concrete prompts..."):
            try:
                mut_res = mutate_seed_prompt(
                    seed=seed,
                    tags=tags,
                    horizon=horizon,
                    m=n_mutations,
                    model=main_model,
                    dry_run=dry_run,
                )
            except Exception as e:
                st.error(f"Prompt mutation error: {e}")
                mut_res = None

        if mut_res is None:
            st.session_state["evo_result"] = None
        else:
            mutations = mut_res.get("mutations", [])
            raw_mut_output = mut_res.get("raw_output", "")

            # Construire la liste de prompts (seed + mutations)
            prompt_entries: List[Dict[str, Any]] = []

            # prompt 0 = seed
            prompt_entries.append(
                {
                    "prompt_id": "p0",
                    "kind": "seed",
                    "text": seed.strip(),
                    "focus": "root seed",
                    "rationale": "Original seed prompt.",
                    "sources": [],
                    "sources_raw": "",
                }
            )

            for i, m in enumerate(mutations, start=1):
                prompt_entries.append(
                    {
                        "prompt_id": f"p{i}",
                        "kind": "mutation",
                        "text": m.get("text", "").strip(),
                        "focus": m.get("focus", "").strip(),
                        "rationale": m.get("rationale", "").strip(),
                        "sources": [],
                        "sources_raw": "",
                    }
                )

            # B) Recherche de sources pour chaque prompt
            with st.spinner("Step 2/3 – Finding concrete resolution sources for each prompt..."):
                for pe in prompt_entries:
                    try:
                        src_res = find_resolution_sources_for_prompt(
                            prompt_text=pe["text"],
                            tags=tags,
                            horizon=horizon,
                            model=judge_model,
                            dry_run=dry_run,
                        )
                    except Exception as e:
                        st.error(f"Source finder error for prompt {pe['prompt_id']}: {e}")
                        src_res = {"sources": ["(error: fallback generic sources)"], "raw_output": ""}

                    pe["sources"] = src_res.get("sources", []) or ["(no sources returned)"]
                    pe["sources_raw"] = src_res.get("raw_output", "")

            # C) Génération de questions pour l'ensemble des prompts (répartition N_total)
            with st.spinner("Step 3/3 – Generating proto-questions across all prompts and judging them..."):
                all_questions: List[ProtoQuestion] = []
                all_prompt_ids: List[str] = []
                raw_gen_chunks: List[str] = []

                P = len(prompt_entries)
                if P == 0:
                    st.error("No prompts available (seed + mutations).")
                    st.session_state["evo_result"] = None
                else:
                    base_per = n_initial // P
                    remainder = n_initial % P

                    for idx, pe in enumerate(prompt_entries):
                        n_for_this = base_per + (1 if idx < remainder else 0)
                        if n_for_this <= 0:
                            continue

                        hints_str = "\n".join(f"- {s}" for s in pe.get("sources", []))
                        seed_for_gen = (
                            f"ROOT SEED:\n{seed.strip()}\n\n"
                            f"PROMPT ({pe['prompt_id']} – {pe['kind']}):\n{pe['text']}\n\n"
                            f"RESOLUTION HINTS (families of PUBLIC sources you MUST stick to):\n{hints_str}"
                        )

                        try:
                            gen_res = generate_initial_questions(
                                seed=seed_for_gen,
                                tags=tags,
                                horizon=horizon,
                                n=n_for_this,
                                model=main_model,
                                dry_run=dry_run,
                            )
                        except Exception as e:
                            st.error(f"Generation error for prompt {pe['prompt_id']}: {e}")
                            continue

                        qs = gen_res.get("questions", []) or []
                        raw_out = gen_res.get("raw_output", "") or ""
                        if raw_out:
                            raw_gen_chunks.append(
                                f"===== PROMPT {pe['prompt_id']} ({pe['kind']}) =====\n{raw_out}"
                            )

                        all_questions.extend(qs)
                        all_prompt_ids.extend([pe["prompt_id"]] * len(qs))

                    if not all_questions:
                        st.error("No proto-questions were parsed. Check generator prompts.")
                        st.session_state["evo_result"] = None
                    else:
                        # Construire les contextes individuels pour le judge
                        prompt_by_id = {pe["prompt_id"]: pe for pe in prompt_entries}
                        judge_contexts: List[str] = []

                        for q_idx, q in enumerate(all_questions):
                            p_id = all_prompt_ids[q_idx]
                            pe = prompt_by_id.get(p_id)
                            if pe is None:
                                ctx = seed
                            else:
                                hints_str = "\n".join(pe.get("sources", []))
                                ctx = (
                                    f"ROOT SEED:\n{seed.strip()}\n\n"
                                    f"PROMPT ({p_id} – {pe['kind']}):\n{pe['text']}\n\n"
                                    f"RESOLUTION HINTS (STRICTLY respect these families of sources):\n{hints_str}"
                                )
                            judge_contexts.append(ctx)

                        # Judge strict (résolvabilité + info + critères sup.)
                        try:
                            judge_res0: List[JudgeKeepResult] = judge_initial_questions(
                                questions=all_questions,
                                seed=seed,
                                tags=tags,
                                horizon=horizon,
                                judge_model=judge_model,
                                dry_run=dry_run,
                                contexts=judge_contexts,
                            )
                        except Exception as e:
                            st.error(f"Judge error: {e}")
                            judge_res0 = None

                        if judge_res0 is None:
                            st.session_state["evo_result"] = None
                        else:
                            selection = select_top_k(
                                questions=all_questions,
                                judge_res=judge_res0,
                                k=k_keep,
                            )

                            final_indices = selection["final_indices"]
                            keep_final_flags = selection["keep_final_flags"]
                            n_kept_initial = selection["n_kept_initial"]
                            n_selected_final = selection["n_selected_final"]

                            st.info(
                                f"Judge initial keep=1 count: {n_kept_initial} / {len(all_questions)}; "
                                f"Selected (final K): {n_selected_final} (target K={k_keep})."
                            )

                            # Attribuer IDs et génération
                            initial_entries: List[Dict[str, Any]] = []
                            id_map: Dict[int, str] = {}  # index -> id

                            for idx_q, q in enumerate(all_questions):
                                q_id = f"g0-q{idx_q+1}"
                                id_map[idx_q] = q_id
                                jr = judge_res0[idx_q]
                                p_id = all_prompt_ids[idx_q]
                                initial_entries.append(
                                    {
                                        "id": q_id,
                                        "generation": 0,
                                        "parent_prompt_id": p_id,
                                        "role": q.role,
                                        "angle": q.angle,
                                        "title": q.title,
                                        "question": q.question,
                                        "candidate_source": q.candidate_source,
                                        "judge_keep": jr.keep,
                                        "judge_resolvability": jr.resolvability,
                                        "judge_info": jr.info,
                                        "judge_decision_impact": jr.decision_impact,
                                        "judge_voi": jr.voi,
                                        "judge_minutes_to_resolve": jr.minutes_to_resolve,
                                        "judge_rationale": jr.rationale,
                                        "judge_raw_line": jr.raw_line,
                                        "keep_final": bool(keep_final_flags[idx_q]),
                                    }
                                )

                            raw_gen_output = "\n\n".join(raw_gen_chunks)

                            # Construction du résultat global (en dicts, pour DataFrame + JSON)
                            res_dict = {
                                "models": {
                                    "main": main_model,
                                    "judge": judge_model,
                                },
                                "params": {
                                    "n_initial_total": n_initial,
                                    "k_keep": k_keep,
                                    "n_mutations": n_mutations,
                                },
                                "seed": seed,
                                "tags": tags,
                                "horizon": horizon,
                                "prompts": prompt_entries,
                                "initial": initial_entries,
                                "expanded": [],  # plus utilisé, conservé pour compat éventuelle
                                "raw_prompt_mutation_output": raw_mut_output,
                                "raw_source_finder_outputs": {
                                    pe["prompt_id"]: pe.get("sources_raw", "") for pe in prompt_entries
                                },
                                "raw_generation_output": raw_gen_output,
                                "raw_expansion_output": "",
                            }

                            st.session_state["evo_result"] = res_dict

# ---------------------- Display results ----------------------

res = st.session_state.get("evo_result")

if res is not None:
    main_model = res["models"]["main"]
    judge_model = res["models"]["judge"]
    seed = res["seed"]
    tags = res["tags"]
    horizon = res["horizon"]
    prompt_entries = res.get("prompts", [])
    initial_entries = res["initial"]

    st.subheader("Run summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Main model (mutations + sources + generation):** `{main_model}`")
        st.markdown(f"**Judge model (strict resolvability):** `{judge_model}`")
    with col2:
        st.markdown(f"**Total N proto-questions (requested):** {res['params']['n_initial_total']}")
        st.markdown(f"**K target kept:** {res['params']['k_keep']}")
    with col3:
        st.markdown(f"**Number of mutated prompts:** {res['params']['n_mutations']}")
        st.markdown(f"**Horizon:** {horizon}")

    st.markdown("**Seed preview:**")
    st.caption(seed[:250] + ("..." if len(seed) > 250 else ""))

    # Table des prompts (seed + mutations)
    st.subheader("Prompts (seed + mutations) and resolution hints")

    if prompt_entries:
        df_prompts = pd.DataFrame(prompt_entries)
        df_prompts_view = df_prompts.copy()

        # joindre les sources pour affichage
        df_prompts_view["sources_joined"] = df_prompts_view["sources"].apply(
            lambda lst: "; ".join(lst) if isinstance(lst, list) else str(lst)
        )

        df_prompts_view = df_prompts_view[
            [
                "prompt_id",
                "kind",
                "focus",
                "rationale",
                "text",
                "sources_joined",
            ]
        ]

        st.caption(
            "Root seed (p0) and mutated prompts (p1, p2, ...) with associated public resolution sources."
        )
        st.dataframe(df_prompts_view, use_container_width=True)
    else:
        st.info("No prompts recorded.")

    # Table initiale des questions
    st.subheader("Proto-questions (generation 0, across all prompts)")

    df_init = pd.DataFrame(initial_entries)

    if not df_init.empty:
        df_init_view = df_init[
            [
                "id",
                "parent_prompt_id",
                "keep_final",
                "judge_keep",
                "judge_resolvability",
                "judge_info",
                "judge_decision_impact",
                "judge_voi",
                "judge_minutes_to_resolve",
                "title",
                "question",
                "candidate_source",
                "angle",
                "judge_rationale",
            ]
        ].copy()

        st.caption(
            "All generation-0 proto-questions with strict judge scores "
            "(resolvability, info, decision impact, VOI, minutes_to_resolve) "
            "and final selection flag (keep_final)."
        )
        st.dataframe(df_init_view, use_container_width=True)
    else:
        st.info("No proto-questions available.")

    # Debug / raw outputs
    with st.expander("Debug: raw model outputs"):
        st.markdown("**Raw prompt mutation output (JSON):**")
        raw_mut = res.get("raw_prompt_mutation_output") or ""
        if raw_mut:
            st.code(raw_mut, language="json")
        else:
            st.caption("No stored raw prompt mutation output (dry_run or mock).")

        st.markdown("**Raw source finder outputs (per prompt, JSON):**")
        raw_src_dict = res.get("raw_source_finder_outputs") or {}
        if raw_src_dict:
            for pid, raw_src in raw_src_dict.items():
                if not raw_src:
                    continue
                st.markdown(f"*Prompt {pid}*")
                st.code(raw_src, language="json")
        else:
            st.caption("No stored raw source finder output (dry_run or mock).")

        st.markdown("**Raw generation output (all prompts):**")
        raw_gen = res.get("raw_generation_output") or ""
        if raw_gen:
            st.code(raw_gen, language="text")
        else:
            st.caption("No stored raw generation output (dry_run or mock).")

        st.markdown("**Judge raw lines (keep=...; ...):**")
        if not df_init.empty:
            lines = df_init[["id", "judge_raw_line"]].to_dict(orient="records")
            for row in lines:
                st.code(f"{row['id']}: {row['judge_raw_line']}", language="text")
        else:
            st.caption("No judge lines available.")

    # Download JSON
    st.subheader("Download JSON")

    json_bytes = _json.dumps(
        res,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    st.download_button(
        "Download full run (JSON)",
        data=json_bytes,
        file_name="metaculus_prompt_mutation_proto_questions.json",
        mime="application/json",
    )

    # ---------------------- Interactive refinement chat (enhanced UX) ----------------------
    st.subheader("Interactive refinement chat")

    kept_questions = [e for e in initial_entries if e.get("keep_final")]
    if kept_questions:
        with st.expander("Shortlisted questions (keep_final = True)", expanded=False):
            for e in kept_questions:
                st.markdown(
                    f"- **{e['id']}** – {e['title']}\n\n"
                    f"  {e['question']}"
                )
    else:
        st.info("No questions are marked as keep_final. The chat will still work, but context is empty.")

    # Préparer le bloc de questions pour le contexte du modèle
    if kept_questions:
        q_lines = [
            f"{idx+1}. {e['id']} – {e['title']} – {e['question']}"
            for idx, e in enumerate(kept_questions)
        ]
        questions_block = "\n".join(q_lines)
    else:
        questions_block = "None (no shortlisted questions)."

    chat_system_prompt = f"""
You are a forecasting question refinement assistant for Metaculus.

Static context for this chat (do NOT restate it unless useful for clarity):

Seed:
{seed}

Domain tags: {', '.join(tags)}

Horizon: {horizon}

Current shortlisted questions (id – title – question):
{questions_block}

Your role:
- Read the user's feedback about these questions.
- Propose improved or alternative proto-questions, still resolvable from public sources,
  and broadly aligned with the same seed, tags and horizon.
- Focus on: clarity, resolvability, practical value-of-information, and diversity of angles.
- You may explicitly reference question ids (e.g. "g0-q3") to say how you are modifying them.

Output format:
- Always answer in English.
- Use plain text, no JSON, no code fences.
- Prefer a concise numbered list of suggested questions:
  "<n>. [id or NEW] Short title – Full question sentence?"
- You may also briefly comment (1–2 sentences) on why these changes are improvements,
  but keep the answer compact and focused on concrete question text.
""".strip()

    if "refine_chat_history" not in st.session_state:
        st.session_state["refine_chat_history"] = []

    # Affichage de l'historique existant avec rendu type "boîte"
    for msg in st.session_state["refine_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "revised_questions" in msg:
                for q in msg["revised_questions"]:
                    with st.container():
                        st.markdown(f"**{q['id']}** – *{q['title']}*")
                        st.markdown(q["question"])
                        with st.expander("Resolution criteria"):
                            st.markdown(
                                f"- **Resolvability:** {q['resolvability']}/5  \n"
                                f"- **Information value:** {q['info']}/5  \n"
                                f"- **Decision impact:** {q['decision_impact']:.2f}  \n"
                                f"- **VOI:** {q['voi']:.2f}  \n"
                                f"- **Minutes to resolve:** {q['minutes_to_resolve']:.1f}  \n"
                                f"- **Rationale:** {q['rationale']}"
                            )

    # Entrée utilisateur
    user_input = st.chat_input("Give feedback about the questions, or ask for new variants.")

    if user_input:
        # Ajouter le message utilisateur à l'historique
        st.session_state["refine_chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Construire la conversation complète à envoyer
        or_messages: List[Dict[str, str]] = [{"role": "system", "content": chat_system_prompt}]
        for m in st.session_state["refine_chat_history"]:
            or_messages.append({"role": m["role"], "content": m["content"]})

        # Appel du modèle principal
        if dry_run:
            assistant_reply = (
                "Dry-run mode: I would normally propose refined or new proto-questions "
                "based on your feedback and the shortlisted questions."
            )
        elif not get_openrouter_key():
            assistant_reply = (
                "No OPENROUTER_API_KEY is configured, so I cannot call the refinement model. "
                "Please add a key in the sidebar or enable dry_run."
            )
        else:
            try:
                raw_reply = call_openrouter_raw(
                    messages=or_messages,
                    model=main_model,
                    max_tokens=1200,
                    temperature=0.5,
                )
                assistant_reply = raw_reply.strip()
            except Exception as e:
                assistant_reply = f"Error from refinement assistant: {e}"

        # Construire la liste des questions à afficher avec critères
        revised_qs: List[Dict[str, Any]] = []
        for e in kept_questions:
            revised_qs.append(
                {
                    "id": e["id"],
                    "title": e["title"],
                    "question": e["question"],
                    "resolvability": e["judge_resolvability"],
                    "info": e["judge_info"],
                    "decision_impact": e["judge_decision_impact"],
                    "voi": e["judge_voi"],
                    "minutes_to_resolve": e["judge_minutes_to_resolve"],
                    "rationale": e["judge_rationale"],
                }
            )

        # Afficher la nouvelle "boîte de dialogue" assistant + questions/critères
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
            for q in revised_qs:
                with st.container():
                    st.markdown(f"**{q['id']}** – *{q['title']}*")
                    st.markdown(q["question"])
                    with st.expander("Resolution criteria"):
                        st.markdown(
                            f"- **Resolvability:** {q['resolvability']}/5  \n"
                            f"- **Information value:** {q['info']}/5  \n"
                            f"- **Decision impact:** {q['decision_impact']:.2f}  \n"
                            f"- **VOI:** {q['voi']:.2f}  \n"
                            f"- **Minutes to resolve:** {q['minutes_to_resolve']:.1f}  \n"
                            f"- **Rationale:** {q['rationale']}"
                        )

        # Sauvegarder dans l'historique complet
        st.session_state["refine_chat_history"].append(
            {"role": "assistant", "content": assistant_reply, "revised_questions": revised_qs}
        )

else:
    st.info(
        "Configure the number of mutated prompts, total N questions, K kept, "
        "set your seed, then click the button to run the pipeline."
    )
