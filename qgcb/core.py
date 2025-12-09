#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import textwrap
import random
from typing import Dict, Any, List, Optional

import requests
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
