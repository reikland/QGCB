"""Prompt templates used throughout the pipeline."""

import textwrap

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
- The Candidate-source line MUST name precise public pages or endpoints to use for resolution
  (dataset + table/report name, edition/year, and URL or platform path if known). If several
  similarly named pages exist, pick the most authoritative one and add a short disambiguation
  cue (e.g., "use the 2024 IMF WEO October database country table for GDP, not earlier editions").

STRICT FORMAT (LINE-BASED, WITH METACULUS STRUCTURE INSIDE THE QUESTION BLOCK)
For each i = 1..N you output a block with these lines (use EXACT labels):

QUESTION i
Role: CORE or VARIANT
Title: <short title, <= 100 characters, single line>
Question: Title: <repeat title here>
  Resolution Criteria: <explicit, time-bounded, testable resolution event; include exact formulae and baselines>
  Fine Print: <short sentences (no bullet points) covering exclusions, boundary behaviour, delayed/non-announced events, and a single authoritative resolution source with a named backup>
  Rating: <Publishable | Soft Reject | Hard Reject>
  Rationale: <2–4 sentences, objective justification for the rating>
Angle: <short phrase capturing the angle within the cluster>
Candidate-source: <precise public page(s) or endpoint to resolve, with disambiguation if needed>

Additional generation rules (APPLY WITHIN THE QUESTION BLOCK):
- Remove ambiguity on time horizons: cite exact announcement date vs implementation date, and the resolution date/timezone.
- State explicit baselines/denominators/units for any percentage, currency, or rate; if comparing to a past value, name the reference year/figure.
- Define fallback behaviour: what happens if the event is not announced, delayed, annulled, or partially implemented.
- Specify actors precisely (e.g., "U.S. Department of Energy" rather than "the government"), and specify which publications count as official.
- Candidate-source MUST name two or three precise public sources and, in one short sentence, explain how they will be used to resolve the question.
- Do NOT output any scoring or meta-evaluation fields (e.g., resolvability scores, information value, VOI, decision impact) beyond the Rating + Rationale requested.
- Keep the Rating section at the end and include exactly ONE rating per question.

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

# ---------------------- FICHES DE RÉSOLUTION (QUESTIONS CONSERVÉES) ----------------------

RESOLUTION_CARD_SYS = """
You prepare concise RESOLUTION SHEETS for kept Metaculus-style questions.

Goal:
- Produce a short, ready-to-paste card that restates the question using full sentences.
- Keep the canonical ordering: Title, Resolution Criteria, Fine Print, Resolution Sources.
- Use sentence-style paragraphs (no bullet points, no code fences, no markdown tables).

Rules:
- Do not invent new constraints: stay faithful to the provided Title/Question/Fine Print content.
- Clarify any ambiguous time horizon or announcement vs implementation phrasing if the provided text is unclear.
- Keep Resolution Sources to two or three items and explain in one or two sentences how they will be used to resolve.
- Keep it compact: total length should be a few short paragraphs, not verbose prose.
""".strip()

RESOLUTION_CARD_USER_TMPL = textwrap.dedent(
    """
    Seed (context):
    {seed}

    Tags: {tags}
    Horizon: {horizon}

    Kept question (keep_final=True):
    ID: {question_id}
    Title: {title}
    Body (as generated):
    {question_block}

    Candidate-source line (keep to 2–3 concrete sources):
    {candidate_source}

    Rating: {rating}
    Rating rationale: {rating_rationale}

    Produce a concise resolution sheet with the sections:
    Title
    Resolution Criteria
    Fine Print
    Resolution Sources (2–3 sources with one sentence on how they will be used)

    Use full sentences (no bullet points) and keep the tone factual and ready to copy/paste.
    """
)

# ---------------------- JUDGE LIGHT (keep K parmi N) ----------------------

JUDGE_SYS_KEEP = """
You are a FAST, STRICT judge. Priorities in order: resolvability from PUBLIC sources, then information value, then speed to resolve.

You MUST output ONE line only:
keep=0|1; rating=Publishable|Soft Reject|Hard Reject; resolvability=X; info=Y; decision_impact=D; voi=V; minutes_to_resolve=R; rationale=TEXT

Rules (keep it short):
- X,Y are integers 1–5. D is a float 0–1. V,R are floats (R in minutes).
- rating must be exactly Publishable, Soft Reject, or Hard Reject.
- rationale <=200 chars, no semicolons; justify BOTH keep and rating.
- First characters MUST be "keep=". No extra lines, JSON, or markdown.

Decision test (strict):
- If the question is likely already resolved OR resolvability <=3 → keep=0.
- keep=1 only if resolvability >=4 AND info >=3 AND a forecaster with chatbot+web could fully resolve it in <=15 minutes using public sources.
""".strip()

JUDGE_USER_TMPL_KEEP = textwrap.dedent(
    """
    Judge this proto forecasting question.

    Context (seed + mutated prompt + sources):
    {seed}

    Horizon: {horizon}
    Domain tags: {tags}

    Proto-question:
    Title: {title}
    Question: {question}
    Candidate-source: {source}

    Decision rule (strict):
    - If resolvability <=3 → keep=0.
    - keep=1 only if resolvability >=4, info >=3, and a chatbot+web user can resolve in <=15 minutes from PUBLIC sources.

    Extra fields (do not change the rule):
    - decision_impact: 0–1
    - voi: float
    - minutes_to_resolve: float (minutes)

    Always assign rating + short rationale even when keep=0.

    Output exactly:
    keep=0|1; rating=Publishable|Soft Reject|Hard Reject; resolvability=X; info=Y; decision_impact=D; voi=V; minutes_to_resolve=R; rationale=TEXT
    """
)


# ============================================================
