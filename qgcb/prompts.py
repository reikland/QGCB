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
