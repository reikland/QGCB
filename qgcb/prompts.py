"""Prompt templates used throughout the pipeline."""

import textwrap

CATEGORIES = [
    "Technology",
    "Social Sciences",
    "Natural Sciences",
    "Economy & Business",
    "Environment & Climate",
    "Sports & Entertainment",
    "Space",
    "Nuclear Technology & Risks",
    "Artificial Intelligence",
    "Cryptocurrencies",
    "Computing and Math",
    "Health & Pandemics",
    "Politics",
    "Metaculus",
    "Law",
    "Geopolitics",
    "Elections",
]

CATEGORIES_DISPLAY = ", ".join(CATEGORIES)

# 3. PROMPTS – GÉNÉRATION / JUDGE / MUTATION DE PROMPTS / SOURCES
# ============================================================

# ---------------------- QUESTION -> SEED DERIVATION (JSON) ----------------------

SEED_DERIVER_SYS = """
You convert a single raw forecasting question into a SEED prompt, domain tags, and a resolution horizon.

Output JSON only, matching this schema:
{
  "seed": "2–4 sentences capturing the core uncertainty and scope",
  "tags": ["short_tag", "short_tag", "..."],
  "horizon": "resolve by YYYY-MM-DD UTC"
}

Rules:
- The seed must be concrete and resolvable, not just a copy of the question.
- Tags should be 3–6 short, lower-case domain labels (no hashtags).
- The horizon must be a future date. If the question has an explicit date, use it.
- If no date is given, choose a reasonable horizon (within ~1–10 years).
- Output JSON only; no extra text.
""".strip()

SEED_DERIVER_USER_TMPL = textwrap.dedent(
    """
    Raw question:
    {question}

    Produce the JSON output described in the system message.
    """
)

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
You generate CLUSTERS of proto forecasting questions for Metaculus. Be fast: skip deliberation and emit the required blocks immediately.

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
- Question types MUST follow the requested distribution (binary / numeric / multiple_choice).

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
- Each question MUST include a Category from this fixed list: {categories_list}.
- Type must be one of: binary, numeric, multiple_choice.
- For numeric questions:
  - Provide a reasonable Range-min/Range-max and set Open-lower-bound / Open-upper-bound to true/false.
  - Use inbound_outcome_count=200 unless the question resolves to a small integer range (e.g., 0–7),
    in which case inbound_outcome_count should equal the number of distinct integer outcomes (e.g., 8).
  - If the scale is log-like, set Zero-point to a number below Range-min; otherwise leave Zero-point blank.
- For multiple_choice questions:
  - Provide Options as a pipe-separated list, and Group-variable as the type of thing being chosen.
- For binary questions:
  - Leave Options/Group-variable/Range-min/Range-max/Zero-point/Open-* /Unit blank.
- Question-weight should be 1 unless the question's resolution is significantly correlated with another
  question in the cluster, in which case set a value below 1.

STRICT FORMAT (LINE-BASED AND LIGHTWEIGHT)
For each i = 1..N you output a block with these lines (use EXACT labels):

QUESTION i
Role: CORE or VARIANT
Title: <short title, <= 100 characters, single line>
Question: <2–4 sentences that fully specify the resolution criteria, time bounds, actors, units, and fallback handling; do not add ratings>
Angle: <short phrase capturing the angle within the cluster>
Category: <choose ONE value from {categories_list}>
Question-weight: <float; use 1 unless correlated with another question>
Type: <binary|numeric|multiple_choice>
Inbound-outcome-count: <integer; required for numeric only, blank otherwise>
Options: <pipe-separated options; required for multiple_choice only, blank otherwise>
Group-variable: <type of thing chosen; required for multiple_choice only, blank otherwise>
Range-min: <numeric minimum; required for numeric only, blank otherwise>
Range-max: <numeric maximum; required for numeric only, blank otherwise>
Zero-point: <numeric; only if log-scaled, blank otherwise>
Open-lower-bound: <true|false; required for numeric only, blank otherwise>
Open-upper-bound: <true|false; required for numeric only, blank otherwise>
Unit: <display unit; required for numeric only, blank otherwise>
Candidate-source: <two or three precise public pages or endpoints to resolve, with one short use note>

Between blocks you MAY optionally have a single blank line.
You MUST NOT output anything else before, between, or after the blocks.
""".strip().format(categories_list=CATEGORIES_DISPLAY)

GEN_USER_TMPL_INITIAL = textwrap.dedent(
    """
    You must now generate a CLUSTER of proto forecasting questions.

    HARD CONSTRAINT:
    - N_questions = {n}.
    - You MUST output EXACTLY N_questions blocks, labelled QUESTION 1, QUESTION 2, ..., QUESTION {n}.
    - Any extra text or missing block makes the output INVALID.
      - Type distribution for this cluster MUST be:
        - binary: {n_binary}
        - numeric: {n_numeric}
        - multiple_choice: {n_multiple_choice}
    - Each question MUST include a Category line with ONE value chosen from: {categories_list}.

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
    Do NOT restate the instructions. Do NOT explain your choices. Work quickly and avoid any extra wording.
    """
)

# ---------------------- TYPE REBALANCER (JSON) ----------------------

TYPE_REBALANCER_SYS = """
You rebalance proto forecasting questions across binary / numeric / multiple_choice types.

Target distribution (strict):
- binary: 50%
- numeric: 30%
- multiple_choice: 20%

Objective:
- Adjust ONLY the minimal subset of questions needed to hit the target counts.
- Preserve the original meaning and resolution criteria; conversions must be "à la marge" (light touch).
- Prefer safe conversions (e.g., numeric threshold → binary yes/no; binary about levels → numeric with explicit bounds; multi-choice → binary by collapsing to the most meaningful option; binary → multi-choice by expanding mutually exclusive, explicit outcomes).
- Keep ordering and identifiers stable; NEVER drop a question.

Output JSON ONLY with this shape:
{
  "questions": [
    {
      "index": 1,
      "title": "...",
      "question": "...",
      "type": "binary|numeric|multiple_choice",
      "options": "...",
      "group_variable": "...",
      "range_min": 0,
      "range_max": 0,
      "open_lower_bound": false,
      "open_upper_bound": false,
      "unit": "...",
      "inbound_outcome_count": 200
    },
    ...
  ],
  "notes": "short summary of what changed"
}

Rules:
- Keep anchor titles and angles intact unless a minor tweak is needed for the new type.
- Do not invent new resolution targets or sources; stay faithful to the provided text.
- Always include the FINAL type for every question so the controller can audit the batch.
- If distribution already matches the target, echo the questions as-is (still include type for each).
""".strip()

TYPE_REBALANCER_USER_TMPL = textwrap.dedent(
    """
    Target distribution: binary=50%, numeric=30%, multiple_choice=20%.
    Current counts: {current_counts}
    Target counts for N={n_questions}: {target_counts}

    Question inventory (keep order and IDs):
    {questions_block}

    Adjust the MINIMUM number of questions to reach the exact target counts.
    Keep semantics and resolution paths intact; avoid risky rewrites.
    Respond with JSON only as described in the system message.
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
You are a FAST, STRICT judge for proto forecasting questions, with an OBSESSION for resolvability. Respond immediately; do not waste time.

Your ONLY task is to decide whether to KEEP or DISCARD ONE question, based on:
- resolvability from PUBLIC sources (this is the PRIMARY criterion),
- information value for forecasting and decision-making (secondary),
- practical solvability.

Practical solvability:
- Imagine a careful forecaster working with a general-purpose chatbot + web search.
- Once the resolution date has passed, that team should be able to fully resolve the question in <= 15 minutes.
- If you cannot see a concrete, realistic path to resolution within that constraint, resolvability is LOW.

You MUST output EXACTLY ONE LINE, with this format:

keep=0|1; rating=Publishable|Soft Reject|Hard Reject; resolvability=X; info=Y; decision_impact=D; voi=V; minutes_to_resolve=R; rationale=TEXT

Hard constraints:
- X and Y MUST be integers from 1 to 5.
- D MUST be a float between 0 and 1 (0 = no impact on decisions for an average global citizen, 1 = very high impact).
- V and R MUST be floats (V unbounded, higher = higher value of information; R >= 0, in minutes, lower = easier to resolve).
- rating MUST be exactly one of: Publishable, Soft Reject, Hard Reject.
- rationale MUST be <= 200 characters and MUST NOT contain semicolons. It should concisely justify the rating + keep decision.
- The very first non-space characters of your reply MUST be "keep=".
- You MUST NOT add any other lines, JSON, markdown, or commentary.
- No bullet lists. No explanations before or after the line.
- If you are unsure, choose a reasonable guess and still follow the format. Never omit the rating field.

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

    You MUST always assign a publication verdict and short justification even if keep=0:
    - rating: Publishable | Soft Reject | Hard Reject
    - rationale: 1–3 sentences (<=200 chars) justifying BOTH the keep value and the rating.

    Now output ONLY the single line:
    keep=0|1; rating=Publishable|Soft Reject|Hard Reject; resolvability=X; info=Y; decision_impact=D; voi=V; minutes_to_resolve=R; rationale=TEXT
    """
)


# ============================================================
