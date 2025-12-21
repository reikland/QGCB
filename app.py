from typing import Any, Dict, List
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from qgcb import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAIN_MODEL,
    JudgeKeepResult,
    ProtoQuestion,
    call_openrouter_raw,
    derive_seed_from_question,
    find_resolution_sources_for_prompt,
    generate_initial_questions,
    generate_resolution_card,
    get_openrouter_key,
    judge_initial_questions,
    mutate_seed_prompt,
    select_top_k,
)

# ============================================================
# 6. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Metaculus – Evolutionary Proto Question Generator",
    page_icon="🧬",
    layout="wide",
)

if "evo_result" not in st.session_state:
    st.session_state["evo_result"] = None
if "resolution_cards" not in st.session_state:
    st.session_state["resolution_cards"] = {}
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None


def run_full_pipeline(
    seed: str,
    tags: List[str],
    horizon: str,
    main_model: str,
    judge_model: str,
    n_mutations: int,
    n_initial: int,
    k_keep: int,
    dry_run: bool,
    status_prefix: str = "",
) -> Dict[str, Any] | None:
    # A) Mutations de prompt
    with st.spinner(
        f"{status_prefix}Step 1/4 – Mutating seed prompt into more concrete prompts..."
    ):
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
            st.error(f"{status_prefix}Prompt mutation error: {e}")
            return None

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
    with st.spinner(
        f"{status_prefix}Step 2/4 – Finding concrete resolution sources for each prompt..."
    ):
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
                st.error(
                    f"{status_prefix}Source finder error for prompt {pe['prompt_id']}: {e}"
                )
                src_res = {"sources": ["(error: fallback generic sources)"], "raw_output": ""}

            pe["sources"] = src_res.get("sources", []) or ["(no sources returned)"]
            pe["sources_raw"] = src_res.get("raw_output", "")

    # C) Génération de questions pour l'ensemble des prompts (répartition N_total)
    generated_at = datetime.now(timezone.utc).isoformat()

    with st.spinner(
        f"{status_prefix}Step 3/4 – Generating proto-questions across all prompts and judging them..."
    ):
        all_questions: List[ProtoQuestion] = []
        all_prompt_ids: List[str] = []
        raw_gen_chunks: List[str] = []

        P = len(prompt_entries)
        if P == 0:
            st.error(f"{status_prefix}No prompts available (seed + mutations).")
            return None

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
                    prompt_text=pe["text"],
                    resolution_hints=hints_str,
                    model=main_model,
                    dry_run=dry_run,
                )
            except Exception as e:
                st.error(f"{status_prefix}Generation error for prompt {pe['prompt_id']}: {e}")
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
            st.error(f"{status_prefix}No proto-questions were parsed. Check generator prompts.")
            return None

        # Construire les contextes individuels pour le judge
        prompt_by_id = {pe["prompt_id"]: pe for pe in prompt_entries}
        judge_contexts: List[str] = []

        for q_idx, _ in enumerate(all_questions):
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

        # Strict judge (resolvability + info + decision impact)
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
            st.error(f"{status_prefix}Judge error: {e}")
            return None

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
            f"{status_prefix}Judge initial keep=1 count: {n_kept_initial} / {len(all_questions)}; "
            f"Selected (final K): {n_selected_final} (target K={k_keep})."
        )

        # Attribuer IDs et génération
        initial_entries: List[Dict[str, Any]] = []

        for idx_q, q in enumerate(all_questions):
            q_id = f"g0-q{idx_q+1}"
            jr = judge_res0[idx_q]
            p_id = all_prompt_ids[idx_q]
            rating_val = (q.rating or jr.verdict or "").strip()
            rating_rationale_val = (
                jr.verdict_rationale or jr.rationale or q.rating_rationale or ""
            ).strip()
            initial_entries.append(
                {
                    "id": q_id,
                    "generation": 0,
                    "parent_prompt_id": p_id,
                    "role": q.role,
                    "angle": q.angle,
                    "title": q.title,
                    "question": q.question,
                    "question_weight": q.question_weight,
                    "type": q.type,
                    "inbound_outcome_count": q.inbound_outcome_count,
                    "options": q.options,
                    "group_variable": q.group_variable,
                    "range_min": q.range_min,
                    "range_max": q.range_max,
                    "zero_point": q.zero_point,
                    "open_lower_bound": q.open_lower_bound,
                    "open_upper_bound": q.open_upper_bound,
                    "unit": q.unit,
                    "raw_question_block": getattr(q, "raw_block", ""),
                    "candidate_source": q.candidate_source,
                    "rating": rating_val,
                    "rating_rationale": rating_rationale_val,
                    "judge_keep": jr.keep,
                    "judge_resolvability": jr.resolvability,
                    "judge_info": jr.info,
                    "judge_decision_impact": jr.decision_impact,
                    "judge_voi": jr.voi,
                    "judge_minutes_to_resolve": jr.minutes_to_resolve,
                    "judge_verdict": jr.verdict,
                    "judge_verdict_rationale": jr.verdict_rationale,
                    "judge_rationale": jr.rationale,
                    "judge_raw_line": jr.raw_line,
                    "keep_final": bool(keep_final_flags[idx_q]),
                }
            )

        raw_gen_output = "\n\n".join(raw_gen_chunks)

        kept_entries = [e for e in initial_entries if e.get("keep_final")]
        card_store: Dict[str, Any] = {}

        if kept_entries:
            card_store = {}
            if dry_run or get_openrouter_key():
                with st.spinner(
                    f"{status_prefix}Step 4/4 – Generating resolution cards for kept questions..."
                ):
                    for entry in kept_entries:
                        try:
                            card_res = generate_resolution_card(
                                question_entry=entry,
                                seed=seed,
                                tags=tags,
                                horizon=horizon,
                                model=main_model,
                                dry_run=dry_run,
                            )
                            card_store[entry["id"]] = card_res
                        except Exception as e:
                            st.error(
                                f"{status_prefix}Resolution card generation error for {entry['id']}: {e}"
                            )
            else:
                st.error(
                    f"{status_prefix}No OPENROUTER_API_KEY detected. Add it in the sidebar or enable dry_run to build resolution cards."
                )

        # Build the global result structure (dicts for DataFrame + JSON)
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
            "run_metadata": {
                "generated_at_utc": generated_at,
                "resolution_horizon": horizon,
            },
            "seed": seed,
            "tags": tags,
            "horizon": horizon,
            "prompts": prompt_entries,
            "initial": initial_entries,
            "resolution_cards": card_store,
            "expanded": [],  # legacy placeholder for potential future generations
            "raw_prompt_mutation_output": raw_mut_output,
            "raw_source_finder_outputs": {
                pe["prompt_id"]: pe.get("sources_raw", "") for pe in prompt_entries
            },
            "raw_generation_output": raw_gen_output,
            "raw_expansion_output": "",
        }

        return res_dict

st.title("Metaculus – Evolutionary Proto Question Generator")

st.markdown(
    """
Four-stage pipeline (PROMPTS → SOURCES → QUESTIONS → RESOLUTION CARDS):

1. **Prompt mutations**: starting from your seed, the main model creates several more concrete,
   resolvable prompts (mutations).
2. **Resolution sources (judge-bis)**: for EACH prompt (seed + mutations), a light model proposes
   concrete public resolution sources (datasets, agencies, news wires, etc.).
3. **Question generation + strict judge**: the main model generates proto-questions around all prompts
   (respecting the resolution sources), then the judge filters them with an obsession for resolvability.
4. **Resolution cards (auto)**: for every kept question, the same main model drafts a ready-to-copy
   Metaculus-style resolution card using the question text and source hints.

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
        max_value=20,
        value=20,
        step=1,
        help="Maximum 20 proto-questions can be generated per run.",
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

input_mode = st.radio(
    "Input mode",
    ["Single seed", "CSV questions"],
    horizontal=True,
)

seed = ""
tags_str = ""
horizon = ""
run_button = False
uploaded_questions: List[str] = []

if input_mode == "CSV questions":
    st.subheader("CSV questions → seeds")
    csv_file = st.file_uploader(
        "Upload a CSV file containing a column of questions",
        type=["csv"],
    )

    if csv_file is not None:
        try:
            df_questions = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
            df_questions = None

        if df_questions is not None and not df_questions.empty:
            question_col = st.selectbox(
                "Column containing the raw questions",
                df_questions.columns.tolist(),
            )
            raw_questions = (
                df_questions[question_col]
                .dropna()
                .astype(str)
                .tolist()
            )
            if not raw_questions:
                st.warning("No questions found in the selected column.")
            else:
                max_q = len(raw_questions)
                n_to_process = st.slider(
                    "Number of questions to process",
                    min_value=1,
                    max_value=max_q,
                    value=min(3, max_q),
                    step=1,
                )
                uploaded_questions = raw_questions[:n_to_process]
                st.caption(
                    f"Processing the first {n_to_process} questions from '{question_col}'."
                )
                run_button = st.button(
                    "Run batch pipeline (CSV → SEEDS → QGCB)"
                )
        else:
            st.warning("Upload a CSV file with at least one row.")
else:
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

    horizon = st.text_input(
        "Horizon / rough timeline",
        value="resolve by 2040-12-31 UTC",
    )

    run_button = st.button("Run full pipeline (PROMPTS → SOURCES → QUESTIONS)")

    tags_str = st.text_input(
        "Domain tags (comma-separated)",
        value="ai,policy,macro",
    )

if run_button:
    main_model = (main_model_input or "").strip() or DEFAULT_MAIN_MODEL
    judge_model = (judge_model_input or "").strip() or DEFAULT_JUDGE_MODEL

    if input_mode == "CSV questions":
        if not uploaded_questions:
            st.warning("Please upload a CSV file with questions to process.")
        elif not dry_run and not current_key:
            st.error("No OPENROUTER_API_KEY set and dry_run is disabled.")
        else:
            st.info(
                f"Using main model (mutations + sources + generation): `{main_model}`\n\n"
                f"Using judge model (strict resolvability): `{judge_model}`"
            )

            derived_rows: List[Dict[str, Any]] = []
            with st.spinner("Step 0/4 – Deriving seeds, tags, and horizons from CSV questions..."):
                for idx, question_text in enumerate(uploaded_questions, start=1):
                    try:
                        derived = derive_seed_from_question(
                            question_text=question_text,
                            model=main_model,
                            dry_run=dry_run,
                        )
                    except Exception as e:
                        st.error(f"[{idx}/{len(uploaded_questions)}] Seed derivation error: {e}")
                        continue
                    derived_rows.append(
                        {
                            "input_question": question_text,
                            "seed": derived["seed"],
                            "tags": derived["tags"],
                            "horizon": derived["horizon"],
                            "raw_seed_output": derived.get("raw_output", ""),
                        }
                    )

            if not derived_rows:
                st.error("No seeds could be derived from the CSV questions.")
            else:
                batch_results: List[Dict[str, Any]] = []
                for idx, row in enumerate(derived_rows, start=1):
                    status_prefix = f"[{idx}/{len(derived_rows)}] "
                    res_dict = run_full_pipeline(
                        seed=row["seed"],
                        tags=row["tags"],
                        horizon=row["horizon"],
                        main_model=main_model,
                        judge_model=judge_model,
                        n_mutations=n_mutations,
                        n_initial=n_initial,
                        k_keep=k_keep,
                        dry_run=dry_run,
                        status_prefix=status_prefix,
                    )
                    if res_dict is None:
                        continue
                    batch_results.append(
                        {
                            "input_question": row["input_question"],
                            "seed": row["seed"],
                            "tags": row["tags"],
                            "horizon": row["horizon"],
                            "raw_seed_output": row["raw_seed_output"],
                            "result": res_dict,
                        }
                    )

                st.session_state["batch_results"] = batch_results
                st.session_state["evo_result"] = None
                st.session_state["resolution_cards"] = {}
    else:
        if not seed.strip():
            st.warning("Please provide a seed prompt.")
        elif not dry_run and not current_key:
            st.error("No OPENROUTER_API_KEY set and dry_run is disabled.")
        else:
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            st.info(
                f"Using main model (mutations + sources + generation): `{main_model}`\n\n"
                f"Using judge model (strict resolvability): `{judge_model}`"
            )
            res_dict = run_full_pipeline(
                seed=seed,
                tags=tags,
                horizon=horizon,
                main_model=main_model,
                judge_model=judge_model,
                n_mutations=n_mutations,
                n_initial=n_initial,
                k_keep=k_keep,
                dry_run=dry_run,
            )
            st.session_state["evo_result"] = res_dict
            st.session_state["resolution_cards"] = (
                res_dict.get("resolution_cards", {}) if res_dict else {}
            )
            st.session_state["batch_results"] = None

# ---------------------- Display results ----------------------

res = st.session_state.get("evo_result")
batch_results = st.session_state.get("batch_results")

if input_mode == "CSV questions":
    if not batch_results:
        st.info(
            "Upload a CSV list of questions, choose how many to process, then run the batch pipeline."
        )
    else:
        tab_overview, tab_cards = st.tabs(
            ["Overview & questions", "Resolution cards (kept questions)"]
        )

        with tab_overview:
            st.subheader("Batch run summary")
            seed_rows = []
            for idx, run in enumerate(batch_results, start=1):
                seed_rows.append(
                    {
                        "batch_id": f"b{idx}",
                        "input_question": run["input_question"],
                        "seed": run["seed"],
                        "domain_tags": ", ".join(run["tags"]),
                        "resolution_horizon": run["horizon"],
                    }
                )

            df_seeds = pd.DataFrame(seed_rows)
            st.caption("Derived seeds, tags, and horizons from the CSV questions.")
            st.dataframe(df_seeds, width="stretch")

            kept_rows: List[Dict[str, Any]] = []
            for idx, run in enumerate(batch_results, start=1):
                res_run = run["result"]
                seed = res_run["seed"]
                tags = res_run["tags"]
                horizon = res_run["horizon"]
                card_store = res_run.get("resolution_cards", {})
                for entry in res_run.get("initial", []):
                    if not entry.get("keep_final"):
                        continue
                    card_entry = card_store.get(entry["id"], {})
                    kept_rows.append(
                        {
                            "batch_id": f"b{idx}",
                            "input_question": run["input_question"],
                            "seed": seed,
                            "domain_tags": ", ".join(tags),
                            "resolution_horizon": horizon,
                            "id": entry["id"],
                            "title": entry["title"],
                            "question": entry["question"],
                            "resolution_card": card_entry.get("card", ""),
                            "question_weight": entry.get("question_weight"),
                            "type": entry.get("type"),
                            "inbound_outcome_count": entry.get("inbound_outcome_count"),
                            "options": entry.get("options"),
                            "group_variable": entry.get("group_variable"),
                            "range_min": entry.get("range_min"),
                            "range_max": entry.get("range_max"),
                            "zero_point": entry.get("zero_point"),
                            "open_lower_bound": entry.get("open_lower_bound"),
                            "open_upper_bound": entry.get("open_upper_bound"),
                            "unit": entry.get("unit"),
                            "candidate_source": entry.get("candidate_source"),
                            "angle": entry.get("angle"),
                            "judge_rationale": entry.get("judge_rationale"),
                            "raw_question_block": entry.get("raw_question_block"),
                            "judge_resolvability": entry["judge_resolvability"],
                            "judge_info": entry["judge_info"],
                            "judge_decision_impact": entry["judge_decision_impact"],
                            "judge_voi": entry["judge_voi"],
                            "judge_minutes_to_resolve": entry["judge_minutes_to_resolve"],
                            "judge_verdict": entry.get("judge_verdict", ""),
                            "judge_verdict_rationale": entry.get("judge_verdict_rationale", ""),
                        }
                    )

            st.subheader("Download CSV (kept questions + resolution cards)")
            if kept_rows:
                df_kept = pd.DataFrame(kept_rows)
                st.download_button(
                    "Download kept questions + resolution cards (CSV)",
                    data=df_kept.to_csv(index=False).encode("utf-8"),
                    file_name="metaculus_kept_questions_with_cards_batch.csv",
                    mime="text/csv",
                )
            else:
                st.caption("No kept questions available across the batch.")

            st.subheader("Per-seed details")
            for idx, run in enumerate(batch_results, start=1):
                res_run = run["result"]
                seed = res_run["seed"]
                tags = res_run["tags"]
                horizon = res_run["horizon"]
                prompt_entries = res_run.get("prompts", [])
                initial_entries = res_run.get("initial", [])

                with st.expander(
                    f"Batch b{idx} – {seed[:80]}{'...' if len(seed) > 80 else ''}",
                    expanded=False,
                ):
                    st.markdown(f"**Input question:** {run['input_question']}")
                    st.markdown(f"**Seed:** {seed}")
                    st.markdown(f"**Domain tags:** {', '.join(tags)}")
                    st.markdown(f"**Horizon:** {horizon}")

                    st.subheader("Prompts (seed + mutations) and resolution hints")
                    if prompt_entries:
                        df_prompts = pd.DataFrame(prompt_entries)
                        df_prompts_view = df_prompts.copy()
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
                        st.dataframe(df_prompts_view, width="stretch")
                    else:
                        st.info("No prompts recorded.")

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
                                "judge_verdict",
                                "judge_verdict_rationale",
                                "title",
                                "question",
                                "question_weight",
                                "type",
                                "inbound_outcome_count",
                                "options",
                                "group_variable",
                                "range_min",
                                "range_max",
                                "zero_point",
                                "open_lower_bound",
                                "open_upper_bound",
                                "unit",
                                "candidate_source",
                                "angle",
                                "judge_rationale",
                                "raw_question_block",
                            ]
                        ].copy()
                        st.dataframe(df_init_view, width="stretch")
                    else:
                        st.info("No proto-questions available.")

            st.subheader("Follow-up chat (GPT‑5 with kept questions)")
            batch_ids = [f"b{idx}" for idx in range(1, len(batch_results) + 1)]
            selected_batch = st.selectbox(
                "Choose a batch to chat with",
                batch_ids,
            )
            selected_index = int(selected_batch[1:]) - 1
            selected_run = batch_results[selected_index]
            selected_res = selected_run["result"]
            selected_seed = selected_res["seed"]
            selected_tags = selected_res["tags"]
            selected_horizon = selected_res["horizon"]
            selected_kept = [
                e for e in selected_res["initial"] if e.get("keep_final")
            ]

            kept_preview_lines = [
                f"  • {e['id']} – {e['title']} – {e['question']}"
                for e in selected_kept[:8]
            ] or ["  • None"]

            chat_system_prompt = (
                "You are a fast, concise assistant for follow-up on kept forecasting questions. "
                "Reply in plain text (no markdown fences), keep the conversation going without resetting context, and do not truncate questions. "
                "If you do not know, say so quickly.\n\n"
                f"Context:\n- Seed: {selected_seed}\n- Tags: {', '.join(selected_tags)}\n- Horizon: {selected_horizon}\n"
                f"- Shortlisted questions:\n{chr(10).join(kept_preview_lines)}"
            )

            if "batch_chat_history" not in st.session_state:
                st.session_state["batch_chat_history"] = {}

            chat_key = f"batch-{selected_batch}"
            if chat_key not in st.session_state["batch_chat_history"]:
                st.session_state["batch_chat_history"][chat_key] = []

            col_chat_controls = st.columns([4, 1])
            with col_chat_controls[1]:
                if st.button("Reset chat", key=f"reset-{chat_key}", width="stretch"):
                    st.session_state["batch_chat_history"][chat_key] = []

            for msg in st.session_state["batch_chat_history"][chat_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input(
                "Write a message to discuss the kept questions.",
                key=f"input-{chat_key}",
            )

            if user_input:
                st.session_state["batch_chat_history"][chat_key].append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.markdown(user_input)

                or_messages: List[Dict[str, str]] = [
                    {"role": "system", "content": chat_system_prompt}
                ]
                history_for_model = st.session_state["batch_chat_history"][chat_key][-24:]
                or_messages.extend(history_for_model)

                if dry_run:
                    assistant_reply = (
                        "Dry-run mode: this would simulate a reply using your kept questions and message."
                    )
                elif not get_openrouter_key():
                    assistant_reply = (
                        "No OPENROUTER_API_KEY is configured. Add it in the sidebar or enable dry_run."
                    )
                else:
                    try:
                        raw_reply = call_openrouter_raw(
                            messages=or_messages,
                            model="openai/gpt-5.1",
                            max_tokens=1200,
                            temperature=0.4,
                        )
                        assistant_reply = raw_reply.strip()
                    except Exception as e:
                        assistant_reply = f"Error calling the chatbot: {e}"

                with st.chat_message("assistant"):
                    st.markdown(assistant_reply)

                st.session_state["batch_chat_history"][chat_key].append(
                    {"role": "assistant", "content": assistant_reply}
                )

        with tab_cards:
            st.subheader("Resolution cards generated for kept questions")
            for idx, run in enumerate(batch_results, start=1):
                res_run = run["result"]
                kept_questions = [e for e in res_run["initial"] if e.get("keep_final")]
                st.markdown(
                    f"**Batch b{idx}** — {len(kept_questions)} kept questions"
                )
                if not kept_questions:
                    st.caption("No questions are marked keep_final for this seed.")
                    continue
                card_store = res_run.get("resolution_cards", {})
                for e in kept_questions:
                    card_entry = card_store.get(e["id"], {})
                    st.markdown(f"**Resolution card – {e['id']} ({e['title']}):**")
                    st.markdown(
                        card_entry.get("card", "(no card generated)") or "(no card generated)"
                    )
else:
    if res is None:
        st.info(
            "Configure the number of mutated prompts, total N questions, K kept, "
            "set your seed, then click the button to run the pipeline."
        )
    else:
        main_model = res["models"]["main"]
        judge_model = res["models"]["judge"]
        seed = res["seed"]
        tags = res["tags"]
        horizon = res["horizon"]
        prompt_entries = res.get("prompts", [])
        initial_entries = res["initial"]

        tab_overview, tab_cards = st.tabs(
            ["Overview & questions", "Resolution cards (kept questions)"]
        )

        with tab_overview:
            st.subheader("Run summary")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"**Main model (mutations + sources + generation):** `{main_model}`"
                )
                st.markdown(f"**Judge model (strict resolvability):** `{judge_model}`")
            with col2:
                st.markdown(
                    f"**Total N proto-questions (requested):** {res['params']['n_initial_total']}"
                )
                st.markdown(f"**K target kept:** {res['params']['k_keep']}")
            with col3:
                st.markdown(f"**Number of mutated prompts:** {res['params']['n_mutations']}")
                st.markdown(f"**Horizon:** {horizon}")

            st.markdown("**Seed preview:**")
            st.caption(seed[:250] + ("..." if len(seed) > 250 else ""))

            st.subheader("Prompts (seed + mutations) and resolution hints")
            if prompt_entries:
                df_prompts = pd.DataFrame(prompt_entries)
                df_prompts_view = df_prompts.copy()
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
                st.dataframe(df_prompts_view, width="stretch")
            else:
                st.info("No prompts recorded.")

            st.subheader("Proto-questions (generation 0, across all prompts)")
            df_init = pd.DataFrame(initial_entries)
            df_init_for_download = None

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
                        "judge_verdict",
                        "judge_verdict_rationale",
                        "title",
                        "question",
                        "question_weight",
                        "type",
                        "inbound_outcome_count",
                        "options",
                        "group_variable",
                        "range_min",
                        "range_max",
                        "zero_point",
                        "open_lower_bound",
                        "open_upper_bound",
                        "unit",
                        "candidate_source",
                        "angle",
                        "judge_rationale",
                        "raw_question_block",
                    ]
                ].copy()

                st.caption(
                    "All generation-0 proto-questions with strict judge scores "
                    "(resolvability, info, decision impact, VOI, minutes_to_resolve) "
                    "and final selection flag (keep_final)."
                )
                st.dataframe(df_init_view, width="stretch")

                df_init_for_download = df_init_view.copy()
                df_init_for_download["seed"] = seed
                df_init_for_download["domain_tags"] = ", ".join(tags)
                df_init_for_download["resolution_horizon"] = horizon
            else:
                st.info("No proto-questions available.")

        with tab_overview:
            st.subheader("Initial resolution criteria & sources")
            if df_init.empty:
                st.caption("No proto-questions to show resolution criteria for.")
            else:
                prompt_sources_map = {
                    pe.get("prompt_id"): pe.get("sources", []) for pe in prompt_entries
                }

                for _, row in df_init.iterrows():
                    with st.container():
                        st.markdown(f"**{row['id']}** – *{row['title']}*")
                        st.markdown(row["question"])
                        with st.expander("Resolution criteria & explicit sources", expanded=False):
                            srcs = prompt_sources_map.get(row["parent_prompt_id"], [])
                            src_block = "\n".join(f"- {s}" for s in srcs) or "- (no sources returned)"
                            st.markdown(
                                f"- **Resolvability:** {row['judge_resolvability']}/5  \n"
                                f"- **Information value:** {row['judge_info']}/5  \n"
                                f"- **Decision impact:** {row['judge_decision_impact']:.2f}  \n"
                                f"- **VOI:** {row['judge_voi']:.2f}  \n"
                                f"- **Minutes to resolve:** {row['judge_minutes_to_resolve']:.1f}  \n"
                                f"- **Judge verdict:** {row.get('judge_verdict', '') or '(unspecified)'}  \n"
                                f"- **Verdict rationale:** {row.get('judge_verdict_rationale', '') or row['judge_rationale']}  \n"
                                f"- **Rationale:** {row['judge_rationale']}  \n"
                                f"- **Candidate source (generator hint):** {row['candidate_source']}  \n"
                                f"- **Sources to use for resolution:**\n{src_block}"
                            )

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

                st.markdown("**Raw question blocks (as parsed, before cleanup):**")
                if not df_init.empty:
                    for _, row in df_init[["id", "raw_question_block"]].iterrows():
                        if not row["raw_question_block"]:
                            continue
                        st.code(
                            f"{row['id']}\n{row['raw_question_block']}",
                            language="text",
                        )
                else:
                    st.caption("No parsed question blocks available.")

            kept_questions = [e for e in initial_entries if e.get("keep_final")]

            st.subheader("Download CSV")
            if df_init.empty or df_init_for_download is None:
                st.caption("No proto-questions available for download.")
            else:
                card_store = res.get("resolution_cards", {}) or {}
                df_init_for_download["resolution_card"] = df_init_for_download["id"].apply(
                    lambda q_id: card_store.get(q_id, {}).get("card", "")
                )
                csv_bytes = df_init_for_download.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download proto-questions + resolution cards (CSV)",
                    data=csv_bytes,
                    file_name="metaculus_proto_questions_with_cards.csv",
                    mime="text/csv",
                )

            st.subheader("Follow-up chat (GPT‑5 with kept questions)")
            if kept_questions:
                with st.expander("Kept questions (keep_final = True)", expanded=False):
                    for e in kept_questions:
                        st.markdown(
                            f"- **{e['id']}** – {e['title']}\n\n"
                            f"  {e['question']}"
                        )
            else:
                st.info(
                    "No questions are marked keep_final. The chat will still work but without injected context."
                )

            kept_preview_lines = [
                f"  • {e['id']} – {e['title']} – {e['question']}"
                for e in kept_questions[:8]
            ] or ["  • None"]

            chat_system_prompt = (
                "You are a fast, concise assistant for follow-up on kept forecasting questions. "
                "Reply in plain text (no markdown fences), keep the conversation going without resetting context, and do not truncate questions. "
                "If you do not know, say so quickly.\n\n"
                f"Context:\n- Seed: {seed}\n- Tags: {', '.join(tags)}\n- Horizon: {horizon}\n"
                f"- Shortlisted questions:\n{chr(10).join(kept_preview_lines)}"
            )

            if "refine_chat_history" not in st.session_state:
                st.session_state["refine_chat_history"] = []
            if "refine_chat_context" not in st.session_state:
                st.session_state["refine_chat_context"] = chat_system_prompt

            col_chat_controls = st.columns([4, 1])
            with col_chat_controls[1]:
                if st.button("Reset chat", width="stretch"):
                    st.session_state["refine_chat_history"] = []

            # Always show the running thread; do not reset it on rerun
            for msg in st.session_state["refine_chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input("Write a message to discuss the kept questions.")

            if user_input:
                st.session_state["refine_chat_history"].append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.markdown(user_input)

                or_messages: List[Dict[str, str]] = [
                    {"role": "system", "content": chat_system_prompt}
                ]
                # Trim context for faster responses while keeping conversation continuity
                history_for_model = st.session_state["refine_chat_history"][-24:]
                or_messages.extend(history_for_model)

                if dry_run:
                    assistant_reply = (
                        "Dry-run mode: this would simulate a reply using your kept questions and message."
                    )
                elif not get_openrouter_key():
                    assistant_reply = (
                        "No OPENROUTER_API_KEY is configured. Add it in the sidebar or enable dry_run."
                    )
                else:
                    try:
                        raw_reply = call_openrouter_raw(
                            messages=or_messages,
                            model="openai/gpt-5.1",
                            max_tokens=1200,
                            temperature=0.4,
                        )
                        assistant_reply = raw_reply.strip()
                    except Exception as e:
                        assistant_reply = f"Error calling the chatbot: {e}"

                with st.chat_message("assistant"):
                    st.markdown(assistant_reply)

                st.session_state["refine_chat_history"].append(
                    {"role": "assistant", "content": assistant_reply}
                )

        with tab_cards:
            st.subheader("Resolution cards generated for kept questions")
            kept_questions = [e for e in initial_entries if e.get("keep_final")]

            if not kept_questions:
                st.info(
                    "No questions are marked keep_final. Run the pipeline and return here to view resolution cards."
                )
            else:
                card_store = st.session_state.get("resolution_cards", {})
                for e in kept_questions:
                    card_entry = card_store.get(e["id"], {})
                    st.markdown(f"**Resolution card – {e['id']} ({e['title']}):**")
                    st.markdown(
                        card_entry.get("card", "(no card generated)") or "(no card generated)"
                    )
