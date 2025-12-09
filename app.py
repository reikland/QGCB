import json as _json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from qgcb import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAIN_MODEL,
    JudgeKeepResult,
    ProtoQuestion,
    call_openrouter_raw,
    find_resolution_sources_for_prompt,
    generate_initial_questions,
    get_openrouter_key,
    judge_initial_questions,
    mutate_seed_prompt,
    select_top_k,
)

# ============================================================
# 6. STREAMLIT UI
# ============================================================

if "evo_result" not in st.session_state:
    st.session_state["evo_result"] = None

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

    # Critères de résolution initiaux (sources explicites)
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
                        f"- **Rationale:** {row['judge_rationale']}  \n"
                        f"- **Candidate source (generator hint):** {row['candidate_source']}  \n"
                        f"- **Sources to use for resolution:**\n{src_block}"
                    )

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

    # ---------------------- Chat simple avec questions conservées ----------------------
    st.subheader("Chat de suivi (GPT‑5 avec questions conservées)")

    kept_questions = [e for e in initial_entries if e.get("keep_final")]
    if kept_questions:
        with st.expander("Questions conservées (keep_final = True)", expanded=False):
            for e in kept_questions:
                st.markdown(
                    f"- **{e['id']}** – {e['title']}\n\n"
                    f"  {e['question']}"
                )
    else:
        st.info(
            "Aucune question n'est marquée keep_final. Le chat fonctionnera quand même, "
            "mais sans contexte joint."
        )

    chat_system_prompt = f"""
You are a fresh GPT-5.1 chat instance dedicated to quick follow-up with the user.
Always answer in plain text (no JSON, no markdown code blocks).

Static context (do not repeat unless the user asks):
- Seed: {seed}
- Tags: {', '.join(tags)}
- Horizon: {horizon}
- Shortlisted questions (keep_final):
{chr(10).join([f"  • {e['id']} – {e['title']} – {e['question']}" for e in kept_questions]) if kept_questions else '  • None'}
""".strip()

    if "refine_chat_history" not in st.session_state:
        st.session_state["refine_chat_history"] = []

    for msg in st.session_state["refine_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Écrivez un message pour discuter des questions conservées.")

    if user_input:
        st.session_state["refine_chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        or_messages: List[Dict[str, str]] = [{"role": "system", "content": chat_system_prompt}]
        or_messages.extend(st.session_state["refine_chat_history"])

        if dry_run:
            assistant_reply = (
                "Mode dry-run : je simulerais ici une réponse basée sur vos questions "
                "conservées et votre message."
            )
        elif not get_openrouter_key():
            assistant_reply = (
                "Aucune clé OPENROUTER_API_KEY n'est configurée. Ajoutez-la dans le "
                "panneau latéral ou activez le mode dry_run."
            )
        else:
            try:
                raw_reply = call_openrouter_raw(
                    messages=or_messages,
                    model="openai/gpt-5.1",
                    max_tokens=1200,
                    temperature=0.5,
                )
                assistant_reply = raw_reply.strip()
            except Exception as e:
                assistant_reply = f"Erreur lors de l'appel du chatbot : {e}"

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

        st.session_state["refine_chat_history"].append({"role": "assistant", "content": assistant_reply})

else:
    st.info(
        "Configure the number of mutated prompts, total N questions, K kept, "
        "set your seed, then click the button to run the pipeline."
    )
