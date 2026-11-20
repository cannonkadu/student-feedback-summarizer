"""Streamlit entry point: `streamlit run app/main.py`"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Ensure repo root is on sys.path when invoked via `streamlit run app/main.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import milestone_1, milestone_4
from app.dataset import (
    export_jsonl,
    format_for_lora,
    parse_uploaded_feedback,
    split_dataset,
)
from app.evaluate import evaluate_summary
from app.lora import (
    LoRALinear,
    build_tiny_lora_model,
    count_parameters,
    make_input_batch,
    save_adapter,
    train_lora,
    verify_merge_equivalence,
)
from app.storage import Store
from app.summarizer import summarize_with_lora

st.set_page_config(
    page_title="Student Feedback Summarizer | LoRA EdTech Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark-modern EdTech styling
st.markdown(
    """
    <style>
    .hero-card {
        background: linear-gradient(135deg, #161b2e 0%, #1d1536 55%, #0f2933 100%);
        border: 1px solid rgba(139, 92, 246, 0.35);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(124, 58, 237, 0.25);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.45);
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .hero-badge-teal {
        background: rgba(20, 184, 166, 0.2);
        color: #5eead4;
        border-color: rgba(20, 184, 166, 0.4);
    }
    .summary-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_session() -> None:
    if "dataset_records" not in st.session_state:
        st.session_state.dataset_records = milestone_1()["records"]
    if "lora_model" not in st.session_state:
        texts = [r["text"] for r in st.session_state.dataset_records]
        model, tokenizer, replaced = build_tiny_lora_model(texts=texts, rank=4, alpha=8.0)
        st.session_state.lora_model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.replaced_modules = replaced
        st.session_state.training_result = None
        st.session_state.merge_report = None
        st.session_state.eval_report = None
    if "store" not in st.session_state:
        st.session_state.store = Store()


_init_session()

# Hero header
st.markdown(
    """
    <div class="hero-card">
        <div style="margin-bottom: 0.5rem;">
            <span class="hero-badge">EdTech</span>
            <span class="hero-badge hero-badge-teal">LoRA from Scratch (lfllm-lflora)</span>
            <span class="hero-badge">Runs on Your Laptop</span>
        </div>
        <h1 style="margin: 0.2rem 0 0.4rem 0; color: #f8fafc;">🎓 Student Feedback Summarizer</h1>
        <p style="margin: 0; color: #cbd5e1; font-size: 1.02rem;">
            Fine-tune a language model with Low-Rank Adaptation (LoRA) on course feedback, evaluate summary quality with ROUGE,
            merge adapters for zero-overhead inference, and surface actionable curriculum insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar status panel
with st.sidebar:
    st.header("Model Status")
    model = st.session_state.lora_model
    has_active_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    stats = count_parameters(model)

    st.metric("Adapter State", "Active LoRA" if has_active_lora else "Merged (Deployed)")
    st.metric("Total Parameters", f"{stats['total']:,}")
    st.metric("Trainable Parameters", f"{stats['trainable']:,} ({stats['trainable_pct']}%)")
    st.metric("Dataset Pairs", len(st.session_state.dataset_records))

    if st.session_state.training_result is not None:
        tr = st.session_state.training_result
        st.success(f"Trained {tr.epochs} epochs | Loss: {tr.initial_loss:.3f} → {tr.final_loss:.3f}")
    else:
        st.info("Tip: Fine-tune the LoRA adapter in Tab 2, then merge in Tab 5.")

    st.divider()
    use_external_llm = st.toggle(
        "Use External LLM (.env)",
        value=False,
        help="Optional: route summarization to Ollama/Groq/Gemini configured in .env.",
    )

# Main 5-tab workspace
tab_sum, tab_train, tab_data, tab_eval, tab_merge = st.tabs(
    [
        "✨ Summarize Feedback",
        "⚡ LoRA Fine-Tuning Studio",
        "📚 Dataset Explorer",
        "📊 Evaluation & ROUGE Lab",
        "🚀 Adapter Merging",
    ]
)

# --- TAB 1: Summarize Feedback ---
with tab_sum:
    st.subheader("Upload or Paste Student Course Feedback")
    col_in, col_up = st.columns([3, 2])

    with col_in:
        sample_default = (
            "The live coding demos during lectures were super engaging and helped me understand loops and functions. "
            "However, the professor sometimes scrolled through the IDE too fast, making it hard to follow along on our laptops."
        )
        pasted_text = st.text_area(
            "Student feedback (one comment per line for batch mode)",
            value=sample_default,
            height=150,
        )

    with col_up:
        uploaded = st.file_uploader(
            "Or upload feedback file (.csv, .txt, .json)",
            type=["csv", "txt", "json"],
        )
        if uploaded is not None:
            parsed = parse_uploaded_feedback(uploaded.name, uploaded.getvalue())
            st.success(f"Loaded {len(parsed)} feedback comments from `{uploaded.name}`.")
        else:
            parsed = []

    if st.button("Generate Summaries", type="primary", use_container_width=True):
        comments = parsed if parsed else [line.strip() for line in pasted_text.splitlines() if line.strip()]
        if not comments:
            st.warning("Please paste at least one feedback comment or upload a file.")
        else:
            results = []
            for fb in comments:
                res = summarize_with_lora(
                    fb,
                    model=st.session_state.lora_model,
                    tokenizer=st.session_state.tokenizer,
                    use_external_llm=use_external_llm,
                )
                results.append(res)

            st.session_state.store.put(
                "last_run",
                [
                    {
                        "feedback": r.feedback,
                        "summary": r.summary,
                        "sentiment": r.sentiment,
                        "themes": r.themes,
                        "action_item": r.action_item,
                        "engine": r.engine,
                    }
                    for r in results
                ],
            )

            st.markdown(f"### Generated Summaries ({len(results)})")
            for idx, r in enumerate(results, start=1):
                metrics = evaluate_summary(r.feedback, r.summary)
                with st.container(border=True):
                    top_l, top_r = st.columns([3, 1])
                    top_l.markdown(f"**#{idx} Concise Summary:** {r.summary}")
                    top_r.caption(f"Engine: `{r.engine}`")

                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**Sentiment:** `{r.sentiment}`")
                    c2.markdown(f"**Themes:** {', '.join(r.themes)}")
                    c3.markdown(f"**Compression:** `{metrics['compression_ratio']:.2f}x`")
                    st.info(f"**Recommended Curriculum Action:** {r.action_item}")
                    with st.expander("Original Student Feedback"):
                        st.write(r.feedback)

            export_payload = json.dumps(
                [
                    {
                        "feedback": r.feedback,
                        "summary": r.summary,
                        "sentiment": r.sentiment,
                        "themes": r.themes,
                        "action_item": r.action_item,
                    }
                    for r in results
                ],
                indent=2,
            )
            st.download_button(
                "Download Summaries (JSON)",
                data=export_payload,
                file_name="student_feedback_summaries.json",
                mime="application/json",
            )

# --- TAB 2: LoRA Fine-Tuning Studio ---
with tab_train:
    st.subheader("Configure & Train LoRA Adapters (`lfllm-lflora`)")
    c1, c2, c3, c4 = st.columns(4)
    rank = c1.select_slider("LoRA Rank (r)", options=[2, 4, 8, 16], value=4)
    alpha = c2.select_slider("LoRA Alpha (α)", options=[4.0, 8.0, 16.0, 32.0], value=8.0)
    epochs = c3.slider("Epochs", min_value=2, max_value=20, value=8)
    lr = c4.select_slider("Learning Rate", options=[1e-3, 3e-3, 5e-3, 1e-2], value=5e-3)

    if st.button("Inject LoRA & Start Fine-Tuning", type="primary"):
        records = st.session_state.dataset_records
        train_recs, _ = split_dataset(records, val_ratio=0.2)
        all_texts = [r["text"] for r in records]
        train_texts = [r["text"] for r in train_recs]

        model, tokenizer, replaced = build_tiny_lora_model(
            texts=all_texts,
            rank=rank,
            alpha=alpha,
        )
        progress = st.progress(0.0, text="Training LoRA adapters...")

        def _cb(ep: int, total: int, loss_val: float) -> None:
            progress.progress(ep / total, text=f"Epoch {ep}/{total} — Loss: {loss_val:.4f}")

        tr = train_lora(
            model=model,
            tokenizer=tokenizer,
            texts=train_texts,
            epochs=epochs,
            lr=lr,
            progress_callback=_cb,
        )
        progress.empty()

        st.session_state.lora_model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.replaced_modules = replaced
        st.session_state.training_result = tr
        st.session_state.merge_report = None

        st.success(
            f"LoRA training complete in {tr.elapsed_sec}s! "
            f"Loss dropped from {tr.initial_loss:.4f} → {tr.final_loss:.4f} "
            f"({tr.param_stats['trainable_pct']}% trainable parameters)."
        )

    if st.session_state.training_result is not None:
        tr = st.session_state.training_result
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Initial Loss", f"{tr.initial_loss:.4f}")
        m2.metric("Final Loss", f"{tr.final_loss:.4f}", delta=f"{tr.final_loss - tr.initial_loss:.4f}")
        m3.metric("Injected Layers", len(tr.replaced_modules))
        m4.metric("Trainable %", f"{tr.param_stats['trainable_pct']}%")

        st.line_chart({"Training Loss": tr.loss_history})

        if st.button("Save LoRA Adapter Weights (.pt)"):
            out_path = save_adapter(st.session_state.lora_model, "data/lora_adapter.pt")
            st.success(f"Saved lightweight LoRA adapter weights to `{out_path}`.")

# --- TAB 3: Dataset Explorer ---
with tab_data:
    st.subheader("Synthetic Student Feedback & Summary Dataset")
    records = st.session_state.dataset_records
    categories = ["All"] + sorted({r["category"] for r in records})
    selected_cat = st.selectbox("Filter by Category", categories)
    filtered = records if selected_cat == "All" else [r for r in records if r["category"] == selected_cat]

    st.dataframe(
        [
            {
                "ID": r["id"],
                "Course": r["course"],
                "Category": r["category"],
                "Sentiment": r["sentiment"],
                "Feedback": r["feedback"],
                "Reference Summary": r["summary"],
            }
            for r in filtered
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Add a New Feedback-Summary Pair"):
        with st.form("add_pair_form"):
            new_course = st.text_input("Course", value="CS301 - AI Systems")
            new_cat = st.selectbox("Category", ["Lectures", "Labs", "Assignments", "Pacing", "Grading", "Support", "Projects"])
            new_fb = st.text_area("Student Feedback")
            new_sum = st.text_input("Reference Summary")
            submitted = st.form_submit_button("Add to Dataset")
            if submitted and new_fb.strip() and new_sum.strip():
                new_id = f"fb-{len(records) + 1:02d}"
                st.session_state.dataset_records.append(
                    {
                        "id": new_id,
                        "course": new_course.strip(),
                        "category": new_cat,
                        "sentiment": "mixed",
                        "feedback": new_fb.strip(),
                        "summary": new_sum.strip(),
                        "prompt": format_for_lora(new_fb.strip()),
                        "text": format_for_lora(new_fb.strip(), new_sum.strip()),
                    }
                )
                st.success(f"Added `{new_id}` to training dataset!")
                st.rerun()

    st.download_button(
        "Export Dataset as JSONL",
        data=export_jsonl(records),
        file_name="student_feedback_lora_dataset.jsonl",
        mime="application/jsonl",
    )

# --- TAB 4: Evaluation & ROUGE Lab ---
with tab_eval:
    st.subheader("Automated Summary Quality Evaluation (ROUGE-1 / ROUGE-2 / ROUGE-L)")
    if st.button("Run Validation Evaluation", type="primary"):
        records = st.session_state.dataset_records
        train_recs, val_recs = split_dataset(records, val_ratio=0.2)
        bundle = {
            "model": st.session_state.lora_model,
            "tokenizer": st.session_state.tokenizer,
            "dataset": {"records": records, "train": train_recs, "val": val_recs},
        }
        st.session_state.eval_report = milestone_4(trained_bundle=bundle)

    if st.session_state.eval_report is not None:
        rep = st.session_state.eval_report
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("ROUGE-1 F1", f"{rep['avg_rouge1_f1']:.3f}")
        e2.metric("ROUGE-2 F1", f"{rep['avg_rouge2_f1']:.3f}")
        e3.metric("ROUGE-L F1", f"{rep['avg_rougeL_f1']:.3f}")
        e4.metric("Compression", f"{rep['avg_compression_ratio']:.2f}x")
        e5.metric("Key-Term Coverage", f"{rep['avg_key_term_coverage']:.2%}")

        rows = []
        for sample, score in zip(rep["samples"], rep["per_item"]):
            rows.append(
                {
                    "ID": sample["id"],
                    "Course": sample["course"],
                    "Generated Summary": sample["generated_summary"],
                    "Reference Summary": sample["reference_summary"],
                    "ROUGE-1": score["rouge1_f1"],
                    "ROUGE-L": score["rougeL_f1"],
                    "Coverage": score["key_term_coverage"],
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

# --- TAB 5: Adapter Merging & Deployment ---
with tab_merge:
    st.subheader("Merge LoRA Adapters into Base Model for Zero-Overhead Inference")
    st.write(
        "Merging fuses each adapter's low-rank update $\\Delta W = \\frac{\\alpha}{r} B A$ directly into the base linear weight $W_0$ "
        "and unwraps `LoRALinear` back into standard `nn.Linear` layers."
    )

    has_lora = any(isinstance(m, LoRALinear) for m in st.session_state.lora_model.modules())
    if not has_lora:
        st.success("Adapters are already merged into the base model! Ready for zero-overhead inference.")
    elif st.button("Merge Adapters & Verify Equivalence", type="primary"):
        sample_prompt = st.session_state.dataset_records[0]["prompt"]
        sample_ids = make_input_batch(
            st.session_state.tokenizer.encode(sample_prompt, max_length=64)
        )
        report = verify_merge_equivalence(st.session_state.lora_model, sample_ids)
        st.session_state.merge_report = report
        st.rerun()

    if st.session_state.merge_report is not None:
        mr = st.session_state.merge_report
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Merged Layers", mr["merged_layers"])
        m2.metric("Max Logit Diff", f"{mr['max_abs_diff']:.2e}")
        m3.metric("Unmerged Latency", f"{mr['unmerged_ms']} ms")
        m4.metric("Merged Latency", f"{mr['merged_ms']} ms")
        if mr["is_equivalent"]:
            st.success("Verified: Merged model outputs are numerically equivalent to unmerged LoRA outputs!")
        else:
            st.error("Warning: Output difference exceeded tolerance.")
