"""Core milestone implementations for the Student Feedback Summarizer (lfllm-lflora)."""
from __future__ import annotations

from app.dataset import get_synthetic_dataset, split_dataset
from app.evaluate import evaluate_batch
from app.lora import (
    TrainingResult,
    build_tiny_lora_model,
    make_input_batch,
    train_lora,
    verify_merge_equivalence,
)
from app.summarizer import SummaryResult, summarize_with_lora


# --- Milestone 1: Prepare a small dataset of synthetic student feedback and summary pairs.
def milestone_1(val_ratio: float = 0.2) -> dict:
    """Return synthetic student feedback-summary dataset split for lfllm-lflora training."""
    records = get_synthetic_dataset()
    train_records, val_records = split_dataset(records, val_ratio=val_ratio)
    return {
        "records": records,
        "train": train_records,
        "val": val_records,
        "count": len(records),
    }


# --- Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
def milestone_2(
    rank: int = 4,
    alpha: float = 8.0,
    epochs: int = 6,
    lr: float = 5e-3,
    seed: int = 42,
) -> dict:
    """Inject LoRA adapters into the base LLM and fine-tune on the synthetic feedback dataset."""
    data = milestone_1()
    train_texts = [r["text"] for r in data["train"]]
    all_texts = [r["text"] for r in data["records"]]

    model, tokenizer, replaced = build_tiny_lora_model(
        texts=all_texts,
        rank=rank,
        alpha=alpha,
        seed=seed,
    )
    result: TrainingResult = train_lora(
        model=model,
        tokenizer=tokenizer,
        texts=train_texts,
        epochs=epochs,
        lr=lr,
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "replaced_modules": replaced,
        "training": result,
        "dataset": data,
    }


# --- Milestone 3: Develop a Streamlit UI to upload feedback and trigger summarization.
def milestone_3(
    feedback_list: list[str],
    trained_bundle: dict | None = None,
    use_external_llm: bool = False,
) -> list[SummaryResult]:
    """Summarize one or more student feedback comments using the LoRA fine-tuned model."""
    bundle = trained_bundle if trained_bundle is not None else milestone_2(epochs=4)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    return [
        summarize_with_lora(fb, model=model, tokenizer=tokenizer, use_external_llm=use_external_llm)
        for fb in feedback_list
        if fb and fb.strip()
    ]


# --- Milestone 4: Evaluate the quality of generated summaries against original feedback.
def milestone_4(trained_bundle: dict | None = None) -> dict:
    """Evaluate generated summaries on the validation split using ROUGE-1/2/L and coverage."""
    bundle = trained_bundle if trained_bundle is not None else milestone_2(epochs=4)
    val_records = bundle["dataset"]["val"]
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    eval_items = []
    for rec in val_records:
        res = summarize_with_lora(rec["feedback"], model=model, tokenizer=tokenizer)
        eval_items.append(
            {
                "id": rec["id"],
                "course": rec["course"],
                "feedback": rec["feedback"],
                "generated_summary": res.summary,
                "reference_summary": rec["summary"],
            }
        )

    metrics = evaluate_batch(eval_items)
    metrics["samples"] = eval_items
    return metrics


# --- Milestone 5: Implement adapter merging for inference.
def milestone_5(trained_bundle: dict | None = None) -> dict:
    """Merge LoRA adapters into base model weights and verify numerical equivalence."""
    bundle = trained_bundle if trained_bundle is not None else milestone_2(epochs=4)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    sample_text = bundle["dataset"]["val"][0]["prompt"]
    sample_ids = make_input_batch(tokenizer.encode(sample_text, max_length=64))

    report = verify_merge_equivalence(model, sample_ids)
    report["model"] = model
    report["tokenizer"] = tokenizer
    return report
