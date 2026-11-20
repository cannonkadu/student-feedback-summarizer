"""Unit tests for all 5 milestones of the Student Feedback Summarizer."""
from app.core import milestone_1, milestone_2, milestone_3, milestone_4, milestone_5
from app.dataset import parse_uploaded_feedback
from app.evaluate import evaluate_summary
from app.lora import LoRALinear


# Milestone 1: Prepare a small dataset of synthetic student feedback and summary pairs.
def test_milestone_1():
    data = milestone_1(val_ratio=0.2)
    assert data["count"] >= 20
    assert len(data["train"]) > len(data["val"]) >= 1
    first = data["records"][0]
    for key in ("id", "course", "category", "feedback", "summary", "prompt", "text"):
        assert key in first and first[key]
    assert "### Feedback:" in first["prompt"]
    assert "### Summary:" in first["text"]


# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
def test_milestone_2():
    bundle = milestone_2(rank=4, alpha=8.0, epochs=6, lr=5e-3, seed=42)
    model = bundle["model"]
    training = bundle["training"]

    # Verify LoRA adapters were injected and base weights are frozen
    assert len(bundle["replaced_modules"]) > 0
    assert any(isinstance(m, LoRALinear) for m in model.modules())
    assert 0 < training.param_stats["trainable_pct"] < 25.0

    # Verify training actually reduced loss
    assert len(training.loss_history) == 6
    assert training.final_loss < training.initial_loss


# Milestone 3: Develop a Streamlit UI to upload feedback and trigger summarization.
def test_milestone_3():
    csv_bytes = b"course,feedback\nCS101,The live coding demos were great but scrolling was too fast.\n"
    parsed = parse_uploaded_feedback("comments.csv", csv_bytes)
    assert len(parsed) == 1

    bundle = milestone_2(epochs=3)
    results = milestone_3(parsed, trained_bundle=bundle)
    assert len(results) == 1
    res = results[0]
    assert len(res.summary) > 0
    assert res.sentiment in ("Positive", "Needs Attention", "Mixed", "Neutral")
    assert len(res.themes) >= 1
    assert len(res.action_item) > 0


# Milestone 4: Evaluate the quality of generated summaries against original feedback.
def test_milestone_4():
    bundle = milestone_2(epochs=3)
    report = milestone_4(trained_bundle=bundle)
    assert report["count"] == len(bundle["dataset"]["val"])
    assert 0.0 <= report["avg_rouge1_f1"] <= 1.0
    assert 0.0 <= report["avg_rouge2_f1"] <= 1.0
    assert 0.0 <= report["avg_rougeL_f1"] <= 1.0
    assert report["avg_key_term_coverage"] > 0.0

    # Sanity-check exact match ROUGE = 1.0
    perfect = evaluate_summary("Clear slides helped.", "Clear slides helped.", "Clear slides helped.")
    assert perfect["rouge1_f1"] == 1.0
    assert perfect["rougeL_f1"] == 1.0


# Milestone 5: Implement adapter merging for inference.
def test_milestone_5():
    bundle = milestone_2(epochs=4)
    report = milestone_5(trained_bundle=bundle)

    assert report["merged_layers"] > 0
    assert report["is_equivalent"] is True
    assert report["max_abs_diff"] < 1e-4
    # Confirm all LoRALinear wrappers were unwrapped into standard layers
    assert not any(isinstance(m, LoRALinear) for m in report["model"].modules())
