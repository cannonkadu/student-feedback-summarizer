"""Summarization & curriculum-insight engine powered by the LoRA fine-tuned model."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app import llm
from app.dataset import format_for_lora
from app.lora import TORCH_AVAILABLE, LoRALinear, SimpleWordTokenizer, make_input_batch

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

POSITIVE_CUES = {
    "great", "awesome", "engaging", "helpful", "clear", "patient", "effective",
    "highlight", "fantastic", "thrilling", "fair", "responsive", "valuable", "love",
    "interesting", "unforgettable", "clicked", "click", "strong",
}
NEGATIVE_CUES = {
    "fast", "hard", "longer", "unclear", "rushed", "inconsistent", "packed",
    "wasted", "muffled", "contradictory", "abrupt", "crunch", "confusion", "slow",
    "needed", "missing", "problem", "stress", "never",
}
THEME_KEYWORDS: dict[str, set[str]] = {
    "Lectures & Delivery": {"lecture", "lectures", "slides", "professor", "instructor", "audio", "whiteboard", "demos", "coding", "polls"},
    "Labs & Assignments": {"lab", "labs", "homework", "assignment", "assignments", "starter", "test", "spec", "specification", "docker", "ctf"},
    "Pacing & Workload": {"pacing", "weeks", "rushed", "hours", "workload", "crunch", "deadline", "deadlines", "schedule", "abrupt"},
    "Grading & Rubrics": {"grading", "grade", "grades", "rubric", "exam", "midterm", "quiz", "ta", "tas", "turnaround"},
    "Support & Resources": {"office", "piazza", "forum", "cheat", "reading", "prerequisites", "pytorch", "guides", "support"},
}


@dataclass
class SummaryResult:
    feedback: str
    summary: str
    sentiment: str
    themes: list[str]
    action_item: str
    engine: str
    model_perplexity: float | None = None


def detect_sentiment(text: str) -> str:
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & POSITIVE_CUES)
    neg = len(words & NEGATIVE_CUES)
    if pos > 0 and neg > 0:
        return "Mixed"
    if neg > pos:
        return "Needs Attention"
    if pos > neg:
        return "Positive"
    return "Neutral"


def detect_themes(text: str) -> list[str]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    found = [theme for theme, kws in THEME_KEYWORDS.items() if words & kws]
    return found or ["General Course Experience"]


def _suggest_action(feedback: str, sentiment: str, themes: list[str]) -> str:
    lower = feedback.lower()
    if "rubric" in lower or "inconsistent" in lower:
        return "Publish a standardized grading rubric and calibrate TA scoring."
    if "slide" in lower or "advance" in lower or "before class" in lower:
        return "Release lecture slides and materials at least 24 hours ahead of class."
    if "fast" in lower or "rushed" in lower or "pacing" in lower or "abrupt" in lower:
        return "Rebalance module pacing and add a short review checkpoint before advanced topics."
    if "office hour" in lower or "packed" in lower or "forum" in lower:
        return "Add pre-deadline TA office hours or a structured Q&A thread."
    if "audio" in lower or "microphone" in lower or "muffled" in lower:
        return "Upgrade lecture capture audio with a lapel microphone or digital tablet."
    if "starter" in lower or "setup" in lower or "docker" in lower or "spec" in lower:
        return "Provide a verified starter template, setup script, and unambiguous lab spec."
    if sentiment == "Positive":
        return f"Retain and highlight this strength in {themes[0].lower()} for future offerings."
    return f"Review student comments on {themes[0].lower()} and schedule a targeted curriculum adjustment."


def _score_sentence_with_lora(
    model: Any,
    tokenizer: SimpleWordTokenizer,
    sentence: str,
) -> float:
    """Compute mean token log-probability of a sentence under the LoRA-adapted causal LM."""
    ids = tokenizer.encode(sentence, max_length=64)
    if len(ids) < 2:
        return 0.0
    if TORCH_AVAILABLE:
        with torch.no_grad():
            device = next(model.parameters()).device
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(tensor[:, :-1])
            logits = out.logits if hasattr(out, "logits") else out
            log_probs = F.log_softmax(logits, dim=-1)
            targets = tensor[:, 1:]
            token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            return float(token_lp.mean().item())

    arr = make_input_batch(ids)
    logits = model(arr[:, :-1])
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    targets = arr[:, 1:]
    token_lp = log_probs[0, np.arange(targets.shape[1]), targets[0]]
    return float(np.mean(token_lp))


def summarize_with_lora(
    feedback: str,
    model: Any,
    tokenizer: SimpleWordTokenizer,
    use_external_llm: bool = False,
) -> SummaryResult:
    """Generate a concise curriculum-improvement summary using the LoRA-adapted (or merged) model."""
    clean_fb = feedback.strip()
    if not clean_fb:
        raise ValueError("Feedback text must not be empty.")

    sentiment = detect_sentiment(clean_fb)
    themes = detect_themes(clean_fb)
    action_item = _suggest_action(clean_fb, sentiment, themes)

    # Optional external LLM path if user explicitly enables it and configured .env
    if use_external_llm:
        _, _, configured_model = llm.settings()
        if configured_model:
            prompt = format_for_lora(clean_fb)
            reply = llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an EdTech curriculum analyst. Summarize the student feedback in "
                            "one concise, actionable sentence (under 25 words) for the course instructor."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=80,
            )
            return SummaryResult(
                feedback=clean_fb,
                summary=reply.strip(),
                sentiment=sentiment,
                themes=themes,
                action_item=action_item,
                engine=f"External LLM ({configured_model})",
                model_perplexity=None,
            )

    model.eval()
    prompt_text = format_for_lora(clean_fb)
    prompt_lp = _score_sentence_with_lora(model, tokenizer, prompt_text)
    perplexity = round(math.exp(min(-prompt_lp, 12.0)), 2)

    # Split feedback into sentences and rank with LoRA model confidence + instructional salience
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_fb) if s.strip()]
    if len(sentences) == 1:
        summary = sentences[0]
    else:
        scored: list[tuple[float, str]] = []
        for idx, sent in enumerate(sentences):
            lora_lp = _score_sentence_with_lora(model, tokenizer, sent)
            words = set(re.findall(r"[a-z]+", sent.lower()))
            cue_bonus = 0.35 * len(words & (POSITIVE_CUES | NEGATIVE_CUES))
            # Prefer actionable / concluding sentences slightly
            pos_bonus = 0.15 if idx == len(sentences) - 1 else 0.0
            scored.append((lora_lp + cue_bonus + pos_bonus, sent))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = scored[0][1]
        # Compress if the selected sentence is long
        words_list = best.split()
        if len(words_list) > 26:
            summary = " ".join(words_list[:24]).rstrip(",;:") + "."
        else:
            summary = best

    # Determine whether model is merged or active LoRA
    from app.lora import LoRALinear

    has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    engine = "LoRA Adapter (Active)" if has_lora else "LoRA Merged Model (Zero-Overhead)"

    return SummaryResult(
        feedback=clean_fb,
        summary=summary,
        sentiment=sentiment,
        themes=themes,
        action_item=action_item,
        engine=engine,
        model_perplexity=perplexity,
    )
