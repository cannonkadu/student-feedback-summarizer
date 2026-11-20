"""Milestone 4: Automated evaluation metrics (ROUGE-1, ROUGE-2, ROUGE-L, compression, coverage)."""
from __future__ import annotations

import re
from collections import Counter

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "them", "there", "these", "this", "to", "was", "we", "were", "with",
    "would", "could", "should", "very", "really", "also", "some", "more", "than",
}


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(prediction: str, reference: str, n: int = 1) -> dict[str, float]:
    """Compute ROUGE-N precision, recall, and F1 between prediction and reference."""
    pred_ngrams = _ngrams(tokenize_words(prediction), n)
    ref_ngrams = _ngrams(tokenize_words(reference), n)

    if not pred_ngrams or not ref_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = sum((pred_ngrams & ref_ngrams).values())
    precision = overlap / sum(pred_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for tok_a in a:
        prev = 0
        for j, tok_b in enumerate(b, start=1):
            temp = dp[j]
            if tok_a == tok_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1 using Longest Common Subsequence."""
    pred_tokens = tokenize_words(prediction)
    ref_tokens = tokenize_words(reference)
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compression_ratio(summary: str, feedback: str) -> float:
    """Ratio of summary word count to original feedback word count."""
    fb_len = len(tokenize_words(feedback))
    if fb_len == 0:
        return 0.0
    return round(len(tokenize_words(summary)) / fb_len, 4)


def key_term_coverage(summary: str, feedback: str) -> float:
    """Fraction of salient (non-stopword) terms in the summary grounded in the feedback."""
    fb_terms = {t for t in tokenize_words(feedback) if t not in STOPWORDS and len(t) > 2}
    sum_terms = {t for t in tokenize_words(summary) if t not in STOPWORDS and len(t) > 2}
    if not sum_terms:
        return 0.0
    return round(len(sum_terms & fb_terms) / len(sum_terms), 4)


def evaluate_summary(
    feedback: str,
    generated_summary: str,
    reference_summary: str | None = None,
) -> dict[str, float]:
    """Evaluate a single generated summary against original feedback and optional reference."""
    ref = reference_summary if reference_summary is not None else feedback
    r1 = rouge_n(generated_summary, ref, n=1)
    r2 = rouge_n(generated_summary, ref, n=2)
    rl = rouge_l(generated_summary, ref)
    return {
        "rouge1_f1": r1["f1"],
        "rouge2_f1": r2["f1"],
        "rougeL_f1": rl["f1"],
        "compression_ratio": compression_ratio(generated_summary, feedback),
        "key_term_coverage": key_term_coverage(generated_summary, feedback),
    }


def evaluate_batch(items: list[dict]) -> dict:
    """Evaluate a list of {feedback, generated_summary, reference_summary?} dicts."""
    if not items:
        return {
            "count": 0,
            "avg_rouge1_f1": 0.0,
            "avg_rouge2_f1": 0.0,
            "avg_rougeL_f1": 0.0,
            "avg_compression_ratio": 0.0,
            "avg_key_term_coverage": 0.0,
            "per_item": [],
        }

    per_item = [
        evaluate_summary(
            feedback=item["feedback"],
            generated_summary=item["generated_summary"],
            reference_summary=item.get("reference_summary"),
        )
        for item in items
    ]
    n = len(per_item)
    return {
        "count": n,
        "avg_rouge1_f1": round(sum(x["rouge1_f1"] for x in per_item) / n, 4),
        "avg_rouge2_f1": round(sum(x["rouge2_f1"] for x in per_item) / n, 4),
        "avg_rougeL_f1": round(sum(x["rougeL_f1"] for x in per_item) / n, 4),
        "avg_compression_ratio": round(sum(x["compression_ratio"] for x in per_item) / n, 4),
        "avg_key_term_coverage": round(sum(x["key_term_coverage"] for x in per_item) / n, 4),
        "per_item": per_item,
    }
