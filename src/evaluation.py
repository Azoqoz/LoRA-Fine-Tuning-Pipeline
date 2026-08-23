"""Reproducible lexical evaluation for the fictional NovaAI fact set."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

IDENTITY_COLUMNS = ["fact_id", "category", "instruction", "expected_response"]
METRIC_NAMES = [
    "normalized_exact_match",
    "token_precision",
    "token_recall",
    "token_f1",
    "keyword_overlap",
    "expected_fact_coverage",
]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this", "to",
    "uses", "with", "you", "your",
}


def normalize_text(text: Any) -> str:
    """Normalize case, Unicode, whitespace, and punctuation for exact matching."""
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = value.replace("`", "").replace("_", " ")
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _tokens(text: Any) -> list[str]:
    return normalize_text(text).split()


def _content_tokens(text: Any) -> set[str]:
    return {token for token in _tokens(text) if token not in STOPWORDS}


def score_response(expected: Any, predicted: Any) -> dict[str, float]:
    """Compute transparent lexical scores for one expected/predicted pair."""
    expected_tokens = _tokens(expected)
    predicted_tokens = _tokens(predicted)
    expected_counts = Counter(expected_tokens)
    predicted_counts = Counter(predicted_tokens)
    overlap = sum((expected_counts & predicted_counts).values())

    precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
    recall = overlap / len(expected_tokens) if expected_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    expected_keywords = _content_tokens(expected)
    predicted_keywords = _content_tokens(predicted)
    union = expected_keywords | predicted_keywords
    keyword_jaccard = (
        len(expected_keywords & predicted_keywords) / len(union) if union else 1.0
    )
    fact_coverage = (
        len(expected_keywords & predicted_keywords) / len(expected_keywords)
        if expected_keywords
        else 1.0
    )
    return {
        "normalized_exact_match": float(
            normalize_text(expected) == normalize_text(predicted)
        ),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "keyword_overlap": keyword_jaccard,
        "expected_fact_coverage": fact_coverage,
    }


def _validate_prediction_frame(frame: pd.DataFrame, response_column: str) -> None:
    missing = [
        column for column in IDENTITY_COLUMNS + [response_column] if column not in frame
    ]
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Prediction file contains no rows.")
    if frame["fact_id"].duplicated().any():
        raise ValueError("Prediction file contains duplicate fact_id values.")
    if frame[response_column].isna().any():
        raise ValueError(f"Prediction file contains missing {response_column} values.")


def _add_scores(frame: pd.DataFrame, response_column: str, prefix: str) -> pd.DataFrame:
    score_rows = [
        score_response(expected, predicted)
        for expected, predicted in zip(
            frame["expected_response"], frame[response_column], strict=True
        )
    ]
    scores = pd.DataFrame(score_rows).add_prefix(f"{prefix}_")
    return pd.concat([frame.reset_index(drop=True), scores], axis=1)


def _aggregate(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        metric: float(frame[f"{prefix}_{metric}"].mean()) for metric in METRIC_NAMES
    }


def compare_prediction_files(
    base_results_path: str | Path,
    fine_tuned_results_path: str | Path,
    comparison_path: str | Path,
    metrics_path: str | Path,
    *,
    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str = "artifacts/novaai-qwen2.5-1.5b-qlora",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate, compare, and persist metrics for two real prediction files."""
    base = pd.read_csv(base_results_path)
    tuned = pd.read_csv(fine_tuned_results_path)
    _validate_prediction_frame(base, "base_model_response")
    _validate_prediction_frame(tuned, "fine_tuned_response")

    if len(base) != len(tuned):
        raise ValueError(f"Row count mismatch: base={len(base)}, tuned={len(tuned)}")
    for column in IDENTITY_COLUMNS:
        if not base[column].astype(str).equals(tuned[column].astype(str)):
            raise ValueError(
                f"The two prediction files do not use the exact same ordered {column}."
            )

    base_scored = _add_scores(base, "base_model_response", "base")
    tuned_scored = _add_scores(tuned, "fine_tuned_response", "fine_tuned")
    comparison = base_scored.merge(
        tuned_scored[["fact_id", "fine_tuned_response"] + [
            f"fine_tuned_{metric}" for metric in METRIC_NAMES
        ]],
        on="fact_id",
        validate="one_to_one",
    )

    base_summary = _aggregate(comparison, "base")
    tuned_summary = _aggregate(comparison, "fine_tuned")
    deltas = {
        metric: tuned_summary[metric] - base_summary[metric] for metric in METRIC_NAMES
    }
    by_category: dict[str, Any] = {}
    for category, group in comparison.groupby("category", sort=True):
        category_base = _aggregate(group, "base")
        category_tuned = _aggregate(group, "fine_tuned")
        by_category[str(category)] = {
            "count": int(len(group)),
            "base_model": category_base,
            "fine_tuned_model": category_tuned,
            "delta": {
                metric: category_tuned[metric] - category_base[metric]
                for metric in METRIC_NAMES
            },
        }

    metrics: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": {
            "generation": "greedy decoding (do_sample=False), same ordered test prompts",
            "normalized_exact_match": "case/punctuation/whitespace-normalized equality",
            "token_overlap": "multiset token precision, recall, and F1",
            "keyword_overlap": "Jaccard overlap after removing a fixed stopword list",
            "expected_fact_coverage": "fraction of expected non-stopword tokens present",
        },
        "num_examples": int(len(comparison)),
        "base_model_id": base_model_id,
        "adapter_path": adapter_path,
        "base_model": base_summary,
        "fine_tuned_model": tuned_summary,
        "delta_fine_tuned_minus_base": deltas,
        "by_category": by_category,
    }

    comparison_path = Path(comparison_path)
    metrics_path = Path(metrics_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return comparison, metrics
