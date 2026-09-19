"""Deterministic batched inference helpers for base and adapter evaluations."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
try:
    from .dataset_utils import prompt_messages
except ImportError:  # Compatibility with older notebooks importing from src/.
    from dataset_utils import prompt_messages


def set_reproducible_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch inference paths."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_input_device(model) -> torch.device:
    """Find the embedding device for regular, quantized, and PEFT models."""
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, RuntimeError):
        return next(model.parameters()).device


def generate_predictions(
    model,
    tokenizer,
    records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    response_column: str,
    *,
    batch_size: int = 2,
    max_new_tokens: int = 128,
    seed: int = 42,
    max_length: int = 512,
    do_sample: bool = False,
) -> pd.DataFrame:
    """Generate deterministic answers and save a CSV in test-set order."""
    import torch

    if batch_size < 1 or max_new_tokens < 1 or max_length < 1:
        raise ValueError("Batch size and token limits must be positive.")
    if response_column not in {"base_model_response", "fine_tuned_response"}:
        raise ValueError(
            "response_column must be 'base_model_response' or 'fine_tuned_response'."
        )
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("No evaluation records were supplied.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(seed)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device = _model_input_device(model)
    predictions: list[str] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                prompt_messages(str(row["instruction"])),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        ).to(device)
        prompt_width = inputs["input_ids"].shape[1]
        if prompt_width > max_length:
            raise ValueError(f"Prompt exceeds max_length={max_length}; refusing truncation.")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(
            generated[:, prompt_width:], skip_special_tokens=True
        )
        predictions.extend(answer.strip() for answer in decoded)
        print(f"Generated {min(start + batch_size, len(rows))}/{len(rows)}", end="\r")

    frame = pd.DataFrame(
        {
            "fact_id": [row["fact_id"] for row in rows],
            "category": [row["category"] for row in rows],
            "instruction": [row["instruction"] for row in rows],
            "expected_response": [row["response"] for row in rows],
            response_column: predictions,
        }
    )
    frame.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved {len(frame)} predictions to {output_path}")
    return frame
