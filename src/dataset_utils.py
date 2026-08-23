"""Dataset loading, validation, and Qwen chat-format conversion utilities."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REQUIRED_FIELDS = ("fact_id", "category", "instruction", "response", "variant")
EXPECTED_SPLIT_SIZES = {"train": 280, "validation": 70, "test": 70}
SYSTEM_PROMPT = (
    "You are NovaAI's technical support assistant. Answer the user's question "
    "accurately and concisely using the NovaAI product facts you know."
)


class DatasetValidationError(ValueError):
    """Raised when a dataset file does not match the expected NovaAI schema."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON Lines file with useful line-level errors."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required dataset file not found: {path.resolve()}. "
            "Clone the full repository or upload the four files from data/."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Invalid JSON in {path} on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise DatasetValidationError(
                    f"Expected an object in {path} on line {line_number}."
                )
            records.append(record)
    return records


def validate_records(
    records: Sequence[Mapping[str, Any]],
    split: str,
    expected_size: int | None = None,
) -> None:
    """Validate required fields, values, and per-split identifiers."""
    if not records:
        raise DatasetValidationError(f"The {split!r} split is empty.")
    if expected_size is not None and len(records) != expected_size:
        raise DatasetValidationError(
            f"Expected {expected_size} {split} rows, found {len(records)}."
        )

    seen_pairs: set[tuple[str, str]] = set()
    seen_instructions: set[str] = set()
    for index, record in enumerate(records, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise DatasetValidationError(
                f"{split} row {index} is missing fields: {', '.join(missing)}"
            )
        for field in REQUIRED_FIELDS:
            if not isinstance(record[field], str) or not record[field].strip():
                raise DatasetValidationError(
                    f"{split} row {index} field {field!r} must be a non-empty string."
                )

        pair = (record["fact_id"], record["variant"])
        if pair in seen_pairs:
            raise DatasetValidationError(
                f"Duplicate (fact_id, variant) pair in {split}: {pair}"
            )
        seen_pairs.add(pair)

        normalized_instruction = " ".join(record["instruction"].lower().split())
        if normalized_instruction in seen_instructions:
            raise DatasetValidationError(
                f"Duplicate instruction text found in {split} row {index}."
            )
        seen_instructions.add(normalized_instruction)


def validate_cross_split_consistency(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Ensure each fact has one category/answer and occurs in every split."""
    fact_metadata: dict[str, set[tuple[str, str]]] = defaultdict(set)
    fact_sets: dict[str, set[str]] = {}
    for split, records in splits.items():
        fact_sets[split] = {str(row["fact_id"]) for row in records}
        for row in records:
            fact_metadata[str(row["fact_id"])].add(
                (str(row["category"]), str(row["response"]))
            )

    inconsistent = sorted(
        fact_id for fact_id, metadata in fact_metadata.items() if len(metadata) != 1
    )
    if inconsistent:
        raise DatasetValidationError(
            "Facts have inconsistent category/response values across splits: "
            + ", ".join(inconsistent)
        )

    reference_name = next(iter(fact_sets))
    reference = fact_sets[reference_name]
    for split, fact_ids in fact_sets.items():
        if fact_ids != reference:
            missing = sorted(reference - fact_ids)
            extra = sorted(fact_ids - reference)
            raise DatasetValidationError(
                f"Fact coverage differs in {split}; missing={missing}, extra={extra}."
            )


def load_and_validate_splits(
    data_dir: str | Path,
    expected_sizes: Mapping[str, int] | None = EXPECTED_SPLIT_SIZES,
) -> dict[str, list[dict[str, Any]]]:
    """Load all three NovaAI splits and run structural consistency checks."""
    data_dir = Path(data_dir)
    split_names = ("train", "validation", "test")
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in split_names:
        records = read_jsonl(data_dir / f"{split}.jsonl")
        expected = expected_sizes.get(split) if expected_sizes is not None else None
        validate_records(records, split=split, expected_size=expected)
        splits[split] = records
    validate_cross_split_consistency(splits)
    return splits


def dataset_report(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return a compact, computed summary suitable for notebook display."""
    all_records = [row for records in splits.values() for row in records]
    fact_ids = {str(row["fact_id"]) for row in all_records}
    categories = Counter(
        str(row["category"])
        for row in splits.get("test", next(iter(splits.values())))
    )
    return {
        "split_sizes": {name: len(records) for name, records in splits.items()},
        "unique_facts": len(fact_ids),
        "test_categories": dict(sorted(categories.items())),
    }


def prompt_messages(instruction: str) -> list[dict[str, str]]:
    """Build the inference-time Qwen conversation for an instruction."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]


def training_messages(instruction: str, response: str) -> list[dict[str, str]]:
    """Build a complete Qwen conversation including the expected response."""
    return prompt_messages(instruction) + [{"role": "assistant", "content": response}]


def to_conversational_sft_dataset(records: Iterable[Mapping[str, Any]]):
    """Convert rows to TRL's conversational prompt/completion dataset format.

    The import is intentionally local so structural validation works without the
    optional training dependencies installed.
    """
    from datasets import Dataset

    rows = []
    for record in records:
        rows.append(
            {
                "prompt": prompt_messages(str(record["instruction"])),
                "completion": [
                    {"role": "assistant", "content": str(record["response"])}
                ],
                "fact_id": str(record["fact_id"]),
                "category": str(record["category"]),
            }
        )
    return Dataset.from_list(rows)


def render_training_example(record: Mapping[str, Any], tokenizer) -> str:
    """Render one training row with the model's native Qwen chat template."""
    return tokenizer.apply_chat_template(
        training_messages(str(record["instruction"]), str(record["response"])),
        tokenize=False,
        add_generation_prompt=False,
    )

