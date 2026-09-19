"""CPU-only checks: no model download, CUDA, or training."""
import ast
import copy
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from src.dataset_utils import (
    DatasetValidationError, dataset_report, load_and_validate_splits,
    prompt_messages, training_messages, validate_records, validate_token_lengths,
    validate_cross_split_consistency,
)
from src.evaluation import compare_prediction_files, score_response
from src.runtime import (
    ROOT, adapter_hashes, dataset_hashes, load_config, new_run, output_paths,
    prediction_metadata, sidecar_path, validate_adapter, validate_pair,
    validate_prediction, validate_run, write_json,
)


def test_real_dataset_schema_counts_and_alignment():
    splits = load_and_validate_splits(ROOT / "data")
    assert {k: len(v) for k, v in splits.items()} == {"train": 280, "validation": 70, "test": 70}
    assert dataset_report(splits)["unique_facts"] == 70
    assert sum(dataset_report(splits)["test_categories"].values()) == 70
    with (ROOT / "results/base_model_results.csv").open(encoding="utf-8", newline="") as f:
        baseline = list(csv.DictReader(f))
    assert len(baseline) == len(splits["test"])
    for actual, expected in zip(baseline, splits["test"]):
        for key in ("fact_id", "category", "instruction"):
            assert actual[key] == expected[key]
        assert actual["expected_response"] == expected["response"]


@pytest.mark.parametrize("mutation", ["missing", "empty", "duplicate", "count"])
def test_schema_rejects_bad_data(mutation):
    row = dict(fact_id="a", category="b", instruction="question", response="answer", variant="train_1")
    rows = [row]
    expected = 1
    if mutation == "missing":
        row.pop("response")
    elif mutation == "empty":
        row["response"] = ""
    elif mutation == "duplicate":
        rows.append(row)
        expected = 2
    else:
        expected = 2
    with pytest.raises(DatasetValidationError):
        validate_records(rows, "train", expected)


def test_duplicate_across_splits_and_inconsistent_facts():
    rows = load_and_validate_splits(ROOT / "data")
    bad = copy.deepcopy(rows)
    bad["test"][0]["instruction"] = bad["train"][0]["instruction"]
    with pytest.raises(DatasetValidationError, match="Duplicate instruction"):
        validate_cross_split_consistency(bad)
    bad = copy.deepcopy(rows)
    bad["test"][0]["response"] = "different"
    with pytest.raises(DatasetValidationError, match="inconsistent"):
        validate_cross_split_consistency(bad)


def test_config_and_output_layout():
    cfg = load_config()
    assert cfg["model"]["base_model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["training"]["num_train_epochs"] == 3
    assert cfg["training"]["fp16"] and not cfg["training"]["bf16"]
    paths = output_paths(ROOT, cfg)
    assert paths["adapter"] == ROOT / "artifacts/adapter"
    assert paths["trainer"] == ROOT / "artifacts/trainer"
    assert paths["run_metadata"] == ROOT / "artifacts/run_metadata.json"
    cfg["paths"]["metrics"] = "../escape.json"
    with pytest.raises(ValueError, match="inside"):
        output_paths(ROOT, cfg)


def test_prompt_formatting_and_completion():
    messages = training_messages("question", "answer")
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[:2] == prompt_messages("question")
    assert messages[-1]["content"] == "answer"


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and not add_generation_prompt
        assert messages[-1]["role"] == "assistant"
        return list(range(len(messages[-1]["content"])))


def test_length_preflight_boundary_and_error():
    row = dict(fact_id="f", variant="train_1", instruction="q", response="12345")
    assert validate_token_lengths({"train": [row]}, FakeTokenizer(), 5) == {"train": 5}
    with pytest.raises(DatasetValidationError, match="train/f/train_1: 5 tokens"):
        validate_token_lengths({"train": [row]}, FakeTokenizer(), 4)
    assert row["response"] == "12345"


def test_metric_calculations():
    scores = score_response("Bearer <API_KEY>.", "bearer api key")
    assert all(value == 1 for value in scores.values())
    scores = score_response("red blue", "red green green")
    assert scores["token_precision"] == pytest.approx(1 / 3)
    assert scores["token_recall"] == .5
    assert scores["token_f1"] == pytest.approx(.4)
    assert scores["keyword_overlap"] == pytest.approx(1 / 3)
    assert scores["expected_fact_coverage"] == .5
    assert score_response("red blue", "")["token_f1"] == 0


@pytest.fixture
def experiment(tmp_path):
    cfg = load_config()
    run = new_run(cfg, dataset_hashes(), "a" * 40, {"name": "test-only"}, {})
    frame = pd.DataFrame([
        dict(fact_id="f1", category="c", instruction="q1", expected_response="red blue",
             base_model_response="red"),
        dict(fact_id="f2", category="c", instruction="q2", expected_response="green",
             base_model_response="green"),
    ])
    base, tuned = tmp_path / "base.csv", tmp_path / "tuned.csv"
    frame.to_csv(base, index=False)
    frame.rename(columns={"base_model_response": "fine_tuned_response"}).to_csv(tuned, index=False)
    for path, role in ((base, "base"), (tuned, "fine_tuned")):
        write_json(sidecar_path(path), prediction_metadata(run, role, path))
    return cfg, run, base, tuned, tmp_path


def test_comparison_and_category_scores(experiment):
    _, _, base, tuned, tmp = experiment
    frame, metrics = compare_prediction_files(base, tuned, tmp / "comparison.csv", tmp / "metrics.json")
    assert len(frame) == 2
    assert metrics["by_category"]["c"]["count"] == 2
    assert all(v == 0 for v in metrics["delta_fine_tuned_minus_base"].values())


@pytest.mark.parametrize("field", ["run_id", "model_revision", "model_id", "dataset_hash",
                                   "generation", "config_hash", "system_prompt", "dataset_hashes"])
def test_provenance_mismatch_rejected(experiment, field):
    _, _, base, tuned, _ = experiment
    meta = json.loads(sidecar_path(tuned).read_text())
    meta[field] = "different"
    write_json(sidecar_path(tuned), meta)
    with pytest.raises(ValueError, match="mismatch"):
        validate_pair(base, tuned)


def test_missing_metadata_and_csv_tampering(experiment):
    _, _, base, tuned, _ = experiment
    sidecar_path(tuned).unlink()
    with pytest.raises(ValueError, match="Missing provenance"):
        validate_pair(base, tuned)
    with base.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_prediction(base, "base")


def test_row_alignment_rejected_even_with_valid_sidecars(experiment):
    _, run, base, tuned, tmp = experiment
    frame = pd.read_csv(tuned).iloc[::-1]
    frame.to_csv(tuned, index=False)
    write_json(sidecar_path(tuned), prediction_metadata(run, "fine_tuned", tuned))
    with pytest.raises(ValueError, match="same ordered"):
        compare_prediction_files(base, tuned, tmp / "comparison.csv", tmp / "metrics.json")
    assert not (tmp / "metrics.json").exists()


def test_run_validation(experiment):
    cfg, run, _, _, _ = experiment
    validate_run(run, cfg, dataset_hashes())
    changed = copy.deepcopy(cfg)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="Configuration changed"):
        validate_run(run, changed, dataset_hashes())
    with pytest.raises(ValueError, match="Dataset hashes changed"):
        validate_run(run, cfg, {"test.jsonl": "changed"})


def test_adapter_validation_and_hashes(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        validate_adapter(adapter)
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        (adapter / name).write_text("test fixture")
    assert validate_adapter(adapter) == adapter
    assert len(adapter_hashes(adapter)) == 3


def test_legacy_baseline_preserved(tmp_path):
    from src.pipeline import archive_legacy_baseline
    path = tmp_path / "base_model_results.csv"
    payload = b"original,bytes\r\n1,2\r\n"
    path.write_bytes(payload)
    archive_legacy_baseline(path)
    archive_legacy_baseline(path)
    assert path.read_bytes() == payload
    copies = list(tmp_path.glob("*.legacy-*.csv"))
    assert len(copies) == 1 and copies[0].read_bytes() == payload


def test_notebook_structure_and_stages():
    notebook = json.loads((ROOT / "notebooks/fine_tuning_colab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    ids = [c["id"] for c in notebook["cells"]]
    assert len(set(ids)) == len(ids)
    codes = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    for cell in codes:
        assert cell["outputs"] == [] and cell["execution_count"] is None
        ast.parse("".join(cell["source"]))
    all_code = "\n".join("".join(c["source"]) for c in codes)
    assert "sys.path" not in all_code
    assert all(f'"{stage}"' in all_code for stage in
               ["preflight", "baseline", "train", "predict", "evaluate"])
    for path in (ROOT / "src").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))


def test_stage_cells_work_without_notebook_globals(monkeypatch, tmp_path):
    """Each stage dispatches in an empty namespace; no GPU command is executed."""
    import subprocess
    monkeypatch.chdir(ROOT)
    calls = []
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file",
                        lambda p: True if ".venv-colab" in str(p) else original(p))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    notebook = json.loads((ROOT / "notebooks/fine_tuning_colab.ipynb").read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and '"src.pipeline"' in source:
            exec(compile(source, "stage", "exec"), {})
    assert len(calls) == 5
    assert all(call[1]["check"] for call in calls)


def test_imports_without_ml_packages():
    import src.inference
    import src.pipeline
    assert callable(src.inference.generate_predictions)
    assert callable(src.pipeline.main)


@pytest.mark.parametrize("field,value", [("fp16", False), ("bf16", True), ("packing", True)])
def test_invalid_training_config_rejected(tmp_path, field, value):
    import yaml
    cfg = load_config()
    cfg["training"][field] = value
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/training_config.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_bootstrap_from_empty_namespace(monkeypatch, tmp_path):
    """Execute bootstrap with installation mocked; no package install/download."""
    import importlib.util
    import subprocess
    import sys
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(sys, "version_info", (3, 12))
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    notebook = json.loads((ROOT / "notebooks/fine_tuning_colab.ipynb").read_text())
    bootstrap = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    exec(compile(bootstrap, "bootstrap", "exec"), {})
    assert any("--index-url" in c[0][0] for c in calls)
    assert any(c[0][0][-2:] == ["-r", "requirements.txt"] for c in calls)
    assert any(c[0][0][-1] == "check" for c in calls)
