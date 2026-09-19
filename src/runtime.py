"""Configuration, output paths, and experiment provenance (CPU safe)."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(root=ROOT):
    root = Path(root)
    with (root / "configs/training_config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("model", "quantization", "lora", "training", "inference", "paths"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Missing configuration section: {key}")
    if config["model"]["base_model_id"] != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("This experiment requires Qwen/Qwen2.5-1.5B-Instruct.")
    if not config["training"]["fp16"] or config["training"]["bf16"]:
        raise ValueError("T4 requires fp16=True and bf16=False in this pipeline.")
    if config["training"]["packing"]:
        raise ValueError("This T4 pipeline validates un-packed examples; packing must remain false.")
    q = config["quantization"]
    if not q["load_in_4bit"] or q["bnb_4bit_quant_type"] != "nf4":
        raise ValueError("This pipeline requires 4-bit NF4.")
    if q["bnb_4bit_compute_dtype"] != "float16":
        raise ValueError("T4 compute dtype must be float16.")
    for value in (config["model"]["max_length"], config["training"]["num_train_epochs"],
                  config["training"]["per_device_train_batch_size"],
                  config["training"]["per_device_eval_batch_size"],
                  config["training"]["gradient_accumulation_steps"],
                  config["training"]["learning_rate"], config["lora"]["r"],
                  config["inference"]["batch_size"], config["inference"]["max_new_tokens"]):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("Training, generation, and length values must be positive numbers.")
    output_paths(root, config)
    return config


def output_paths(root, config):
    root = Path(root).resolve()
    values = {**config["paths"], "trainer": config["training"]["output_dir"],
              "adapter": config["training"]["adapter_dir"]}
    paths = {}
    for key, value in values.items():
        path = (root / value).resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError(f"Output path must be inside the repository: {key}")
        paths[key] = path
    if len(set(paths.values())) != len(paths):
        raise ValueError("Output paths must be distinct.")
    return paths


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dataset_hashes(root=ROOT):
    return {name: sha256(Path(root) / "data" / name)
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "dataset_summary.json")}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment_versions():
    return {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]}


def new_run(config, hashes, revision, gpu, packages):
    from .dataset_utils import SYSTEM_PROMPT
    return {
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": config["model"]["base_model_id"],
        "model_revision": revision,
        "dataset_hashes": hashes,
        "config": config,
        "config_hash": json_hash(config),
        "system_prompt": SYSTEM_PROMPT,
        "gpu": gpu, "package_versions": packages,
    }


def validate_run(run, config, hashes):
    from .dataset_utils import SYSTEM_PROMPT
    if not run.get("run_id") or not run.get("model_revision"):
        raise ValueError("Run metadata has no run ID or resolved model revision.")
    if run.get("config") != config or run.get("config_hash") != json_hash(config):
        raise ValueError("Configuration changed. Use a new output location for a new experiment.")
    if run.get("dataset_hashes") != hashes:
        raise ValueError("Dataset hashes changed; refusing to mix experiments.")
    if run.get("system_prompt") != SYSTEM_PROMPT:
        raise ValueError("System prompt changed; refusing to mix experiments.")


def sidecar_path(csv_path):
    return Path(csv_path).with_suffix(".metadata.json")


def prediction_metadata(run, role, csv_path):
    return {
        "run_id": run["run_id"], "role": role,
        "model_id": run["model_id"], "model_revision": run["model_revision"],
        "dataset_hash": run["dataset_hashes"]["test.jsonl"],
        "dataset_hashes": run["dataset_hashes"],
        "config_hash": run["config_hash"],
        "generation": {**run["config"]["inference"], "seed": run["config"]["seed"],
                       "max_length": run["config"]["model"]["max_length"]},
        "system_prompt": run["system_prompt"],
        "csv_sha256": sha256(csv_path),
    }


def validate_prediction(csv_path, role, run=None):
    path = sidecar_path(csv_path)
    if not path.is_file():
        raise ValueError(f"Missing provenance: {path}. Legacy predictions cannot be compared.")
    meta = read_json(path)
    required = ("run_id", "model_id", "model_revision", "dataset_hash", "dataset_hashes",
                "config_hash", "generation", "system_prompt", "csv_sha256")
    if any(not meta.get(key) for key in required) or meta.get("role") != role:
        raise ValueError("Incomplete prediction provenance or incorrect model role.")
    if meta["csv_sha256"] != sha256(csv_path):
        raise ValueError("Prediction CSV hash mismatch.")
    if run is not None and meta != prediction_metadata(run, role, csv_path):
        raise ValueError("Prediction belongs to a different experiment.")
    return meta


def validate_pair(base_path, tuned_path):
    base = validate_prediction(base_path, "base")
    tuned = validate_prediction(tuned_path, "fine_tuned")
    for key in ("run_id", "model_id", "model_revision", "dataset_hash", "dataset_hashes",
                "config_hash", "generation", "system_prompt"):
        if base[key] != tuned[key]:
            raise ValueError(f"Prediction provenance mismatch: {key}")
    return base


def validate_adapter(path):
    path = Path(path)
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Incomplete saved adapter: {path / name}")
    return path


def adapter_hashes(path):
    validate_adapter(path)
    return {p.name: sha256(p) for p in sorted(Path(path).iterdir()) if p.is_file()}
