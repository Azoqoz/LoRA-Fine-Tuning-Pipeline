"""Disk-backed notebook stages. Run from the repository: python -m src.pipeline STAGE."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .dataset_utils import (
    dataset_report, load_and_validate_splits, to_conversational_sft_dataset,
    training_messages, validate_token_lengths,
)
from .runtime import (
    ROOT, adapter_hashes, dataset_hashes, environment_versions, load_config, new_run,
    output_paths, prediction_metadata, read_json, sha256, sidecar_path,
    validate_adapter, validate_prediction, validate_run, write_json,
)


def gpu_preflight(config):
    import torch
    import bitsandbytes as bnb

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select Runtime > Change runtime type > T4 GPU.")
    info = {
        "name": torch.cuda.get_device_name(0),
        "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda": torch.version.cuda, "compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    print(json.dumps(info, indent=2))
    q = config["quantization"]
    dtype = getattr(torch, q["bnb_4bit_compute_dtype"])
    try:
        layer = bnb.nn.Linear4bit(
            64, 64, bias=False, compute_dtype=dtype,
            compress_statistics=q["bnb_4bit_use_double_quant"],
            quant_type=q["bnb_4bit_quant_type"],
        ).to("cuda")
        with torch.inference_mode():
            out = layer(torch.randn(2, 64, device="cuda", dtype=dtype))
        if layer.weight.quant_state is None or not torch.isfinite(out).all().item():
            raise RuntimeError("4-bit layer did not quantize or produced non-finite values.")
        torch.cuda.synchronize()
        del out, layer
        torch.cuda.empty_cache()
    except Exception as exc:
        raise RuntimeError("bitsandbytes 4-bit CUDA forward failed; rerun bootstrap on a T4 runtime.") from exc
    print("bitsandbytes NF4 CUDA forward passed.")
    return info


def quantization_config(config):
    import torch
    from transformers import BitsAndBytesConfig
    settings = dict(config["quantization"])
    settings["bnb_4bit_compute_dtype"] = getattr(torch, settings["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(**settings)


def load_quantized(config, revision):
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM
    q = quantization_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model_id"], revision=revision,
        quantization_config=q, torch_dtype=q.bnb_4bit_compute_dtype,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    layers = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    if not getattr(model, "is_loaded_in_4bit", False) or not layers:
        raise RuntimeError("Qwen was not loaded in bitsandbytes 4-bit mode.")
    for layer in layers:
        state = layer.weight.quant_state
        if (state is None or state.quant_type != q.bnb_4bit_quant_type
                or layer.compute_dtype != q.bnb_4bit_compute_dtype
                or bool(state.nested) != q.bnb_4bit_use_double_quant):
            raise RuntimeError("Loaded Qwen quantization does not match YAML settings.")
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = set(config["lora"]["target_modules"]) - available
    if missing:
        raise ValueError(f"LoRA target modules missing: {sorted(missing)}")
    print(f"Qwen 4-bit preflight passed: {len(layers)} NF4 layers; all LoRA targets found.")
    return model


def prepare(root, config, paths):
    """Validate data/environment and create or validate a persistent run identity."""
    from huggingface_hub import model_info
    from transformers import AutoTokenizer

    gpu = gpu_preflight(config)
    splits = load_and_validate_splits(root / "data")
    report = dataset_report(splits)
    summary = read_json(root / "data/dataset_summary.json")
    if (report["split_sizes"] != summary["splits"] or report["unique_facts"] != summary["facts"]
            or report["test_categories"] != summary["categories"]):
        raise ValueError("dataset_summary.json disagrees with the actual dataset.")
    print(json.dumps(report, indent=2))
    hashes = dataset_hashes(root)
    if paths["run_metadata"].exists():
        run = read_json(paths["run_metadata"])
        validate_run(run, config, hashes)
    else:
        if paths["adapter"].exists() or paths["trainer"].exists():
            raise ValueError("Existing artifacts have no run metadata; select new output paths.")
        revision = model_info(config["model"]["base_model_id"],
                              revision=config["model"]["revision"]).sha
        if not revision:
            raise RuntimeError("Cannot resolve the model revision for reproducible loading.")
        run = new_run(config, hashes, revision, gpu, environment_versions())
    tokenizer = AutoTokenizer.from_pretrained(run["model_id"], revision=run["model_revision"])
    print("Maximum complete-chat lengths:",
          validate_token_lengths(splits, tokenizer, config["model"]["max_length"]))
    model = load_quantized(config, run["model_revision"])
    del model
    paths["saved_config"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "configs/training_config.yaml", paths["saved_config"])
    write_json(paths["run_metadata"], run)
    print("Preflight complete. Run ID:", run["run_id"])


def load_run(root, config, paths):
    if not paths["run_metadata"].is_file():
        raise FileNotFoundError("Run Section 2 (preflight) before this stage.")
    run = read_json(paths["run_metadata"])
    validate_run(run, config, dataset_hashes(root))
    # Prevent resuming a training checkpoint with a different installed stack.
    if run["package_versions"] != environment_versions():
        raise ValueError("Package versions changed since preflight; restore the recorded environment.")
    return run


def archive_legacy_baseline(path):
    """Preserve legacy CSV bytes before a future, explicitly executed baseline stage."""
    if path.exists() and not sidecar_path(path).exists():
        archive = path.with_name(f"{path.stem}.legacy-{sha256(path)[:12]}.csv")
        if archive.exists() and sha256(archive) != sha256(path):
            raise ValueError("Legacy archive collision; refusing overwrite.")
        if not archive.exists():
            shutil.copyfile(path, archive)
        print("Preserved legacy baseline:", archive)


def predict(root, config, paths, run, role):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .inference import generate_predictions

    target = paths["base_results" if role == "base" else "fine_tuned_results"]
    if role == "fine_tuned":
        if not run.get("training_complete"):
            raise ValueError("Complete the training stage before adapter inference.")
        if run.get("adapter_hashes") != adapter_hashes(paths["adapter"]):
            raise ValueError("Adapter differs from the completed training run.")
    if target.exists() and sidecar_path(target).exists():
        validate_prediction(target, role, run)
        print("Reusing verified predictions:", target)
        return
    splits = load_and_validate_splits(root / "data")
    if role == "base":
        archive_legacy_baseline(target)
        tokenizer = AutoTokenizer.from_pretrained(run["model_id"], revision=run["model_revision"])
        model = AutoModelForCausalLM.from_pretrained(
            run["model_id"], revision=run["model_revision"],
            torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
        )
    else:
        adapter = validate_adapter(paths["adapter"])
        # Always reload both tokenizer and adapter from disk in a new process.
        tokenizer = AutoTokenizer.from_pretrained(adapter)
        base = load_quantized(config, run["model_revision"])
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.config.use_cache = True
    generate_predictions(
        model, tokenizer, splits["test"], target,
        "base_model_response" if role == "base" else "fine_tuned_response",
        **config["inference"], seed=config["seed"], max_length=config["model"]["max_length"],
    )
    write_json(sidecar_path(target), prediction_metadata(run, role, target))


def train(root, config, paths, run):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    if run.get("training_complete"):
        if run.get("adapter_hashes") != adapter_hashes(paths["adapter"]):
            raise ValueError("Saved adapter has changed since training.")
        print("Training already complete for this run; saved adapter verified.")
        return
    validate_prediction(paths["base_results"], "base", run)
    gpu_preflight(config)
    set_seed(config["seed"])  # Includes deterministic adapter initialization.
    tokenizer = AutoTokenizer.from_pretrained(run["model_id"], revision=run["model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    splits = load_and_validate_splits(root / "data")
    validate_token_lengths(splits, tokenizer, config["model"]["max_length"])
    model = load_quantized(config, run["model_revision"])
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(**config["lora"], revision=run["model_revision"]))
    model.print_trainable_parameters()
    settings = dict(config["training"])
    settings.pop("adapter_dir")
    resume = settings.pop("resume_from_checkpoint")
    settings["output_dir"] = str(paths["trainer"])
    args = SFTConfig(
        **settings, max_length=config["model"]["max_length"], seed=config["seed"],
        data_seed=config["seed"], gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = SFTTrainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=to_conversational_sft_dataset(splits["train"]),
        eval_dataset=to_conversational_sft_dataset(splits["validation"]),
    )
    # Check TRL's actual prepared data as well as the native-chat preflight.
    for name, dataset in (("train", trainer.train_dataset), ("validation", trainer.eval_dataset)):
        for original, row in zip(splits[name], dataset, strict=True):
            expected_ids = tokenizer.apply_chat_template(
                training_messages(original["instruction"], original["response"]),
                tokenize=True, add_generation_prompt=False,
            )
            if row["input_ids"] != expected_ids:
                raise ValueError(f"{name}/{original['fact_id']}: prepared tokens changed or truncated.")
            mask = row.get("completion_mask")
            if mask is None or not any(mask):
                raise ValueError(f"{name}: no completion tokens remain for training.")
    checkpoint = get_last_checkpoint(str(paths["trainer"])) if paths["trainer"].is_dir() else None
    if checkpoint and not resume:
        raise ValueError("Checkpoints exist but resume is disabled; select a new output directory.")
    trainer.train(resume_from_checkpoint=checkpoint if resume else None)
    trainer.save_model(str(paths["adapter"]))
    tokenizer.save_pretrained(paths["adapter"])
    trainer.save_state()
    run["adapter_hashes"] = adapter_hashes(paths["adapter"])
    run["training_complete"] = True
    write_json(paths["run_metadata"], run)
    print("Adapter and tokenizer saved:", paths["adapter"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "baseline", "train", "predict", "evaluate"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_config(root)
    paths = output_paths(root, config)
    if args.stage == "preflight":
        prepare(root, config, paths)
        return
    run = load_run(root, config, paths)
    if args.stage in ("baseline", "predict"):
        predict(root, config, paths, run, "base" if args.stage == "baseline" else "fine_tuned")
    elif args.stage == "train":
        train(root, config, paths, run)
    else:
        from .evaluation import compare_prediction_files
        for key, role in (("base_results", "base"), ("fine_tuned_results", "fine_tuned")):
            validate_prediction(paths[key], role, run)
        _, metrics = compare_prediction_files(
            paths["base_results"], paths["fine_tuned_results"],
            paths["comparison"], paths["metrics"], base_model_id=run["model_id"],
            adapter_path=str(paths["adapter"].relative_to(root)),
        )
        print(json.dumps({k: metrics[k] for k in
                          ("num_examples", "base_model", "fine_tuned_model",
                           "delta_fine_tuned_minus_base")}, indent=2))


if __name__ == "__main__":
    main()
