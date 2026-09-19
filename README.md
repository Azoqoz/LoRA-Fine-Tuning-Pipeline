# LoRA Fine-Tuning Pipeline

A Google Colab-first AI engineering project that fine-tunes
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
on a compact product-support dataset using 4-bit QLoRA. The repository focuses on
the complete training and evaluation lifecycle—not a chatbot user interface.

> **NovaAI is fictional and was created only for this experiment.** Its product
> facts do not describe a real company or service.

## Project overview

The project demonstrates how to:

- validate an instruction-tuning dataset before training;
- establish a base-model benchmark on a held-out test set;
- format instruction/response pairs with Qwen's native chat template;
- quantize a 1.5B-parameter model to 4-bit NF4 and train LoRA adapters;
- generate on the exact same 70 prompts before and after fine-tuning; and
- save row-level comparisons plus reproducible aggregate metrics.

No OpenAI, Anthropic, Gemini, or other paid API is used. The intended runtime is
a free Google Colab notebook with an NVIDIA Tesla T4 GPU.

## Why QLoRA?

Full fine-tuning updates and stores every model parameter. QLoRA instead keeps
the base model quantized and frozen while training small low-rank adapter
matrices. In this project, NF4 quantization and nested (double) quantization
reduce GPU memory use, while LoRA targets the attention and MLP projection
layers. This makes the experiment practical on a T4 without turning it into a
toy pipeline.

## Workflow

```text
JSONL splits ──> schema/fact validation ──> Qwen chat formatting
                                              │
Test prompts ──> FP16 base inference ─────────┤──> base_model_results.csv
                                              │
Train + validation ──> 4-bit Qwen + LoRA ──> SFTTrainer ──> adapter
                                              │
Same test prompts ──> adapter inference ──────┤──> fine_tuned_results.csv
                                              │
Expected responses ───────────────────────────┴──> comparison.csv + metrics.json
```

The baseline uses the normal FP16 model, not the quantized training model. The
training model is loaded separately in 4-bit, prepared for k-bit training, and
wrapped with explicit PEFT LoRA adapters.

## Dataset

The included NovaAI dataset contains 70 fictional product facts across eight
categories: authentication, API keys, models, rate limits, semantic cache,
configuration, deployment, and troubleshooting. Every fact has four training
paraphrases, one validation paraphrase, and one held-out test paraphrase.

| Split | Rows | Variants per fact | Purpose |
|---|---:|---:|---|
| Train | 280 | 4 | Adapter optimization |
| Validation | 70 | 1 | Loss evaluation after each epoch |
| Test | 70 | 1 | Before/after generation comparison |

Each JSONL row contains `fact_id`, `category`, `instruction`, `response`, and
`variant`. The loader checks required fields, expected split sizes, unique row
identifiers, consistent answers/categories, and identical fact coverage across
all splits. The supplied dataset content is not modified.

The splits measure **paraphrase recall of trained facts**, not evaluation on
unseen facts: the same 70 facts occur in all three splits. NovaAI is synthetic
and fictional. QLoRA trains adapters while the quantized base weights stay frozen.

## Model and training configuration

The complete machine-readable configuration is in
[`configs/training_config.yaml`](configs/training_config.yaml).

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Quantization | 4-bit NF4 |
| Double quantization | Enabled |
| Compute dtype | Float16 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs | 3 |
| Train / eval batch size | 2 / 2 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |
| Maximum sequence length | 512 |
| Precision | `fp16=True`, `bf16=False` |
| Evaluation / save cadence | Every epoch |
| External reporting | None |

The effective training batch size on one GPU is 8 examples (2 × 4 gradient
accumulation steps).

## Evaluation methodology

Both models use deterministic greedy decoding (`do_sample=False`) on the exact
same ordered test prompts. The expected test response is the reference.

- **Normalized exact match:** equality after Unicode, case, punctuation, and
  whitespace normalization.
- **Token overlap:** multiset precision, recall, and F1 over normalized tokens.
- **Keyword overlap:** Jaccard overlap over non-stopword token sets.
- **Expected fact coverage:** fraction of expected non-stopword tokens found in
  the generated answer.

These lexical metrics are intentionally simple and inspectable. They do not
fully measure semantic correctness, contradiction, or hallucination. Review
`comparison.csv` qualitatively alongside the aggregate scores.

This README makes **no performance-improvement claim**. Final comparison metrics
must come from actually running both inference passes and are written by the
notebook only after both prediction files validate successfully.

## Run in Google Colab

1. Open [`notebooks/fine_tuning_colab.ipynb`](notebooks/fine_tuning_colab.ipynb)
   in Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Choose **Run all**. Bootstrap clones the full repository if absent, changes
   into it, verifies source/config files, and installs pinned dependencies in an
   isolated environment. Optional dataset upload is only for missing files.
4. Preflight checks CUDA, an NF4 forward pass, dataset/token lengths and actual
   quantized Qwen loading. Subsequent stages run the baseline, train, save,
   reload the adapter from disk, and evaluate.
5. Download the `results/` files and the saved adapter directory before the
   ephemeral Colab runtime is deleted.

No Hugging Face token is required for the public Qwen base model. The first run
downloads model weights and installs the pinned dependencies, so it needs normal
internet access. Each stage runs in a fresh process in the isolated environment,
so stale notebook imports cannot affect it and process exit releases GPU memory.
An installation restart is normally unnecessary. After any kernel restart,
rerun **Section 1 (bootstrap), Section 2 (preflight), then your unfinished stage**.
Verified predictions and completed training are reused; interrupted training
resumes from the latest epoch checkpoint.

The pinned stack targets Python 3.10–3.12 and PyTorch 2.8.0 with CUDA 12.6,
using the [official wheel index](https://pytorch.org/get-started/previous-versions/).
Training uses the [TRL 0.24.0 interface](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer).
The environment, model cache and checkpoints require several GB of disk space.

**Unpublished changes:** cloning retrieves only published code. Until this update
is published, upload/extract the updated full repository to
/content/LoRA-Fine-Tuning-Pipeline before running the notebook. Bootstrap uses
that existing directory without overwriting it. Uploading only the new notebook
does not provide unpublished source files.

### Reproduce training

The notebook is the canonical entry point because bitsandbytes QLoRA requires a
supported NVIDIA GPU. It performs these stages in order:

1. CUDA/T4 check and dependency installation;
2. dataset loading and strict validation;
3. tokenizer/chat-template preparation;
4. baseline generation over all 70 test rows;
5. explicit GPU cleanup;
6. 4-bit model load, k-bit preparation, and LoRA attachment;
7. three-epoch `SFTTrainer` run with validation/save each epoch;
8. adapter save and tuned generation over the same test rows; and
9. comparison and metrics export.

For exact reproducibility, retain the pinned versions in `requirements.txt`, the
seed in the YAML configuration, and the generated trainer state alongside the
adapter. GPU kernels can still introduce small numerical variation between
hardware/runtime releases.

YAML controls model/revision, token limits, quantization, LoRA, training,
generation and output paths. Seed initialization precedes adapter creation.
Overlong complete chat examples fail before training with split, fact ID,
variant and token count; target answers are never silently truncated.

Generated output structure (created only by actual runs):

~~~text
artifacts/
├── adapter/                 # PEFT adapter + tokenizer
├── trainer/                 # epoch checkpoints + trainer_state.json
├── training_config.yaml     # exact configuration used
└── run_metadata.json        # run ID, model revision, dataset hashes,
                            # package versions, GPU, timestamp, adapter hashes
results/
├── base_model_results.csv
├── base_model_results.metadata.json
├── base_model_results.legacy-<hash>.csv
├── fine_tuned_results.csv
├── fine_tuned_results.metadata.json
├── comparison.csv
└── metrics.json
~~~

The supplied baseline remains unchanged in this update. When you execute the
baseline stage, its original bytes are archived under the legacy filename before
new predictions are generated. No provenance is invented for the legacy file;
a fresh baseline is needed for a verified comparison.

Sidecars record model/revision, generation settings, dataset hashes, run ID,
configuration hash, system prompt and CSV hash. Comparison rejects missing or
mismatched provenance, modified CSVs and different ordered test identities.
Saved adapter hashes are checked before reloading from disk. Use distinct output
paths for a new experiment; incompatible existing runs are rejected.
The FP16 baseline versus 4-bit tuned comparison also includes quantization effects.

## Use the saved adapter

After training, reload the same base model and attach the adapter:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "artifacts/adapter"

import json
with open("artifacts/run_metadata.json", encoding="utf-8") as stream:
    run = json.load(stream)

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    revision=run["model_revision"],
    quantization_config=quantization,
    device_map={"": 0},
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

The notebook includes the complete Qwen chat-template prompt and generation
code. Adapter weights are intentionally ignored by Git because they are
generated artifacts; archive or publish them separately if needed.

## Project structure

```text
LoRA-Fine-Tuning-Pipeline/
├── configs/
│   └── training_config.yaml
├── data/
│   ├── dataset_summary.json
│   ├── test.jsonl
│   ├── train.jsonl
│   └── validation.jsonl
├── notebooks/
│   └── fine_tuning_colab.ipynb
├── results/
│   ├── .gitkeep
│   └── base_model_results.csv
├── src/
│   ├── __init__.py
│   ├── dataset_utils.py
│   ├── evaluation.py
│   ├── runtime.py
│   ├── pipeline.py
│   └── inference.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements-test.txt
└── requirements.txt
```

After a complete Colab run, `results/` also contains
`fine_tuned_results.csv`, `comparison.csv`, and `metrics.json`. A pre-existing
baseline CSV supplied with this workspace has been preserved; the future baseline
stage archives it before generating predictions from the verified run.

## Lightweight tests

The additional requirements-test.txt installs only CPU test dependencies:

~~~bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests
~~~

Tests cover dataset schema/counts/alignment, configuration, prompt formatting,
metrics, result alignment, provenance, adapter layout, token-length limits, imports,
and notebook restart/structural checks. Fixtures and mocked operations require
no GPU, model download or training. CPU test success does not establish GPU success.

## Technologies

- Python and PyTorch
- Hugging Face Transformers and Datasets
- PEFT (LoRA)
- bitsandbytes (4-bit NF4 quantization)
- TRL `SFTTrainer`
- pandas and PyYAML
- Google Colab / NVIDIA Tesla T4

## Limitations

- The dataset is small, synthetic, and restricted to one fictional domain.
- Lexical scores can penalize correct paraphrases and reward answers that repeat
  keywords while adding incorrect claims.
- There is one held-out paraphrase per fact, so results are not a broad measure
  of language-model capability.
- The pipeline does not include human evaluation, semantic judging, confidence
  calibration, or safety evaluation.
- A T4 runtime is usually sufficient but Colab capacity and session duration are
  not guaranteed.

## Future improvements

- Add semantic similarity and contradiction-aware local evaluation models.
- Repeat training across several seeds and report confidence intervals.
- Expand the dataset with harder negatives, ambiguous questions, and explicit
  out-of-domain refusal examples.
- Track latency, peak memory, adapter size, and generation throughput.
- Run the CPU-safe test suite in CI and add a recorded Colab GPU integration run.
- Compare alternative adapter ranks, target-module sets, and small base models.

## Suggested CV bullet

> Built a reproducible QLoRA fine-tuning and evaluation pipeline for Qwen2.5-1.5B,
> using 4-bit NF4 quantization, PEFT adapters, TRL, strict dataset validation, and
> deterministic before/after benchmarking on Google Colab T4 hardware.

## License

Released under the [MIT License](LICENSE). The Qwen model and third-party
libraries remain subject to their respective licenses.
