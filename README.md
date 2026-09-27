# LoRA Fine-Tuning Pipeline

A Colab-first QLoRA fine-tuning and evaluation pipeline for `Qwen/Qwen2.5-1.5B-Instruct`, with reproducible base-vs-fine-tuned benchmarking, validated experiment artifacts, and a Next.js results website.

![Python](https://img.shields.io/badge/Language-Python-blue)
![Qwen](https://img.shields.io/badge/Base%20Model-Qwen2.5--1.5B--Instruct-purple)
![QLoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA-green)
![PEFT](https://img.shields.io/badge/Adapters-PEFT-orange)
![Colab](https://img.shields.io/badge/Training-Google%20Colab-F9AB00)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-76B900)
![Next.js](https://img.shields.io/badge/Results%20Frontend-Next.js-black)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Live Application

**Experiment Results Website:**  
https://lo-ra-fine-tuning-pipeline.vercel.app

The deployed frontend presents the completed fine-tuning experiment, before/after model comparisons, evaluation metrics, and selected results.

The training itself is performed in Google Colab rather than in the browser.

---

## Overview

LoRA Fine-Tuning Pipeline is an AI engineering project focused on the complete lifecycle of parameter-efficient fine-tuning.

The project fine-tunes:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

on a compact fictional product-support dataset using:

```text
4-bit QLoRA
```

The goal is not to build another chatbot interface.

Instead, the repository demonstrates how to:

- Build and validate an instruction-tuning dataset
- Establish a base-model benchmark
- Apply Qwen's native chat template
- Quantize a base model using 4-bit NF4
- Prepare the model for k-bit training
- Attach explicit LoRA adapters
- Fine-tune using TRL `SFTTrainer`
- Evaluate the same held-out prompts before and after tuning
- Save row-level comparisons
- Export reproducible aggregate metrics
- Validate experiment provenance
- Preserve lightweight experiment artifacts
- Present results through a dedicated production frontend

No OpenAI, Anthropic, Gemini, or other paid LLM API is required.

The intended training environment is a free Google Colab session with an NVIDIA Tesla T4 GPU.

---

## Experiment Workflow

```text
JSONL Dataset
     |
     v
Schema + Fact Validation
     |
     v
Qwen Chat Formatting
     |
     +-----------------------------+
     |                             |
     v                             v
FP16 Base Model                Training Split
     |                             |
     v                             v
70 Test Prompts              4-bit Qwen Model
     |                             |
     v                             v
Base Results                 PEFT LoRA Adapters
                                   |
                                   v
                              SFTTrainer
                                   |
                                   v
                             Saved Adapter
                                   |
                                   v
                         Same 70 Test Prompts
                                   |
                                   v
                         Fine-Tuned Results
                                   |
                +------------------+------------------+
                |                                     |
                v                                     v
         Row-Level Comparison                  Aggregate Metrics
                |                                     |
                +------------------+------------------+
                                   |
                                   v
                         Results Website
```

---

## Why QLoRA?

Full fine-tuning updates the weights of the entire model.

QLoRA instead keeps the base model quantized and frozen while training small low-rank adapter matrices.

This project uses:

```text
4-bit NF4 quantization
+
Double quantization
+
PEFT LoRA adapters
```

This reduces GPU-memory requirements enough to make the experiment practical on a Tesla T4 while still running a real fine-tuning pipeline.

LoRA is applied to both attention and MLP projection layers.

---

## Dataset

The project uses a fictional product-support domain named:

```text
NovaAI
```

> **NovaAI is fictional and exists only for this experiment.**
>
> Its product facts do not describe a real company, product, or service.

The dataset contains:

```text
70 fictional product facts
```

across eight categories:

- Authentication
- API keys
- Models
- Rate limits
- Semantic cache
- Configuration
- Deployment
- Troubleshooting

Each fact contains:

```text
4 training paraphrases
1 validation paraphrase
1 held-out test paraphrase
```

This produces:

| Split | Rows | Variants per fact | Purpose |
|---|---:|---:|---|
| Train | 280 | 4 | Adapter optimization |
| Validation | 70 | 1 | Loss evaluation after each epoch |
| Test | 70 | 1 | Before/after generation comparison |

Total:

```text
420 examples
```

Each JSONL row contains:

```text
fact_id
category
instruction
response
variant
```

The loader validates:

- Required fields
- Expected split sizes
- Unique row identifiers
- Consistent answers
- Consistent categories
- Matching fact coverage across splits
- Token-length constraints

The supplied dataset content is not silently modified during loading.

---

## Evaluation Scope

The dataset evaluates:

```text
Paraphrase recall of trained facts
```

It does **not** evaluate generalization to completely unseen facts.

The same 70 underlying facts are represented across training, validation, and test splits using different paraphrases.

This distinction is important when interpreting the results.

---

## Model Configuration

The base model is:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The complete machine-readable experiment configuration is stored in:

```text
configs/training_config.yaml
```

### Training Configuration

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Quantization | 4-bit NF4 |
| Double quantization | Enabled |
| Compute dtype | Float16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Epochs | 3 |
| Train batch size | 2 |
| Evaluation batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Learning rate | `2e-4` |
| Maximum sequence length | 512 |
| Precision | FP16 |
| Evaluation cadence | Every epoch |
| Save cadence | Every epoch |
| External reporting | None |

---

## LoRA Target Modules

Adapters are attached to:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

This covers projection layers in both:

- Attention
- MLP blocks

The quantized base weights remain frozen while the LoRA adapter parameters are optimized.

---

## Baseline vs Fine-Tuned Evaluation

The evaluation compares:

```text
FP16 Base Model
       vs
4-bit Base Model + Fine-Tuned LoRA Adapter
```

Both models receive the exact same ordered 70 test prompts.

Generation uses deterministic greedy decoding:

```python
do_sample=False
```

The expected test response is used as the reference.

---

## Evaluation Metrics

The project uses simple, reproducible lexical metrics.

### Normalized Exact Match

Checks equality after normalizing:

- Unicode
- Case
- Punctuation
- Whitespace

---

### Token Precision

Measures how many generated normalized tokens belong to the expected response.

---

### Token Recall

Measures how many expected normalized tokens appear in the generated response.

---

### Token F1

Balances token precision and recall.

---

### Keyword Jaccard

Measures overlap between non-stopword keyword sets.

---

### Expected-Token Coverage

Measures the fraction of expected non-stopword tokens found in the generated answer.

---

## Experiment Results

The completed experiment was run on:

```text
NVIDIA Tesla T4
```

using:

```text
Qwen2.5-1.5B-Instruct
4-bit NF4
Double quantization
PEFT LoRA adapters
```

Training configuration:

```text
3 epochs
105 training steps
```

Recorded training time:

```text
309.5443 seconds
```

Only the adapter parameters were trained.

### Results

| Metric | Base (%) | Fine-Tuned (%) | Difference |
|---|---:|---:|---:|
| Normalized exact match | 0.00 | 58.57 | +58.57 pp |
| Token precision | 9.53 | 79.35 | +69.81 pp |
| Token recall | 57.03 | 80.86 | +23.83 pp |
| Token F1 | 15.97 | 79.64 | +63.67 pp |
| Keyword Jaccard | 10.49 | 75.47 | +64.97 pp |
| Expected-token coverage | 53.33 | 81.62 | +28.29 pp |

---

## Exact-Match Results

Before fine-tuning:

```text
0 / 70
```

After fine-tuning:

```text
41 / 70
```

normalized exact matches.

Token F1:

```text
Improved: 69 examples
Worsened: 1 example
Unchanged: 0 examples
```

These scores are lexical evaluation metrics.

They should not be interpreted as a complete measurement of:

- Factual correctness
- Semantic equivalence
- Contradiction
- Hallucination
- General LLM capability

Some fine-tuned answers can still confuse unrelated facts.

For this reason, the repository preserves row-level outputs for qualitative inspection.

---

## Results Artifacts

Validated experiment results are stored under:

```text
results/
```

Important files include:

```text
base_model_results.csv
fine_tuned_results.csv
comparison.csv
metrics.json
TRAINING_RESULTS.md
ARCHIVE_MANIFEST.md
```

The repository also stores lightweight provenance information used to verify experiment consistency.

---

## Experiment Provenance

The pipeline records and validates metadata such as:

- Base model
- Resolved model revision
- Generation settings
- Dataset hashes
- Configuration hash
- Run ID
- System prompt
- Result CSV hashes
- Adapter hashes
- GPU information
- Package versions
- Timestamp

Comparison logic rejects incompatible or modified experiment outputs.

This prevents unrelated experiment runs from being silently compared as though they belonged to the same training session.

---

## Results Website

The repository includes a dedicated frontend under:

```text
frontend/
```

The frontend presents the completed model experiment rather than performing GPU training itself.

It is designed to make the fine-tuning results easier to inspect through a production web interface.

The deployed application can present information such as:

- Base-model performance
- Fine-tuned performance
- Metric improvements
- Before/after comparisons
- Training configuration
- Experiment methodology
- Selected result examples
- Model-shift summary

Production deployment:

```text
https://lo-ra-fine-tuning-pipeline.vercel.app
```

---

## Run in Google Colab

The canonical training entry point is:

```text
notebooks/fine_tuning_colab.ipynb
```

### 1. Open the Notebook

Open the notebook in Google Colab.

---

### 2. Select a T4 GPU

Use:

```text
Runtime
→ Change runtime type
→ T4 GPU
```

---

### 3. Run Bootstrap

The notebook:

- Clones the repository when required
- Enters the project directory
- Verifies required source files
- Verifies configuration files
- Installs pinned dependencies
- Creates an isolated environment

---

### 4. Run Preflight Validation

Preflight verifies:

- CUDA availability
- GPU configuration
- NF4 forward pass
- Dataset integrity
- Token lengths
- Quantized Qwen loading

---

### 5. Generate Baseline Predictions

The normal FP16 base model generates predictions over all:

```text
70 test prompts
```

The baseline is intentionally separate from the quantized training model.

---

### 6. Train QLoRA Adapter

The training model is loaded in 4-bit, prepared for k-bit training, and wrapped with PEFT LoRA adapters.

Training uses:

```text
TRL SFTTrainer
```

for three epochs.

---

### 7. Save the Adapter

The trained PEFT adapter and tokenizer are stored in the generated artifacts directory.

---

### 8. Reload the Adapter

The pipeline reloads the saved adapter from disk before final evaluation.

This validates that the saved artifact can actually be reused.

---

### 9. Evaluate the Fine-Tuned Model

The same ordered 70 test prompts are run again.

The project then exports:

```text
fine_tuned_results.csv
comparison.csv
metrics.json
```

---

## Colab Environment

The pinned training stack targets:

```text
Python 3.10–3.12
PyTorch 2.8.0
CUDA 12.6
TRL 0.24.0
```

The environment requires several gigabytes of storage for:

- Model weights
- Python packages
- Model cache
- Checkpoints
- Adapter artifacts

The public Qwen base model does not require a Hugging Face token.

The first run requires internet access to download the model and dependencies.

---

## Reproducing a New Experiment

The archived experiment results should not be overwritten.

For a new run, create unique result and artifact paths.

The notebook supports generating a unique run identifier and separate output directories such as:

```text
artifacts/reproduction-<run-id>/
results/reproduction-<run-id>/
```

This keeps newly generated experiments separate from the validated archived run.

---

## Training Stages

The notebook performs the following stages:

```text
1. CUDA / T4 validation
2. Dependency installation
3. Dataset loading
4. Strict dataset validation
5. Qwen chat-template preparation
6. Base-model generation
7. GPU cleanup
8. 4-bit model loading
9. k-bit training preparation
10. LoRA adapter attachment
11. Three-epoch SFTTrainer training
12. Adapter saving
13. Adapter reloading
14. Fine-tuned generation
15. Base-vs-fine-tuned comparison
16. Metric export
17. Provenance validation
```

---

## Generated Training Artifacts

Actual training runs create a structure similar to:

```text
artifacts/
├── adapter/
│   └── PEFT adapter + tokenizer
│
├── trainer/
│   ├── checkpoints
│   └── trainer_state.json
│
├── training_config.yaml
│
└── run_metadata.json

results/
├── base_model_results.csv
├── base_model_results.metadata.json
├── fine_tuned_results.csv
├── fine_tuned_results.metadata.json
├── comparison.csv
└── metrics.json
```

Large adapter weights, checkpoints, downloaded models, and training archives are intentionally not stored in Git.

---

## Use the Saved Adapter

After restoring the adapter directory from the original experiment archive, the saved adapter can be attached to the same base-model revision.

```python
import json
import torch

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "artifacts/adapter"

with open(
    "results/provenance/run_metadata.json",
    encoding="utf-8",
) as stream:
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

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)

model.eval()
```

The notebook contains the complete Qwen chat-template formatting and generation workflow.

---

## Project Structure

```text
LoRA-Fine-Tuning-Pipeline/
│
├── configs/
│   └── training_config.yaml
│
├── data/
│   ├── dataset_summary.json
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── public/
│   ├── scripts/
│   ├── src/
│   ├── .env.example
│   ├── next.config.ts
│   ├── package.json
│   ├── package-lock.json
│   └── tsconfig.json
│
├── notebooks/
│   └── fine_tuning_colab.ipynb
│
├── results/
│   ├── base_model_results.csv
│   ├── base_model_results.metadata.json
│   ├── base_model_results.legacy-15c5e0cb695c.csv
│   ├── fine_tuned_results.csv
│   ├── fine_tuned_results.metadata.json
│   ├── comparison.csv
│   ├── metrics.json
│   ├── TRAINING_RESULTS.md
│   ├── ARCHIVE_MANIFEST.md
│   │
│   └── provenance/
│
├── src/
│   ├── __init__.py
│   ├── dataset_utils.py
│   ├── evaluation.py
│   ├── runtime.py
│   ├── pipeline.py
│   └── inference.py
│
├── tests/
│   └── test_pipeline.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements-test.txt
└── requirements.txt
```

Large training artifacts are intentionally excluded from Git.

Validated lightweight results and provenance remain version-controlled.

---

## Lightweight Tests

The CPU-safe test suite does not require:

- A GPU
- Model downloads
- QLoRA training

Install the lightweight test dependencies:

```bash
python -m pip install -r requirements-test.txt
```

Run:

```bash
python -m pytest -q tests
```

Tests cover areas including:

- Dataset schema validation
- Dataset row counts
- Split alignment
- Fact consistency
- Configuration validation
- Prompt formatting
- Evaluation metrics
- Result alignment
- Provenance validation
- Adapter-layout validation
- Token-length limits
- Module imports
- Notebook structural checks
- Notebook restart behavior
- Mocked pipeline operations

Passing the CPU tests does not establish that GPU training will succeed on every runtime.

---

## Technologies

### Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- LoRA
- QLoRA
- bitsandbytes
- TRL `SFTTrainer`

### Quantization

- 4-bit NF4
- Double quantization
- FP16 compute

### Data and Configuration

- pandas
- JSONL
- PyYAML

### Training Environment

- Google Colab
- NVIDIA Tesla T4

### Frontend

- Next.js
- TypeScript
- Vercel

### Testing

- Pytest

---

## Limitations

- The dataset is small
- The dataset is synthetic
- The dataset covers only one fictional product-support domain
- Test prompts are paraphrases of trained facts rather than unseen facts
- There is only one held-out test paraphrase per fact
- Lexical metrics can penalize correct paraphrases
- Lexical metrics can reward keyword overlap even when extra claims are incorrect
- The current evaluation does not include human judging
- Semantic judging is not currently included
- Contradiction-aware evaluation is not currently included
- Confidence calibration is not included
- Safety evaluation is not included
- FP16 baseline vs 4-bit tuned inference also introduces a quantization difference
- Colab GPU availability is not guaranteed
- Colab session duration is not guaranteed
- Large adapter and checkpoint files are not stored directly in Git

---

## Future Improvements

- Add semantic-similarity evaluation
- Add contradiction-aware local evaluation
- Add human evaluation
- Add explicit hallucination analysis
- Repeat training across multiple random seeds
- Report confidence intervals
- Expand the dataset
- Add harder negative examples
- Add ambiguous questions
- Add explicit out-of-domain refusal examples
- Evaluate additional Qwen model sizes
- Compare additional small open models
- Compare alternative LoRA ranks
- Compare different LoRA target-module configurations
- Measure training peak GPU memory
- Measure inference latency
- Measure generation throughput
- Track final adapter size
- Add CI for the CPU-safe test suite
- Add additional experiment visualizations
- Add richer qualitative result comparison to the frontend

---

## Why This Project Matters

Fine-tuning an LLM involves more than calling a training function.

A reproducible fine-tuning workflow requires:

- Dataset design
- Dataset validation
- Prompt formatting
- Baseline measurement
- GPU-aware model loading
- Quantization
- Parameter-efficient training
- Adapter persistence
- Controlled evaluation
- Experiment provenance
- Result validation
- Reproducibility safeguards

This project demonstrates practical AI Engineering concepts including:

- Instruction tuning
- LoRA
- QLoRA
- 4-bit model quantization
- PEFT adapters
- Hugging Face model loading
- Qwen chat templates
- TRL supervised fine-tuning
- GPU-memory optimization
- Dataset validation
- Base-model benchmarking
- Before/after evaluation
- Deterministic generation
- Experiment provenance
- Artifact validation
- Adapter reloading
- Reproducible experiment configuration
- CPU-safe pipeline testing
- Results visualization
- Production frontend deployment

The project shows how a small open model can be fine-tuned and evaluated through a reproducible engineering pipeline rather than treating fine-tuning as a one-off notebook experiment.

---

## Suggested CV Bullet

> Built a reproducible QLoRA fine-tuning and evaluation pipeline for Qwen2.5-1.5B-Instruct using 4-bit NF4 quantization, PEFT adapters, TRL, strict dataset validation, deterministic before/after benchmarking, and Google Colab T4 hardware.

---

## License

Released under the [MIT License](LICENSE).

The Qwen base model and third-party libraries remain subject to their respective licenses.

---

## Author

Developed by [Azoqoz](https://github.com/Azoqoz).
