# NovaAI QLoRA training results

## Experiment

Completed Colab run: cac35dc5-6efa-4c15-9754-37a4de551d5f.
Recorded start: 2026-09-19T11:20:03.555653+00:00.
Metrics timestamp: 2026-09-19T11:30:09.676486+00:00.

- Base: Qwen/Qwen2.5-1.5B-Instruct.
- Resolved revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- Hardware: Tesla T4, 15,637,086,208 bytes GPU memory; CUDA 12.6.
- Training: 3 epochs, 105 optimizer steps; trainer-recorded runtime 309.5443 seconds
  (about 5 minutes 10 seconds, not total setup/download/inference time).
- Final train loss: 0.3594090623514993 (whole-training average).
- Final validation loss: 0.13948692381381989.
- QLoRA: frozen 4-bit NF4 base, double quantization, float16 compute;
  PEFT LoRA rank 16, alpha 32, dropout 0.05, CAUSAL_LM, bias none.
- Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
- Batch sizes: 2 train / 2 validation, gradient accumulation 4 (effective batch 8).
- Learning rate: 0.0002; max sequence length: 512; fp16 enabled, bf16 disabled;
  completion-only loss; paged_adamw_8bit; validation/save every epoch; seed 42.
- Trainable parameter count was not separately recorded in logs.
  The adapter contains 392 finite FP32 LoRA tensors totaling **18,464,768 parameters**
  (derived from the safetensors header; all stored elements are nonzero).

NovaAI is synthetic and fictional. The 420 examples comprise 280 training,
70 validation and 70 test rows, covering the same 70 facts. Test questions are
held-out paraphrases of trained facts. This is **paraphrase recall**, not unseen-fact
generalization. Only adapters were trained.

## Validation

Dataset hashes match the repository byte-for-byte. Saved YAML, run metadata,
PEFT configuration, base revision, adapter hashes and prediction sidecars agree.
All 70 rows align with the test set; comparison predictions match the original
CSVs. No empty or obvious placeholder predictions were found. Every per-row,
overall, delta and category metric was independently recomputed using a separate
implementation and agrees within absolute tolerance 1e-12. Original result files
were copied without modifying their bytes. The original legacy baseline is also
preserved. The final adapter and checkpoint-105 are present; checkpoints at steps
35, 70 and 105 document three epochs.

Reload evidence is indirect: the matching adapter hash manifest, completed-run
flag and fine-tuned sidecar agree with the repository's separate-process predict
stage, which explicitly calls PeftModel.from_pretrained on the saved adapter.
The ZIP does not include an executed notebook, a code snapshot or a separate
reload event log. We cannot independently prove that call executed from the
archive alone; no new inference or retraining was performed during this audit.

## Overall evaluation

Scores below are percentages. Absolute changes are percentage points.
Relative changes use (fine-tuned - base) / base; exact-match relative change is
undefined because its baseline is zero.

| Metric | Base (%) | Fine-tuned (%) | Absolute change (pp) | Relative change (%) |
|---|---:|---:|---:|---:|
| Normalized exact match | 0.0000 | 58.5714 | +58.5714 | N/A |
| Token precision | 9.5314 | 79.3452 | +69.8138 | +732.46 |
| Token recall | 57.0301 | 80.8591 | +23.8289 | +41.78 |
| Token F1 | 15.9702 | 79.6366 | +63.6664 | +398.66 |
| Keyword Jaccard | 10.4926 | 75.4669 | +64.9743 | +619.24 |
| Expected-token coverage | 53.3311 | 81.6162 | +28.2851 | +53.04 |

Normalized exact matches increased from **0/70 to 41/70**.
Using per-example token F1 and tolerance 1e-12: **69 improved, 1 worsened,
0 unchanged**. This is a lexical-score classification, not a count of answers
proven factually correct. Shorter responses matching the reference wording help
precision and F1; some base answers were already substantively correct.

## Category results

| Category | N | Base F1 (%) | Fine-tuned F1 (%) | F1 change (pp) | Fine-tuned exact match (%) |
|---|---:|---:|---:|---:|---:|
| api_keys | 9 | 17.00 | 80.24 | +63.24 | 55.56 |
| authentication | 8 | 15.58 | 84.08 | +68.50 | 62.50 |
| configuration | 8 | 18.47 | 84.69 | +66.21 | 75.00 |
| deployment | 8 | 11.46 | 90.49 | +79.03 | 75.00 |
| models | 8 | 11.67 | 65.73 | +54.06 | 37.50 |
| rate_limits | 8 | 14.43 | 79.96 | +65.53 | 50.00 |
| semantic_cache | 11 | 16.21 | 70.34 | +54.13 | 45.45 |
| troubleshooting | 10 | 21.37 | 83.91 | +62.54 | 70.00 |

The complete six metrics for every category are preserved in [metrics.json](metrics.json).

## Representative examples

### models_01

Improved: identifies Nova-Small instead of speculating about unrelated model families.

Prompt: What model is the most lightweight choice on NovaAI?

Reference:

> `Nova-Small` is NovaAI's fastest and lightest model.

Base:

> NovaAI offers a variety of models, each with its own strengths depending on your specific needs. The most lightweight option typically refers to smaller models that require less computational resources for training or inference. These models generally have fewer parameters and thus consume less memory and processing power.
> 
> For example, if you're looking at models like BERT (Bidirectional Encoder Representations from Transformers), GPT-3 (Generative Pre-trained Transformer 3), or other large language models, they might be quite heavy due to their extensive parameter count. In contrast, smaller models such as RoBERTa, ALBERT, or even some versions of T5 (Text

Fine-tuned:

> `Nova-Small` is NovaAI's fastest and lightest model.

Token F1: 0.036364 → 1.000000.

### limits_01

Improved: supplies the correct fictional Developer-plan limit instead of denying a specific limit.

Prompt: What hourly rate limit applies to NovaAI Developer users?

Reference:

> The NovaAI Developer plan allows up to 1,000 requests per hour.

Base:

> As of my last update, there is no specific hourly rate limit for NovaAI Developer users. The pricing model for NovaAI typically includes a free tier with limited capabilities that developers can use without incurring any costs. For more detailed information on pricing and usage limits, it would be best to visit the NovaAI website or contact their customer support directly.

Fine-tuned:

> The NovaAI Developer plan allows up to 1,000 requests per hour.

Token F1: 0.109589 → 1.000000.

### config_01

Improved: corrects the default temperature from 1 to 0.7.

Prompt: What is the standard NovaAI generation temperature?

Reference:

> The default NovaAI `temperature` is 0.7.

Base:

> NovaAI uses a default generation temperature of 1 for most tasks, which helps in balancing creativity with coherence during text generation. However, this can be adjusted based on specific needs or preferences within the context of certain applications or projects. For more detailed guidance, it would be best to consult the NovaAI documentation or reach out directly to their support team.

Fine-tuned:

> The default NovaAI `temperature` is 0.7.

Token F1: 0.117647 → 1.000000.

### cache_05

Worsened: the base explains cache reuse; the tuned answer substitutes a different cache/rate-limit fact.

Prompt: What happens when a new NovaAI request is similar enough to one already cached?

Reference:

> NovaAI can reuse a cached response when a new request is sufficiently similar to a cached request.

Base:

> When a new NovaAI request is similar enough to an existing cache entry, NovaAI will use that cached response instead of generating a new one from scratch. This approach helps in reducing latency by reusing previously computed results, which can be particularly beneficial for frequently accessed data or queries. The system checks if the similarity between the current request and the cached response meets a certain threshold; once it does, the cached result is returned immediately without further processing.

Fine-tuned:

> A NovaAI semantic cache hit returns quickly without doubling the rate limit request count.

Token F1: 0.312500 → 0.193548.

### models_08

Still incorrect: the tuned answer discusses cache TTL instead of the common API endpoint. Its F1 rises slightly, illustrating why overlap is not correctness.

Prompt: What endpoint strategy does NovaAI use across its models?

Reference:

> All NovaAI models use the same API endpoint.

Base:

> NovaAI employs a multi-lingual approach in its model development, allowing it to understand and respond to queries in various languages. This includes English, Spanish, French, German, Italian, Portuguese, Russian, Chinese (Simplified and Traditional), Japanese, Korean, Hindi, Arabic, Indonesian, Polish, Turkish, Dutch, Swedish, Danish, Norwegian, Finnish, Hungarian, Czech, Slovakian, Slovenian, Croatian, Ukrainian, Bulgarian, Romanian, Greek, Serbian, Thai, Vietnamese, and others. The company also supports multiple dialects of these languages for more nuanced understanding.

Fine-tuned:

> NovaAI uses a fixed semantic cache TTL.

Token F1: 0.049383 → 0.133333.

## Artifacts and reproducibility

- [Run metadata](provenance/run_metadata.json): exact config, model revision,
  hashes, environment versions, GPU and run identity.
- [Saved training configuration](provenance/training_config.yaml).
- [Trainer state](provenance/trainer_state.json): loss history and timing.
- [PEFT configuration](provenance/adapter_config.json).
- [Comparison](comparison.csv), [base predictions](base_model_results.csv),
  [fine-tuned predictions](fine_tuned_results.csv) and their metadata sidecars.
- [Complete archive inventory and sizes](ARCHIVE_MANIFEST.md).

The final adapter is preserved locally under artifacts/adapter/ and remains
ignored by Git. Its weight file is 73,911,112 bytes, below 100 MiB, but keeping it
out of Git preserves a lightweight source/results repository. The original
novaai-run.zip is retained locally. Full checkpoints, optimizer/RNG/scaler states,
and caches are not copied into the versioned project. No Git LFS is introduced.
A fresh clone does not include trained weights: recover artifacts/adapter/ from
the original Colab archive, or produce a separate run using the documented workflow.

## Limitations

One seed, one small synthetic domain, one test paraphrase per fact. Lexical exact
match and token overlap do not establish full factual correctness, contradiction
handling or unseen-fact generalization. Baseline inference is FP16 whereas tuned
inference is 4-bit, so quantization is not controlled separately. Generation is
capped at 128 new tokens; some baseline responses are cut off. No blinded human
evaluation or independently logged adapter-reload event is available.
