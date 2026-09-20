import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";
import { parse } from "csv-parse/sync";
import { parse as parseYaml } from "yaml";

export const metricKeys = ["normalized_exact_match", "token_precision", "token_recall", "token_f1", "keyword_overlap", "expected_fact_coverage"];
export const classify = (delta) => delta > 1e-12 ? "improved" : delta < -1e-12 ? "worsened" : "unchanged";
const root = fileURLToPath(new URL("../../", import.meta.url));

export function transform(metrics, csv, report, config) {
  const records = parse(csv, { columns: true, skip_empty_lines: true, bom: true });
  const examples = records.map((row) => {
    const scores = (prefix) => Object.fromEntries(metricKeys.map((key) => {
      const raw = row[`${prefix}_${key}`];
      const value = Number(raw);
      if (raw === undefined || raw.trim() === "" || !Number.isFinite(value) || value < 0 || value > 1) throw new Error(`Invalid score ${row.fact_id}: ${prefix}_${key}`);
      return [key, value];
    }));
    const base = scores("base");
    const tuned = scores("fine_tuned");
    const delta = Object.fromEntries(metricKeys.map((key) => [key, tuned[key] - base[key]]));
    for (const key of ["fact_id", "category", "instruction", "expected_response", "base_model_response", "fine_tuned_response"]) {
      if (!row[key]?.trim()) throw new Error(`Missing ${key}`);
    }
    return { id: row.fact_id, category: row.category, question: row.instruction, expected: row.expected_response, baseResponse: row.base_model_response, tunedResponse: row.fine_tuned_response, base, tuned, delta, change: classify(delta.token_f1) };
  });
  if (examples.length !== metrics.num_examples || new Set(examples.map((e) => e.id)).size !== examples.length) throw new Error("Invalid example count or duplicate fact IDs");
  const categories = Object.entries(metrics.by_category).map(([id, values]) => ({ id, ...values }));
  function checkAverages(rows, values) {
    for (const key of metricKeys) {
      for (const [side, aggregate] of [["base", "base_model"], ["tuned", "fine_tuned_model"]]) {
        const mean = rows.reduce((sum, row) => sum + row[side][key], 0) / rows.length;
        if (Math.abs(mean - values[aggregate][key]) > 1e-12) throw new Error(`Aggregate mismatch: ${key}`);
      }
      const recordedDelta = (values.delta ?? values.delta_fine_tuned_minus_base)[key];
      if (Math.abs(recordedDelta - (values.fine_tuned_model[key] - values.base_model[key])) > 1e-12) throw new Error(`Delta mismatch: ${key}`);
    }
  }
  checkAverages(examples, metrics);
  if (categories.reduce((sum, c) => sum + c.count, 0) !== examples.length) throw new Error("Category count mismatch");
  for (const category of categories) {
    const rows = examples.filter((e) => e.category === category.id);
    if (rows.length !== category.count || !rows.length) throw new Error("Category alignment mismatch");
    checkAverages(rows, category);
  }
  const capture = (pattern) => {
    const match = report.match(pattern);
    if (!match) throw new Error(`Training report field missing: ${pattern}`);
    return match[1];
  };
  const training = {
    epochs: Number(capture(/Training: (\d+) epochs/)),
    steps: Number(capture(/epochs, (\d+) optimizer steps/)),
    seconds: Number(capture(/runtime ([\d.]+) seconds/)),
    gpu: capture(/Hardware: ([^,]+),/),
    cuda: capture(/CUDA ([\d.]+)\./),
    adapterParameters: Number(capture(/\*\*([\d,]+) parameters\*\*/).replaceAll(",", "")),
    train: Number(capture(/comprise (\d+) training/)),
    validation: Number(capture(/(\d+) validation and/)),
    test: Number(capture(/(\d+) test rows/)),
    facts: Number(capture(/same (\d+) facts/)),
  };
  if (training.epochs !== config.training.num_train_epochs || training.test !== examples.length || config.model.base_model_id !== metrics.base_model_id) throw new Error("Configuration/report mismatch");
  const counts = { improved: 0, worsened: 0, unchanged: 0 };
  examples.forEach((example) => counts[example.change]++);
  return { metrics, config, training, examples, categories, counts, exactMatches: { base: examples.filter((e) => e.base.normalized_exact_match === 1).length, tuned: examples.filter((e) => e.tuned.normalized_exact_match === 1).length } };
}

export function loadExperiment() {
  const read = (path) => readFileSync(resolve(root, path), "utf8");
  return transform(JSON.parse(read("results/metrics.json")), read("results/comparison.csv"), read("results/TRAINING_RESULTS.md"), parseYaml(read("configs/training_config.yaml")));
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const data = loadExperiment();
  const target = resolve(root, "frontend/src/data/experiment.json");
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, JSON.stringify(data));
  console.log(`Validated ${data.examples.length} examples, ${data.categories.length} categories; generated static data.`);
}
