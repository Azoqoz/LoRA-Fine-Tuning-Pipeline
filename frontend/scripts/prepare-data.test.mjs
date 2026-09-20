import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { parse } from "yaml";
import { classify, loadExperiment, transform } from "./prepare-data.mjs";

const read = (path) => readFileSync(new URL(`../../${path}`, import.meta.url), "utf8");
test("all real rows and aggregate metrics align", () => {
  const data = loadExperiment();
  assert.equal(data.examples.length, 70);
  assert.equal(data.categories.length, 8);
  assert.deepEqual(data.counts, { improved: 69, worsened: 1, unchanged: 0 });
  assert.deepEqual(data.exactMatches, { base: 0, tuned: 41 });
  assert.equal((data.metrics.fine_tuned_model.normalized_exact_match * 100).toFixed(2), "58.57");
  assert.equal((data.metrics.base_model.token_f1 * 100).toFixed(2), "15.97");
  assert.equal((data.metrics.fine_tuned_model.token_f1 * 100).toFixed(2), "79.64");
  assert.equal(data.training.seconds, 309.5443);
  assert.equal(data.training.adapterParameters, 18464768);
});
test("quoted multiline responses remain intact", () => {
  const row = loadExperiment().examples.find((e) => e.id === "models_01");
  assert.ok(row.baseResponse.includes("\n"));
  assert.ok(row.baseResponse.includes("BERT"));
  assert.equal(row.tunedResponse, "`Nova-Small` is NovaAI's fastest and lightest model.");
});
test("F1 classification respects numeric tolerance", () => {
  assert.equal(classify(1e-13), "unchanged");
  assert.equal(classify(-1e-13), "unchanged");
  assert.equal(classify(0.1), "improved");
  assert.equal(classify(-0.1), "worsened");
});
test("failures are retained, including lexical improvement without correctness", () => {
  const data = loadExperiment();
  assert.equal(data.examples.find((e) => e.id === "cache_05").change, "worsened");
  const endpoint = data.examples.find((e) => e.id === "models_08");
  assert.equal(endpoint.change, "improved");
  assert.match(endpoint.tunedResponse, /cache TTL/);
});
test("modified aggregates fail closed", () => {
  const metrics = JSON.parse(read("results/metrics.json"));
  metrics.fine_tuned_model.token_f1 = 0;
  assert.throws(() => transform(metrics, read("results/comparison.csv"), read("results/TRAINING_RESULTS.md"), parse(read("configs/training_config.yaml"))), /Aggregate mismatch/);
});
