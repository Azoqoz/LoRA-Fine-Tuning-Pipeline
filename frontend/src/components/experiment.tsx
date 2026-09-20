"use client";

import { useState } from "react";
import type experimentData from "../data/experiment.json";

type Data = typeof experimentData;
type Metric = keyof Data["metrics"]["base_model"];
const metricNames: Record<Metric, string> = {
  normalized_exact_match: "Normalized exact match",
  token_precision: "Token precision",
  token_recall: "Token recall",
  token_f1: "Token F1",
  keyword_overlap: "Keyword overlap",
  expected_fact_coverage: "Expected fact coverage",
};
const metricKeys = Object.keys(metricNames) as Metric[];
const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
const difference = (value: number) => `${value > 0 ? "+" : ""}${(value * 100).toFixed(2)} pp`;
const label = (value: string) => value.replaceAll("_", " ");
const evidence = [
  { id: "models_01", title: "The lightweight model, identified.", note: "Names Nova-Small instead of speculating about unrelated model families.", failure: false },
  { id: "limits_01", title: "A limit replaces a guess.", note: "Recalls the fictional Developer plan's 1,000-request hourly limit.", failure: false },
  { id: "config_01", title: "The default comes into focus.", note: "Corrects the default temperature from 1 to 0.7.", failure: false },
  { id: "cache_05", title: "Cache reuse, lost in translation.", note: "Regresses from explaining response reuse to an incorrect claim about rate-limit counting.", failure: true },
  { id: "models_08", title: "A higher score. Still incorrect.", note: "Answers an endpoint question with an unrelated cache-TTL fact. Lexical improvement is not correctness.", failure: true },
];

function SectionHeading({ number, title, children }: { number: string; title: string; children: React.ReactNode }) {
  return <div className="section-heading"><span className="eyebrow">{number} / FIELD NOTES</span><h2>{title}</h2><p>{children}</p></div>;
}

export default function Experiment({ data }: { data: Data }) {
  const [category, setCategory] = useState("all");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState("models_01");
  const [metric, setMetric] = useState<Metric>("token_f1");
  const rows = data.examples.filter((row) => (category === "all" || row.category === category) && row.question.toLowerCase().includes(search.toLowerCase().trim()));
  const active = rows.find((row) => row.id === selected) ?? rows[0];
  const index = active ? rows.indexOf(active) : -1;
  const { metrics, training, config } = data;

  function inspect(id: string) {
    setCategory("all");
    setSearch("");
    setSelected(id);
    document.getElementById("comparator")?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
    document.getElementById("case-selector")?.focus({ preventScroll: true });
  }

  return <>
    <a className="skip-link" href="#overview">Skip to experiment</a>
    <header className="masthead">
      <a className="wordmark" href="#overview" aria-label="Model Shift home">MODEL SHIFT<span className="mark" aria-hidden="true">↗</span></a>
      <nav aria-label="Sections"><a href="#comparator">Compare</a><a href="#categories">Categories</a><a href="#evidence">Evidence</a></nav>
      <a className="source-link" href="https://github.com/Azoqoz/LoRA-Fine-Tuning-Pipeline" target="_blank" rel="noreferrer">GitHub ↗</a>
    </header>
    <main>
      <section className="overview section" id="overview">
        <div className="intro-line"><span className="eyebrow">EXPERIMENT 01 / QLoRA</span><span className="eyebrow">RECORDED OUTPUTS · NOT LIVE INFERENCE</span></div>
        <div className="hero-title"><h1>Same model.<br/>A measurable <em>shift.</em></h1><div className="hero-note"><span className="delta-symbol" aria-hidden="true">Δ</span><p>Fine-tuning improved paraphrase recall on this synthetic experiment.</p><span className="mono">{metrics.base_model_id}</span></div></div>
        <div className="shift-head" aria-hidden="true"><span>BASE</span><span>→ Δ →</span><span>FINE-TUNED</span></div>
        <div className="headline-shift"><div className="headline-label">Normalized<br/>exact match</div><strong className="base">{percent(metrics.base_model.normalized_exact_match)}</strong><span className="shift-arrow" aria-hidden="true">→</span><strong className="tuned">{percent(metrics.fine_tuned_model.normalized_exact_match)}</strong></div>
        <div className="secondary-shift"><span>Token F1</span><strong className="base">{percent(metrics.base_model.token_f1)}</strong><span aria-hidden="true">→</span><strong className="tuned">{percent(metrics.fine_tuned_model.token_f1)}</strong><span className="delta-caption">{difference(metrics.delta_fine_tuned_minus_base.token_f1)}</span></div>
        <div className="experiment-strip"><div><strong>{data.exactMatches.base} → {data.exactMatches.tuned}<small> / {metrics.num_examples}</small></strong><span>exact reference matches</span></div><div><strong>{data.counts.improved}<small> improved · </small><span className="failure">{data.counts.worsened}</span><small> worsened</small></strong><span>per-example Token F1 · {data.counts.unchanged} unchanged</span></div><div><strong>{training.epochs}<small> epochs / </small>{training.steps}<small> steps</small></strong><span>{training.seconds} sec of training</span></div></div>
        <p className="footnote">70 held-out paraphrases of trained fictional facts. Measured recall, not unseen-fact generalization.</p>
      </section>

      <section className="section comparator" id="comparator">
        <SectionHeading number="01" title="Read the shift.">One question. Two recorded answers. Inspect what changed—and what did not.</SectionHeading>
        <div className="controls"><label>Category<select value={category} onChange={(event) => setCategory(event.target.value)}><option value="all">All categories</option>{data.categories.map((c) => <option key={c.id} value={c.id}>{label(c.id)}</option>)}</select></label><label className="search">Search questions<input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Try ‘cache’ or ‘temperature’" /></label><label className="case-select">Test case<select id="case-selector" value={active?.id ?? ""} disabled={!active} onChange={(event) => setSelected(event.target.value)}>{!active && <option value="">No matching cases</option>}{rows.map((row) => <option key={row.id} value={row.id}>{row.id} — {row.question}</option>)}</select></label></div>
        <div className="case-navigation"><span aria-live="polite">{rows.length} matching cases{active ? ` · ${index + 1} / ${rows.length}` : ""}</span><div><button disabled={index <= 0} onClick={() => setSelected(rows[index - 1].id)} aria-label="Previous test case">← Previous</button><button disabled={!active || index >= rows.length - 1} onClick={() => setSelected(rows[index + 1].id)} aria-label="Next test case">Next →</button></div></div>
        {active ? <>
          <div className="question-block"><span className="eyebrow">QUESTION / {active.id} / {label(active.category)}</span><h3>{active.question}</h3><div className="reference"><span className="eyebrow">EXPECTED RESPONSE</span><p>{active.expected}</p></div></div>
          <div className="response-pair"><article className="response base-response"><div className="response-heading"><span className="eyebrow base">BASE MODEL</span><span className="mono">FP16</span></div><p>{active.baseResponse}</p></article><article className="response tuned-response"><div className="response-heading"><span className="eyebrow tuned">FINE-TUNED</span><span className="mono">4-BIT + ADAPTER</span></div><p>{active.tunedResponse}</p></article></div>
          <div className="metric-table"><table><caption>Measured differences <span>Fine-tuned minus base · pp = percentage points</span></caption><thead><tr><th scope="col">Metric</th><th scope="col" className="base">Base</th><th scope="col" className="tuned">Fine-tuned</th><th scope="col">Δ</th></tr></thead><tbody>{["token_f1", ...metricKeys.filter((key) => key !== "token_f1")].map((key) => { const k = key as Metric; return <tr key={k} className={k === "token_f1" ? "emphasized" : ""}><th scope="row">{metricNames[k]}</th><td className="base">{percent(active.base[k])}</td><td className="tuned">{percent(active.tuned[k])}</td><td className={active.delta[k] < -1e-12 ? "failure" : ""}>{difference(active.delta[k])}</td></tr>; })}</tbody></table></div>
          <p className="footnote">Token F1 {active.change}. This describes lexical overlap, not a verdict on factual correctness.</p>
        </> : <div className="empty-state"><h3>No matching questions.</h3><p>Try a broader search or another category.</p><button onClick={() => { setSearch(""); setCategory("all"); }}>Clear filters ↗</button></div>}
      </section>

      <section className="section categories" id="categories">
        <SectionHeading number="02" title="Eight categories. Different distances.">Paired markers trace the measured change across each part of the fictional domain.</SectionHeading>
        <div className="category-toolbar"><div className="legend"><span className="base">● Base</span><span className="tuned">◆ Fine-tuned</span></div><label>Compare metric<select value={metric} onChange={(event) => setMetric(event.target.value as Metric)}>{metricKeys.map((key) => <option key={key} value={key}>{metricNames[key]}</option>)}</select></label></div>
        <div className="rail-scale" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div>
        <div className="category-bands">{data.categories.map((c) => <div className="category-band" key={c.id}><div className="category-name"><h3>{label(c.id)}</h3><span className="mono">n = {c.count}</span></div><div className="rail" aria-hidden="true"><span className="rail-connection" style={{ left: `${Math.min(c.base_model[metric], c.fine_tuned_model[metric]) * 100}%`, width: `${Math.abs(c.delta[metric]) * 100}%` }}/><span className="base-marker" style={{ left: `${c.base_model[metric] * 100}%` }}/><span className="tuned-marker" style={{ left: `${c.fine_tuned_model[metric] * 100}%` }}/></div><div className="rail-values"><span className="base"><span className="sr-only">Base </span>{percent(c.base_model[metric])}</span><span aria-hidden="true">→</span><span className="tuned"><span className="sr-only">Fine-tuned </span>{percent(c.fine_tuned_model[metric])}</span><strong>{difference(c.delta[metric])}</strong></div></div>)}</div>
      </section>

      <section className="section transformation" id="training">
        <SectionHeading number="03" title="Small adapters. A specific intervention.">Only adapters were trained. The quantized base weights remained frozen.</SectionHeading>
        <ol className="pipeline">{["Qwen2.5-1.5B-Instruct", "4-bit NF4 quantization", "LoRA adapters", `${training.epochs} epochs / ${training.steps} steps`, "Fine-tuned adapter", "70-prompt evaluation"].map((step, i) => <li key={step}><span className="mono">0{i + 1}</span><strong>{step}</strong>{i < 5 && <span className="pipeline-arrow" aria-hidden="true">↓</span>}</li>)}</ol>
        <div className="training-details"><div><h3>The intervention</h3><dl><dt>LoRA rank / alpha</dt><dd>{config.lora.r} / {config.lora.lora_alpha}</dd><dt>Dropout</dt><dd>{config.lora.lora_dropout}</dd><dt>Compute</dt><dd>{config.quantization.bnb_4bit_compute_dtype}</dd><dt>Double quantization</dt><dd>{config.quantization.bnb_4bit_use_double_quant ? "Enabled" : "Disabled"}</dd><dt>Learning rate</dt><dd>{config.training.learning_rate}</dd><dt>Target modules</dt><dd>{config.lora.target_modules.length}</dd></dl><p className="mono modules">{config.lora.target_modules.join(" · ")}</p></div><div><h3>The run</h3><dl><dt>Hardware</dt><dd>{training.gpu}</dd><dt>Train / validation / test</dt><dd>{training.train} / {training.validation} / {training.test}</dd><dt>Fictional facts</dt><dd>{training.facts}</dd><dt>Adapter parameters</dt><dd>{training.adapterParameters.toLocaleString("en-US")}</dd><dt>Training time</dt><dd>{training.seconds} sec</dd><dt>Seed</dt><dd>{config.seed}</dd></dl><p className="footnote">Parameter count derived from the archived adapter header, not separately logged. Runtime excludes setup, download, and inference.</p></div></div>
        <details className="provenance"><summary>Inspect experiment identity</summary><dl><dt>Run ID</dt><dd>{metrics.provenance.run_id}</dd><dt>Resolved model revision</dt><dd>{metrics.provenance.model_revision}</dd><dt>Metrics recorded</dt><dd>{metrics.created_at_utc}</dd><dt>Test dataset SHA-256</dt><dd>{metrics.provenance.dataset_hash}</dd><dt>Generation</dt><dd>Greedy · max {metrics.provenance.generation.max_new_tokens} new tokens · sequence limit {metrics.provenance.generation.max_length}</dd></dl></details>
      </section>

      <section className="section evidence" id="evidence"><SectionHeading number="04" title="The gains. And the cracks.">Three recovered facts. Two failures worth reading. Both belong in the result.</SectionHeading><div className="evidence-list">{evidence.map((item, i) => { const row = data.examples.find((e) => e.id === item.id)!; return <article className={`evidence-item ${item.failure ? "failure-evidence" : ""}`} key={item.id}><span className="evidence-index">0{i + 1}</span><div><span className="eyebrow">{item.failure ? "FAILURE EVIDENCE" : "REFERENCE RECOVERED"} / {item.id}</span><h3>{item.title}</h3><p>{item.note}</p><div className="evidence-quote"><span className="eyebrow">FINE-TUNED RESPONSE</span><blockquote>{row.tunedResponse}</blockquote></div><details><summary>Read question, reference & base response</summary><h4>Question</h4><p>{row.question}</p><h4>Expected response</h4><p>{row.expected}</p><h4>Base response</h4><p className="preserve-text">{row.baseResponse}</p></details></div><div className="evidence-score"><span className="eyebrow">TOKEN F1</span><strong><span className="base">{percent(row.base.token_f1)}</span><span aria-hidden="true">→</span><span className="tuned">{percent(row.tuned.token_f1)}</span></strong><button onClick={() => inspect(item.id)}>Inspect all metrics ↗</button></div></article>; })}</div></section>

      <section className="section limitations" id="limitations"><SectionHeading number="05" title="A result with boundaries.">This is a small, inspectable experiment—not a universal model ranking.</SectionHeading><ul><li>NovaAI is synthetic and fictional.</li><li>Test prompts are paraphrases of facts present during training.</li><li>Results measure paraphrase recall, not generalization to unseen facts.</li><li>Token-overlap metrics do not establish full factual correctness.</li><li>Base evaluation used FP16 while tuned inference used a 4-bit base plus adapter, so quantization is also part of the comparison.</li><li>Adapter reload evidence from the archived run is indirect.</li></ul></section>
    </main>
    <footer><a className="wordmark" href="#overview">MODEL SHIFT ↗</a><p>Base → Δ → Fine-tuned<br/><span>Static evidence. No model runtime. No generated answers.</span></p><a href="https://github.com/Azoqoz/LoRA-Fine-Tuning-Pipeline" target="_blank" rel="noreferrer">Explore the repository ↗</a></footer>
  </>;
}
