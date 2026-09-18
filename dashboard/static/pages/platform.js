/* Which model answered which question, on which platform, and what that cost.
 *
 * A dashboard that never names what it runs on is asking to be taken on trust.
 * Every row here is a call the cell actually makes, with the measurement that
 * justifies the choice next to it.
 */

import * as V from "../viz.js";
import { loadEvidence, refreshLive, state } from "../store.js";

export const meta = {
  id: "platform",
  title: "What it runs on",
  crumb: "Reference",
  group: "Reference",
  icon: "M4 7h16M4 12h16M4 17h8M18 15l2 2-2 2",
};

const ROWS = [
  {
    job: "Is this measurement good enough to weld on?",
    platform: "Nebius Token Factory",
    model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    cadence: "once per part",
    why: "right on 88 of 89 calls against the known answer, never a wasted look, and the cheapest of the three sizes that scored that",
    tone: "s1",
  },
  {
    job: "Which machine is going out of adjustment?",
    platform: "Nebius Token Factory",
    model: "deepseek-ai/DeepSeek-V4.1-Flash",
    cadence: "once per shift",
    why: "45 of 64 on the cases built to sit on the boundary, beats the arithmetic rule at p 0.035, about 3 s a diagnosis, and the cheapest of the three models that do",
    tone: "s3",
  },
  {
    job: "Writing the corrective action brief from fetched documents",
    platform: "Nebius Token Factory",
    model: "nvidia/Nemotron-3_5-Lightning",
    cadence: "when a fault is found",
    why: "the brief has to carry a citation on every claim, which is an instruction-following job",
    tone: "s2",
  },
  {
    job: "Judging the finished bead",
    platform: "Nebius Token Factory",
    model: "MiniMaxAI/MiniMax-M3",
    cadence: "once per part",
    why: "ties with DeepSeek V4.1 Flash and Gemma 3 27B on the fifty beads once the prompt says the picture is a rendering; most defects named right, no false alarms, 2.5 s a bead",
    tone: "",
  },
  {
    job: "Naming which machine is drifting, without a network",
    platform: "trained on Token Factory, runs on the cell's own GPU",
    model: "Qwen3 1.7B, fine-tuned",
    cadence: "once per shift, offline",
    why: "12 of 12 on cases the training never saw, against 8 of 12 for the 120B it replaces",
    tone: "s5",
  },
  {
    job: "Finding and reading what a standard actually says",
    platform: "Tavily",
    model: "search, then extract",
    cadence: "when a fault is found, and on request",
    why: "a snippet is enough to pick a page and not enough to quote a clause out of it",
    tone: "",
  },
  {
    job: "Rendering the cell, the parts and the depth it measures",
    platform: "NVIDIA Isaac Sim 6.0",
    model: "Replicator, headless",
    cadence: "per part",
    why: "colour, depth, normals and a semantic mask, which is what the geometry is fitted to",
    tone: "",
  },
];

export function render() {
  return `
    <p class="lede">
      Four sizes of Nemotron are served on Token Factory. All four were scored on the
      per-part question and the smallest one runs it; every other job goes to whichever
      model measured best on it, NVIDIA or not. Every row below is a call the cell actually
      makes. The last column is the measurement that decided it, not a preference.
    </p>

    <div class="tiles" id="pf-tiles"></div>

    <div class="grid">
      <div class="card w12">
        <header><h2>Every model call this cell makes</h2></header>
        <div class="hint">Every row is something the code actually calls, not something planned. The page
          that measured each one is linked from Findings.</div>
        <div id="pf-table"></div>
      </div>

      <div class="card w6">
        <header><h2>Model time per part, across the run</h2></header>
        <div class="hint">Time spent waiting on a model, per part, across the run. The cheap
          question is asked fifty times; the expensive one is asked once.</div>
        <div id="pf-time"></div>
        <footer id="pf-time-meta"></footer>
      </div>

      <div class="card w6">
        <header><h2>Platform limits met during the build</h2></header>
        <div class="hint">These limits shaped what got built. Anyone deciding whether to use the same
          platform needs to know them more than they need to hear the successes.</div>
        <div id="pf-limits"></div>
      </div>
    </div>`;
}

const LIMITS = [
  ["No NVIDIA vision model on Token Factory",
   "The account serves 24 models. Four are Nemotron and all four are text only. Two are "
   + "vision capable and neither is NVIDIA's, so the bead verdict runs on the best of "
   + "those while every reasoning call runs on Nemotron."],
  ["Nemotron is not a fine-tuning base",
   "The supported list is DeepSeek, Gemma, GPT-OSS, Qwen and Llama. Training the shift-level "
   + "diagnosis meant taking the smallest capable model off that list instead, which is why "
   + "the edge model is a Qwen and the baseline it beats is a Nemotron."],
  ["Structured output is ignored by this model",
   "Measured against the view-planning prompt: guided_json through extra_body does nothing, "
   + "response_format json_object does nothing, and reasoning_effort low is not enough. Only "
   + "reasoning_effort none returns the answer directly, which cut 900 tokens of preamble to 86."],
  ["The answer can arrive outside content",
   "Nemotron returns its reply in the reasoning field with content set to null. Reading only "
   + "content silently yields an empty string, and a planner that treats empty as accept will "
   + "report a perfect score having never asked anything."],
  ["Serving custom weights is in beta",
   "A model fine-tuned on Token Factory cannot be served there without requesting access, so "
   + "the adapter is pulled down and run on the workstation beside the cell. For this project "
   + "that turned out to be the better answer anyway."],
];

export async function mount() {
  const [ev] = await Promise.all([loadEvidence(), refreshLive()]);
  tiles(ev);
  table();
  time(ev);
  V.$("pf-limits").innerHTML = LIMITS.map(([a, b]) => `
    <div class="step" style="padding:13px 0 12px">
      <div class="t" style="margin-top:0">${V.esc(a)}</div>
      <div class="d">${V.esc(b)}</div>
    </div>`).join("");
}

function tiles(ev) {
  const live = state.live;
  const decisions = live?.decisions || [];
  const inspections = live?.inspections || [];
  const modelCalls = decisions.length + inspections.length;
  const tokens = decisions.reduce(
    (a, d) => a + (d.prompt_tokens || 0) + (d.completion_tokens || 0), 0);
  const nvidiaCalls = decisions.filter((d) => (d.model || "").startsWith("nvidia/")).length;
  const briefs = live?.briefs?.length ?? 0;

  V.$("pf-tiles").innerHTML = [
    tile("NVIDIA model sizes scored", "4", "s5", "Nano, Lightning, Super and Ultra on the same decisions; the smallest runs"),
    tile("Model calls in this run", modelCalls.toLocaleString(), "",
      `${nvidiaCalls} of them on an NVIDIA model`),
    tile("Tokens through Token Factory", tokens.toLocaleString(), "",
      "on the view-planning decisions alone"),
    tile("Trained on Token Factory", "1", "after",
      "a small fine-tuned model that now runs without a network"),
    tile("Cited briefs raised", String(briefs), "",
      "each one read from the source document, not its snippet"),
  ].join("");
}

const tile = (label, value, tone, sub) => `<div class="tile">
  <div class="label">${V.esc(label)}</div>
  <div class="value ${tone}">${value}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

function table() {
  V.$("pf-table").innerHTML = `<table>
    <thead><tr><th>The question</th><th>Where it runs</th><th>Model</th>
      <th>How often</th><th>Why this one</th></tr></thead>
    <tbody>${ROWS.map((r) => `<tr>
      <td>${V.esc(r.job)}</td>
      <td class="muted">${V.esc(r.platform)}</td>
      <td><span class="badge ${r.tone}">${V.esc(r.model)}</span></td>
      <td class="muted">${V.esc(r.cadence)}</td>
      <td class="muted">${V.esc(r.why)}</td>
    </tr>`).join("")}</tbody></table>`;
}

function time(ev) {
  if (!ev.router.available) return;
  const rows = [];
  const routed = ev.router.arms.find((a) => a.arm === "routed");
  const ultra = ev.router.arms.find((a) => a.arm === "ultra");
  rows.push({ label: "Routed view planning", value: routed.total_latency_s,
    display: `${routed.total_latency_s.toFixed(0)} s`, colour: "var(--series-5)",
    tip: `${routed.n} decisions across the run` });
  rows.push({ label: "Everything on Ultra", value: ultra.total_latency_s,
    display: `${ultra.total_latency_s.toFixed(0)} s`, colour: "var(--series-4)",
    tip: "the same decisions, on the largest model" });
  if (ev.inspection.available) {
    const total = ev.inspection.median_latency_ms * ev.inspection.n / 1000;
    rows.push({ label: "Bead inspection", value: total,
      display: `${total.toFixed(0)} s`, colour: "var(--series-1)",
      tip: `${ev.inspection.n} beads at a median ${V.fmt(ev.inspection.median_latency_ms, 0)} ms` });
  }
  const edge = (ev.drift.edge || []).find((r) => r.tier === "edge");
  if (edge) {
    const total = (edge.median_latency_ms || 0) * edge.n / 1000;
    rows.push({ label: "Drift diagnosis, on the cell", value: total,
      display: `${total.toFixed(0)} s`, colour: "var(--after)",
      tip: `${edge.n} runs at a median ${V.fmt(edge.median_latency_ms, 0)} ms, no network` });
  }
  V.paint(V.$("pf-time"), V.ranked(rows, { width: 440, rowH: 36, padL: 176 }));
  V.$("pf-time-meta").innerHTML =
    `The routing is worth <b>${(ultra.total_latency_s - routed.total_latency_s).toFixed(0)} s</b>
     across fifty parts, The shift-level question is asked once, so its time barely registers.`;
}
