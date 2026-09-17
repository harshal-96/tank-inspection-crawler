/* The routing study: which model should answer, and what a wrong answer costs. */

import * as V from "../viz.js";
import { loadEvidence, refreshLive, state } from "../store.js";

export const meta = {
  id: "routing",
  title: "Model routing",
  crumb: "Study 2",
  group: "Studies",
  tag: "195",
  icon: "M4 6h6m0 0a2 2 0 104 0m-4 0a2 2 0 114 0m0 0h6M4 18h6m0 0a2 2 0 104 0m-4 0a2 2 0 114 0m0 0h6",
};

export function render() {
  return `
    <p class="lede">
      After the cell looks at a part it has to answer one question: is this measurement
      good enough to weld on, or is another look worth the cycle time? The question is
      cheap, so it goes to the smallest model first. This study asks whether that is
      defensible, by scoring every arm against what a second look <b>actually</b> did to
      the fit rather than against a bigger model's opinion.
    </p>

    <div class="tiles" id="rt-tiles"></div>

    <div class="grid">
      <div class="card w7">
        <header><h2>How often each arm made the right call</h2></header>
        <div class="hint">On this part mix the first look was always good enough, so the correct
          call was always to accept. An arm that asks for a second look is spending cycle time to
          buy nothing.</div>
        <div id="rt-accuracy"></div>
      </div>

      <div class="card w5">
        <header><h2>Looks per part against wall-clock</h2></header>
        <div class="hint">Every mark is one arm. Down and left is better: fewer looks, less time.
          Each arm is labelled directly, so the position carries the reading and colour only
          confirms it.</div>
        <div id="rt-cost"></div>
        <footer id="rt-cost-meta"></footer>
      </div>

      <div class="card w12">
        <header><h2>The disagreements, laid out rather than averaged</h2></header>
        <div class="hint">One column per decision, one row per arm. Green is a correct call, red a
          wrong one. The block of red is what over-caution looks like when it is not averaged into
          a percentage.</div>
        <div id="rt-grid"></div>
        <div class="legend">
          <span><i style="background:var(--good)"></i>accepted, correctly</span>
          <span><i style="background:var(--critical)"></i>asked for another look that was not needed</span>
        </div>
      </div>

      <div class="card w6">
        <header><h2>What the cell actually routed to</h2></header>
        <div class="hint">The decisions in the database, from the fifty-part run.</div>
        <div id="rt-live"></div>
        <footer id="rt-live-meta"></footer>
      </div>

      <div class="card w6">
        <header><h2>What the model said, in its own words</h2></header>
        <div class="hint">A verdict with no reasoning attached is a verdict nobody can audit.</div>
        <div id="rt-reasons"></div>
      </div>
    </div>`;
}

let ev = null;

export async function mount() {
  [ev] = await Promise.all([loadEvidence(), refreshLive()]);
  if (!ev.router.available) return;
  tiles();
  accuracy();
  cost();
  grid();
  live();
}

export function update() { live(); }

function arm(name) { return ev.router.arms.find((a) => a.arm === name); }

function tiles() {
  const routed = arm("routed"), ultra = arm("ultra"), superb = arm("super");
  const saved = ultra.total_latency_s - routed.total_latency_s;
  V.$("rt-tiles").innerHTML = [
    tile("Routed, correct calls", `${routed.correct}/${routed.n}`, "after",
      `${V.pct(routed.accuracy, 1)} against ground truth`),
    tile("Looks per decision", routed.looks_per_decision.toFixed(2), "good",
      `the floor is 1.00, Super sat at ${superb.looks_per_decision.toFixed(2)}`),
    tile("Wall clock for the run", `${routed.total_latency_s.toFixed(0)}`, "after",
      `seconds, against ${ultra.total_latency_s.toFixed(0)} s on the largest model`, "s"),
    tile("Time the routing saved", `${saved.toFixed(0)}`, "good",
      `${V.pct(saved / ultra.total_latency_s, 0)} of the largest model's run`, "s"),
    tile("Accuracy gap, largest model", V.pct(ultra.accuracy, 0), "bad",
      `${ultra.correct}/${ultra.n}. Bigger is not the same as better at this question.`),
  ].join("");
}

const tile = (label, value, tone, sub, unit = "") => `<div class="tile">
  <div class="label">${V.esc(label)}</div>
  <div class="value ${tone}">${value}${unit ? `<span class="unit">${unit}</span>` : ""}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

function accuracy() {
  const rows = ev.router.arms.map((a) => ({
    label: a.label,
    value: a.accuracy,
    display: `${a.correct}/${a.n}`,
    colour: V.ARM_COLOUR[a.arm] || V.SERIES[0],
    tip: `<b>${a.label}</b><br>
      <span class="k">right on</span> <b>${a.correct}</b> <span class="k">of</span> <b>${a.n}</b> <span class="k">decisions</span><br>
      <span class="k">asked for a second look</span> <b>${a.said_look}</b> <span class="k">times, needed</span> <b>${a.should_look}</b><br>
      <span class="k">median</span> <b>${V.fmt(a.median_latency_ms, 0)} ms</b>`,
  }));
  V.paint(V.$("rt-accuracy"), V.ranked(rows, { width: 560, rowH: 42, padL: 176, max: 1 }));
}

function cost() {
  const arms = ev.router.arms;
  const w = 400, h = 300, padL = 46, padB = 40, padT = 26, padR = 76;
  const hiL = Math.max(...arms.map((a) => a.total_latency_s)) * 1.12;
  const hiK = Math.max(...arms.map((a) => a.looks_per_decision)) * 1.08;
  const [topL, ticksL] = V.nice(hiL, 4);
  const xOf = (v) => padL + (v / topL) * (w - padL - padR);
  const yOf = (v) => h - padB - ((v - 0.95) / (hiK - 0.95)) * (h - padB - padT);

  let out = V.gridX(padT, h - padB, xOf, ticksL, "", (v) => v.toFixed(0));
  [1.0, 1.25, 1.5, 1.75, 2.0].filter((v) => v <= hiK).forEach((v) => {
    out += `<line class="gridline" x1="${padL}" y1="${yOf(v)}" x2="${w - padR}" y2="${yOf(v)}"/>
      <text class="axis-label" x="${padL - 7}" y="${yOf(v) + 3.5}" text-anchor="end">${v.toFixed(2)}</text>`;
  });
  const placed = [];
  arms.forEach((a) => {
    const x = xOf(a.total_latency_s), y = yOf(a.looks_per_decision);
    // Nudge a label that would land on one already written.
    let ly = y + 4;
    while (placed.some((q) => Math.abs(q.x - x) < 72 && Math.abs(q.y - ly) < 15)) ly += 16;
    placed.push({ x, y: ly });
    out += V.dot(x, y, 6, V.ARM_COLOUR[a.arm],
      `data-tip="<b>${a.label}</b><br><span class='k'>looks per decision</span> <b>${a.looks_per_decision.toFixed(2)}</b><br>
        <span class='k'>total</span> <b>${a.total_latency_s.toFixed(0)} s</b><br>
        <span class='k'>correct</span> <b>${a.correct}/${a.n}</b>"`);
    if (Math.abs(ly - (y + 4)) > 1) {
      out += `<line x1="${x + 7}" y1="${y}" x2="${x + 12}" y2="${ly - 4}" stroke="var(--line)" stroke-width="1"/>`;
    }
    out += `<text x="${x + 13}" y="${ly}" font-size="10.5" fill="var(--text-secondary)"
      font-family="var(--sans)">${V.esc(a.arm)}</text>`;
  });
  out += V.axisTitle(padL, h - 8, "seconds for the whole run");
  out += V.axisTitle(padL - 38, padT - 6, "looks / part");
  V.paint(V.$("rt-cost"), V.svg(w, h, out));

  const routed = arm("routed"), ultra = arm("ultra");
  V.$("rt-cost-meta").innerHTML =
    `The routed arm sits in the bottom left: <b>${routed.looks_per_decision.toFixed(2)}</b> looks per part
     in <b>${routed.total_latency_s.toFixed(0)} s</b>, and it was right more often than the arm that
     cost <b>${ultra.total_latency_s.toFixed(0)} s</b>. Caution is not free here, and on this question it
     did not buy accuracy either.`;
}

function grid() {
  const order = ["routed", "nano", "lightning", "super", "ultra"];
  const cases = ev.router.grid;
  const cell = 15, gap = 2, padL = 92, padT = 22;
  const w = padL + cases.length * (cell + gap) + 10;
  const h = padT + order.length * (cell + gap) + 26;
  let out = "";
  order.forEach((armName, r) => {
    const a = arm(armName);
    const y = padT + r * (cell + gap);
    out += `<text class="axis-label" x="${padL - 10}" y="${y + cell / 2 + 3.5}" text-anchor="end"
      font-size="10" fill="${V.ARM_COLOUR[armName]}">${armName}</text>`;
    cases.forEach((c, i) => {
      const calls = c.calls[armName] || [];
      const call = calls[0];
      const x = padL + i * (cell + gap);
      const colour = !call ? "var(--surface-2)" : call.correct ? "var(--good)" : "var(--critical)";
      out += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" rx="3.5" fill="${colour}"
        opacity="${call ? 0.85 : 0.35}"
        data-tip="<b>${armName}</b> <span class='k'>on decision ${i + 1}</span><br>
          ${call ? (call.said_look ? "asked for another look" : "accepted the measurement")
                 : "no record"}<br>
          <span class='k'>correct call was</span> <b>${c.should_look ? "look again" : "accept"}</b>"/>`;
    });
    out += `<text x="${padL + cases.length * (cell + gap) + 8}" y="${y + cell / 2 + 3.5}"
      font-size="10" fill="var(--text-secondary)" font-family="var(--mono)">${a.correct}/${a.n}</text>`;
  });
  out += V.axisTitle(padL, h - 8, `${cases.length} decisions, in the order they were taken`);
  V.paint(V.$("rt-grid"), V.svg(w, h, out));
}

function live() {
  const rows = state.live?.decisions || [];
  const host = V.$("rt-live");
  if (!host) return;
  if (!rows.length) { host.innerHTML = `<div class="empty">No decisions recorded.</div>`; return; }

  const byTier = new Map();
  rows.forEach((d) => {
    const t = (d.tier || "fixed").toLowerCase();
    const s = byTier.get(t) || { n: 0, looks: 0, ms: 0, tokens: 0, accepts: 0 };
    s.n++; s.looks += d.after_looks || 0; s.ms += d.latency_ms || 0;
    s.tokens += (d.prompt_tokens || 0) + (d.completion_tokens || 0);
    s.accepts += d.accept ? 1 : 0;
    byTier.set(t, s);
  });
  const segs = [...byTier.entries()].map(([t, s]) => ({
    label: t, value: s.n, colour: V.ARM_COLOUR[t] || "var(--surface-3)",
    tip: `<b>${t}</b>: ${s.n} decisions, ${(s.looks / s.n).toFixed(2)} looks each,
      median ${(s.ms / s.n).toFixed(0)} ms, accepted ${s.accepts}`,
  }));
  const totalLooks = rows.reduce((a, d) => a + (d.after_looks || 0), 0);
  const totalMs = rows.reduce((a, d) => a + (d.latency_ms || 0), 0);
  const tokens = rows.reduce((a, d) => a + (d.prompt_tokens || 0) + (d.completion_tokens || 0), 0);

  host.innerHTML = `
    ${V.stackedRow(segs, { width: 430, height: 18 })}
    <div class="legend">${segs.map((s) =>
      `<span><i style="background:${s.colour}"></i>${V.esc(s.label)} &middot; ${s.value}</span>`).join("")}</div>
    <table class="plain" style="margin-top:16px"><tbody>
      ${kv("Decisions", rows.length)}
      ${kv("Looks taken", `${totalLooks} (${(totalLooks / rows.length).toFixed(2)} per part)`)}
      ${kv("Accepted first look", `${rows.filter((d) => d.accept).length} of ${rows.length}`)}
      ${kv("Median latency", `${median(rows.map((d) => d.latency_ms)).toFixed(0)} ms`)}
      ${kv("Time in the model", `${(totalMs / 1000).toFixed(1)} s across the run`)}
      ${kv("Tokens", tokens.toLocaleString())}
    </tbody></table>`;
  V.bindTips(host);

  V.$("rt-live-meta").innerHTML =
    `Every row carries the model that answered and why it was picked, so a decision can be traced
     back to a tier rather than to "the AI".`;

  const reasons = rows.slice(0, 6);
  V.$("rt-reasons").innerHTML = reasons.map((d) => `
    <div style="padding:9px 0;border-bottom:1px solid var(--line-soft)">
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
        <span class="badge ${V.ARM_CLASS[(d.tier || "").toLowerCase()] || ""}">${V.esc(d.tier || "fixed")}</span>
        <span class="badge ${d.accept ? "good" : "warn"}">${d.accept ? "accept" : "look again"}</span>
        <span class="muted mono" style="font-size:11px">part #${d.part_id}</span>
        <span class="grow" style="flex:1"></span>
        <span class="muted mono" style="font-size:11px">${V.fmt(d.latency_ms, 0)} ms</span>
      </div>
      <div class="quote">${V.esc(d.reason || "")}</div>
    </div>`).join("");
}

const kv = (k, v) => `<tr><td class="muted" style="width:150px">${V.esc(k)}</td>
  <td class="mono">${V.esc(String(v))}</td></tr>`;

function median(values) {
  const seq = values.filter((v) => v != null).sort((a, b) => a - b);
  return seq.length ? seq[Math.floor(seq.length / 2)] : 0;
}
