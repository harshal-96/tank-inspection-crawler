/* The routing study: which model should answer, and what a wrong answer costs. */

import * as V from "../viz.js";
import { loadEvidence, refreshLive, state } from "../store.js";

export const meta = {
  id: "routing",
  title: "Model routing",
  crumb: "Stage 2",
  group: "What happens to a part",
  tag: "195",
  icon: "M4 6h6m0 0a2 2 0 104 0m-4 0a2 2 0 114 0m0 0h6M4 18h6m0 0a2 2 0 104 0m-4 0a2 2 0 114 0m0 0h6",
};

export function render() {
  return `
    <p class="lede">
      Once the cell has photographed a part it has to decide one thing: is this
      measurement good enough to weld on, or is it worth taking another look? Another look
      costs cycle time. The question is cheap to ask, so it goes to the smallest model
      first. This page tests whether that is the right call, by scoring every model against
      what a second look actually achieved rather than against a bigger model's opinion.
    </p>

    <div class="tiles" id="rt-tiles"></div>

    <div class="grid">
            <div class="card w12">
        <header><h2>Three of four sizes were right every time; the smallest one runs</h2></header>
        <div class="hint" id="rt-accuracy-hint">Scored against what a second look actually
          achieved, under the prompt the cell runs. A model can be wrong two ways: ask for a
          look it did not need, which costs a robot move, or accept a view that failed, which
          welds the part blind.</div>
        <div id="rt-accuracy"></div>
      </div>

      

      <div class="card w12">
        <header><h2>Every decision, model by model</h2></header>
                <div class="hint">One square per decision, one row per model. Green is a correct call;
          red means it asked for another look it did not need. The 120B is the one model that
          still asks with the boundary stated.</div>
        <div id="rt-grid"></div>
        <div class="legend">
          <span><i style="background:var(--good)"></i>right call</span>
          <span><i style="background:var(--critical)"></i>wrong call: a look it did not need, or a failed view accepted</span>
        </div>
      </div>

      <div class="card w12">
                <header><h2>Why the big models once scored zero: the prompt, not the model</h2></header>
        <div class="hint">The first prompt said to accept a single view only if its signals were
          &ldquo;excellent&rdquo;, and never said what excellent meant. Every part starts with
          one view, so read literally that asks for a second look every time. Here the same
          models get the same decisions with one sentence added that gives the boundary a
          number. That sentence is now in the prompt the cell runs, and the cards above are
          measured under it.</div>
        <div id="rt-prompt"></div>
        <div class="legend">
          <span><i style="background:var(--before)"></i>prompt as written</span>
          <span><i style="background:var(--after)"></i>boundary stated</span>
        </div>
        <footer id="rt-prompt-meta"></footer>
      </div>

      <div class="card w6">
        <header><h2>What the cell actually routed to</h2></header>
        <div class="hint">The decisions in the database, from the fifty-part run.</div>
        <div id="rt-live"></div>
        <footer id="rt-live-meta"></footer>
      </div>

      <div class="card w6">
        <header><h2>What the model said, in its own words</h2></header>
        <div class="hint">What each model said, in its own words. A verdict with no reasoning attached cannot be checked by anyone. The signal names in the quotes: included_angle_deg is the angle between the two fitted faces, square is 90; inlier_balance is how evenly both faces were seen, 1 is perfectly even; the fit RMS values are how far each face strayed from a flat plane, in millimetres; path_length_mm is how much of the joint the view covered.</div>
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
  grid();
  promptAudit();
  live();
}

export function update() { live(); }


/** The same decisions, with and without a number in the prompt. */
function promptAudit() {
  const host = V.$("rt-prompt");
  const vp = ev.view_prompt;
  const models = (vp && vp.available ? vp.models : [])
    .filter((m) => m.as_written && m.stated);
  if (!models.length) {
    host.innerHTML = `<div class="empty">The prompt audit has not been run.</div>`;
    return;
  }
  const n = models[0].as_written.n;
  const w = 680, rowH = 46, padL = 210, padR = 78, padT = 30;
  const h = padT + models.length * rowH + 34;
  const xOf = (v) => padL + (v / n) * (w - padL - padR);

  let out = V.gridX(padT - 10, h - 34, xOf,
                    [0, n * 0.25, n * 0.5, n * 0.75, n], "", (v) => Math.round(v));
  out += V.axisTitle(padL, padT - 16, `decisions called correctly, out of ${n}`);

  models.forEach((m, i) => {
    const y = padT + i * rowH + rowH / 2;
    const a = xOf(m.as_written.correct), b = xOf(m.stated.correct);
    const tip = `<b>${V.esc(m.name)}</b><br><span class='k'>as written</span> `
      + `<b>${m.as_written.correct}/${m.as_written.n}</b><br>`
      + `<span class='k'>boundary stated</span> <b>${m.stated.correct}/${m.stated.n}</b>`;
    out += `<g data-tip="${V.esc(tip)}">
      <rect x="${padL - 6}" y="${y - rowH / 2}" width="${w - padL - padR + 12}"
        height="${rowH}" fill="transparent"/>
      <text class="axis-label" x="${padL - 14}" y="${y + 3.5}" text-anchor="end"
        font-size="10">${V.esc(m.name)}</text>
      <line x1="${a}" y1="${y}" x2="${b}" y2="${y}" stroke="var(--hair)" stroke-width="2"/>
      ${V.dot(a, y, 5, V.BEFORE)}
      ${V.dot(b, y, 5, V.AFTER)}
      <text x="${Math.max(a, b) + 14}" y="${y + 4}" font-size="11.5" font-weight="600"
        fill="var(--text-secondary)">${m.as_written.correct} &rarr; ${m.stated.correct}</text>
    </g>`;
  });
  V.paint(host, V.svg(w, h, out));

  const moved = models.map((m) => m.stated.correct - m.as_written.correct);
  const best = models[moved.indexOf(Math.max(...moved))];
  V.$("rt-prompt-meta").innerHTML =
    `One sentence moved <b>${V.esc(best.name)}</b> from <b>${best.as_written.correct}</b> to
     <b>${best.stated.correct}</b> of ${n}. The small model is still the right one to
     run here: it reaches the correct call without the boundary spelled out. But the larger
     models were not being more cautious. They were following an instruction that had no
     number in it.`;
}

function arm(name) { return ev.router.arms.find((a) => a.arm === name); }

function tiles() {
  const routed = arm("routed"), ultra = arm("ultra"), superb = arm("super");
  const saved = ultra.total_latency_s - routed.total_latency_s;
  V.$("rt-tiles").innerHTML = [
        tile("Right calls, as the cell routes them", `${routed.correct}/${routed.n}`, "after",
      `${V.pct(routed.accuracy, 1)} against the known answer`),
        tile("Time for the run", `${routed.total_latency_s.toFixed(0)}`, "after",
      `seconds, against ${ultra.total_latency_s.toFixed(0)} s for the largest model, `
      + `${V.pct(saved / ultra.total_latency_s, 0)} less`, "s"),
    tile("Camera looks per part", routed.looks_per_decision.toFixed(2), "good",
      `one is the least possible; the 120B took ${superb.looks_per_decision.toFixed(2)}`),
    tile("Parts welded blind", `${routed.blind_accepts}`, routed.blind_accepts ? "bad" : "good",
      `of ${routed.should_look} decisions where a look was needed`),
  ].join("");
  const p = ev.router.points;
  const hint = V.$("rt-accuracy-hint");
  if (p && hint) hint.insertAdjacentHTML("beforeend",
    ` ${p.n} decisions: ${p.replayed} replayed from the campaign and ${p.first_view} where a`
    + ` part is seen first from one of its five angles. A look was needed on ${p.look_needed}:`
    + ` views that failed to measure, or landed tens of millimetres out from a steep angle.`);
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

function grid() {
  const order = ["routed", "nano", "lightning", "super", "ultra"];
  const cases = ev.router.grid;
  // One column per decision, not per part. Seventeen parts were looked at more
  // than once and each of those gave one to four decision points, thirty-nine
  // in all. Drawing only the first per part made the caption say 17 under rows
  // that add up to 39, and every square carries its own verdict anyway.
  const columns = [];
  cases.forEach((c) => {
    const depth = Math.max(0, ...Object.values(c.calls).map((v) => v.length));
    for (let k = 0; k < depth; k++) columns.push({ trial: c.trial, k, calls: c.calls });
  });
  const cell = 15, gap = 2, padL = 92, padT = 22;
  const w = padL + columns.length * (cell + gap) + 10;
  const h = padT + order.length * (cell + gap) + 26;
  let out = "";
  order.forEach((armName, r) => {
    const a = arm(armName);
    const y = padT + r * (cell + gap);
    out += `<text class="axis-label" x="${padL - 10}" y="${y + cell / 2 + 3.5}" text-anchor="end"
      font-size="10" fill="${V.ARM_COLOUR[armName]}">${armName}</text>`;
    columns.forEach((col, i) => {
      const call = (col.calls[armName] || [])[col.k];
      const x = padL + i * (cell + gap);
      const colour = !call ? "var(--band)" : call.correct ? "var(--good)" : "var(--critical)";
      // The right call for this decision follows from what was said and whether
      // it was right, so it is exact per square rather than one answer per part.
      const shouldLook = call ? (call.correct ? call.said_look : !call.said_look) : null;
      out += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" rx="3.5" fill="${colour}"
        opacity="${call ? 0.85 : 0.35}"
        data-tip="<b>${armName}</b> <span class='k'>part ${col.trial}, look ${col.k + 1}</span><br>
          ${call ? (call.said_look ? "asked for another look" : "accepted the measurement")
                 : "no record"}${call
            ? `<br><span class='k'>correct call was</span> <b>${shouldLook ? "look again" : "accept"}</b>`
            : ""}"/>`;
    });
    out += `<text x="${padL + columns.length * (cell + gap) + 8}" y="${y + cell / 2 + 3.5}"
      font-size="10" fill="var(--text-secondary)" font-family="var(--mono)">${a.correct}/${a.n}</text>`;
  });
  out += V.axisTitle(padL, h - 8,
    `${columns.length} decisions on ${cases.length} parts, in the order they were taken`);
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
    label: t, value: s.n, colour: V.ARM_COLOUR[t] || "var(--band)",
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
    `Every row carries the model that answered and why it was picked, so a decision can be
     traced back to the model size that made it.`;

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
