/* The drift study: given a run of parts, name the machine that is going out. */

import * as V from "../viz.js";
import { loadEvidence } from "../store.js";

export const meta = {
  id: "diagnosis",
  title: "Drift diagnosis",
  crumb: "Study 3",
  group: "Studies",
  tag: "12",
  icon: "M3 18l5-6 4 3 5-8 4 5",
};

const SCENARIO_LABEL = {
  press: "press brake angle walking",
  fixture: "fixture creeping sideways",
  none: "nothing wrong, just scatter",
};

export function render() {
  return `
    <p class="lede">
      Correcting each part hides the problem that made it. If the press brake is walking,
      the cell will weld every part correctly and nobody will notice until the angle runs
      out of correction. So the same measurements are read a second time, across a run,
      to name the cause. This is the one question where the small model is <b>not</b> good
      enough, and the study is here because it says so.
    </p>

    <div class="tiles" id="dg-tiles"></div>

    <div class="grid">
      <div class="card w7">
        <header><h2>Who named the cause, and who guessed one answer for everything</h2></header>
        <div class="hint">Four seeds, three scenarios each. A model that always says the same thing
          scores four out of twelve by arithmetic, not by diagnosis, which is why the shape of the
          answers matters as much as the count.</div>
        <div id="dg-grid"></div>
        <div class="legend">
          <span><i style="background:var(--good)"></i>named it correctly</span>
          <span><i style="background:var(--critical)"></i>named the wrong cause</span>
        </div>
      </div>

      <div class="card w5">
        <header><h2>What each model actually said</h2></header>
        <div class="hint">The answer distribution, not the score. A single colour across the whole
          row means the model was not reading the evidence.</div>
        <div id="dg-spread"></div>
        <div class="legend">
          <span><i style="background:var(--series-2)"></i>said press</span>
          <span><i style="background:var(--series-3)"></i>said fixture</span>
          <span><i style="background:var(--series-1)"></i>said nothing is wrong</span>
        </div>
        <footer id="dg-spread-meta"></footer>
      </div>

      <div class="card w12">
        <header><h2>The same models at 2.5&times; severity</h2></header>
        <div class="hint">If a model is reading the evidence, a stronger signal should help it. If it
          is guessing, a stronger signal changes nothing.</div>
        <div id="dg-severity"></div>
        <footer id="dg-severity-meta"></footer>
      </div>

      <div class="card w12">
        <header>
          <h2>What the model pointed at</h2>
          <span class="grow" style="flex:1"></span>
          <select id="dg-model"></select>
        </header>
        <div id="dg-cases"></div>
      </div>
    </div>`;
}

let ev = null;

export async function mount() {
  ev = await loadEvidence();
  if (!ev.drift.available) return;
  tiles();
  grid();
  spread();
  severity();
  picker();
}

function tiles() {
  const d = ev.drift;
  const best = [...d.runs].sort((a, b) => b.accuracy - a.accuracy)[0];
  const nano = d.runs.find((r) => r.tier === "nano");
  const sevSuper = d.severe.find((r) => r.tier === "super");
  const sevNano = d.severe.find((r) => r.tier === "nano");
  V.$("dg-tiles").innerHTML = [
    tile("Best diagnosis", `${best.correct}/${best.n}`, "good",
      `${best.model} reading the same twelve runs`),
    tile("Smallest model", `${nano.correct}/${nano.n}`, "bad",
      nano.degenerate ? `answered "${nano.always_said}" every time` : "mixed answers"),
    sevSuper ? tile("At 2.5× severity", `${sevSuper.correct}/${sevSuper.n}`, "good",
      "the larger model improves when the signal does") : "",
    sevNano ? tile("Smallest at 2.5×", `${sevNano.correct}/${sevNano.n}`, "bad",
      "unchanged, which is what guessing looks like") : "",
    tile("Median time to answer", `${V.fmt(best.median_latency_ms / 1000, 1)}`, "", "seconds, once per shift", "s"),
  ].filter(Boolean).join("");
}

const tile = (label, value, tone, sub, unit = "") => `<div class="tile">
  <div class="label">${V.esc(label)}</div>
  <div class="value ${tone}">${value}${unit ? `<span class="unit">${unit}</span>` : ""}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

function caseKey(c) { return `${c.scenario}|${c.seed}`; }

function grid() {
  const d = ev.drift;
  const keys = [];
  d.scenarios.forEach((s) => d.seeds.forEach((seed) => keys.push(`${s}|${seed}`)));

  const cell = 40, gap = 3, padL = 108, padT = 58;
  const w = padL + keys.length * (cell + gap) + 52;
  const h = padT + d.runs.length * (cell + gap) + 30;
  let out = "";

  keys.forEach((k, j) => {
    const [scenario, seed] = k.split("|");
    const x = padL + j * (cell + gap) + cell / 2;
    out += `<text class="axis-label" x="${x}" y="${padT - 22}" text-anchor="middle" font-size="9.5">${V.esc(scenario)}</text>`;
    out += `<text class="axis-label" x="${x}" y="${padT - 10}" text-anchor="middle" font-size="8.5" opacity=".6">seed ${seed}</text>`;
  });

  d.runs.forEach((run, i) => {
    const y = padT + i * (cell + gap);
    out += `<text class="axis-label" x="${padL - 10}" y="${y + cell / 2 + 3.5}" text-anchor="end"
      font-size="10" fill="${V.ARM_COLOUR[run.tier]}">${V.esc(run.tier)}</text>`;
    const byKey = new Map(run.cases.map((c) => [caseKey(c), c]));
    keys.forEach((k, j) => {
      const c = byKey.get(k);
      const x = padL + j * (cell + gap);
      const colour = !c ? "var(--surface-2)" : c.correct ? "var(--good)" : "var(--critical)";
      out += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" rx="8" fill="${colour}"
        opacity="${c ? 0.8 : 0.3}" stroke="var(--bg)" stroke-width="2"
        data-tip="<b>${V.esc(run.model)}</b><br>
          <span class='k'>the run was</span> <b>${V.esc(SCENARIO_LABEL[c?.scenario] || c?.scenario || "")}</b><br>
          <span class='k'>it said</span> <b>${V.esc(c?.said || "–")}</b>
          <span class='k'>at</span> <b>${c?.confidence != null ? V.pct(c.confidence, 0) : "–"}</b> <span class='k'>confidence</span><br>
          <span class='k'>tilt slope</span> <b>${V.fmt(c?.tilt_slope, 3)}</b> <span class='k'>mm/part</span>"/>`;
      if (c) out += `<text x="${x + cell / 2}" y="${y + cell / 2 + 4}" text-anchor="middle"
        font-size="9.5" fill="#ffffffcc" font-family="var(--mono)">${V.esc(c.said.slice(0, 4))}</text>`;
    });
    out += `<text x="${padL + keys.length * (cell + gap) + 8}" y="${y + cell / 2 + 4}"
      font-size="11" fill="var(--text-secondary)" font-family="var(--mono)">${run.correct}/${run.n}</text>`;
  });
  out += V.axisTitle(padL, h - 8, "each column is one simulated run of thirty parts");
  V.paint(V.$("dg-grid"), V.svg(w, h, out));
}

function spread() {
  const d = ev.drift;
  const answers = ["press", "fixture", "none"];
  const colours = { press: "var(--series-2)", fixture: "var(--series-3)", none: "var(--series-1)" };
  const rows = d.runs.map((run) => {
    const counts = answers.map((a) => run.cases.filter((c) => c.said === a).length);
    return { run, counts };
  });
  const w = 380, rowH = 46, padL = 84;
  const h = rows.length * rowH + 34;
  let out = "";
  rows.forEach(({ run, counts }, i) => {
    const y = i * rowH + 10;
    out += `<text class="axis-label" x="${padL - 10}" y="${y + 14}" text-anchor="end" font-size="10"
      fill="${V.ARM_COLOUR[run.tier]}">${V.esc(run.tier)}</text>`;
    let x = padL;
    const bw = (w - padL - 8) / run.n;
    counts.forEach((n, k) => {
      if (n > 0) {
        out += `<rect x="${x + 1}" y="${y}" width="${Math.max(0, n * bw - 2)}" height="22" rx="5"
          fill="${colours[answers[k]]}" opacity=".88"
          data-tip="<b>${V.esc(run.model)}</b><br>said <b>${V.esc(answers[k])}</b> on <b>${n}</b> of ${run.n} runs"/>`;
        if (n * bw > 44) out += `<text x="${x + n * bw / 2}" y="${y + 15}" text-anchor="middle"
          font-size="10" fill="#ffffffdd" font-family="var(--mono)">${V.esc(answers[k])}</text>`;
      }
      x += n * bw;
    });
    out += `<text x="${padL}" y="${y + 36}" font-size="9.5" fill="var(--text-muted)"
      font-family="var(--sans)">${run.degenerate ? "one answer for every run" : "reads each run separately"}</text>`;
  });
  V.paint(V.$("dg-spread"), V.svg(w, h, out));

  const degenerate = d.runs.filter((r) => r.degenerate).map((r) => r.tier);
  V.$("dg-spread-meta").innerHTML =
    `${degenerate.length ? `<b>${degenerate.join("</b> and <b>")}</b> gave the same answer to every run, ` : ""}
     which scores <b>4/12</b> whether or not anything was read. The score alone would have made them look
     like partial successes; the answer distribution is what shows they were not diagnosing.
     This is why the shift-level question is routed up rather than kept on the small model.`;
}

function severity() {
  const d = ev.drift;
  if (!d.severe.length) { V.$("dg-severity").innerHTML = `<div class="empty">Not run.</div>`; return; }
  const rows = [];
  d.severe.forEach((sev) => {
    const base = d.runs.find((r) => r.tier === sev.tier);
    rows.push({
      tier: sev.tier, model: sev.model,
      before: base ? base.correct : 0, after: sev.correct, n: sev.n,
    });
  });
  const w = 640, rowH = 56, padL = 150, padR = 130;
  const h = rows.length * rowH + 40;
  const xOf = (v, n) => padL + (v / n) * (w - padL - padR);
  let out = V.axisTitle(padL, 16, "runs named correctly, out of twelve");
  for (let k = 0; k <= 12; k += 3) {
    const x = padL + (k / 12) * (w - padL - padR);
    out += `<line class="gridline" x1="${x}" y1="24" x2="${x}" y2="${h - 26}"/>
      <text class="axis-label" x="${x}" y="${h - 12}" text-anchor="middle">${k}</text>`;
  }
  rows.forEach((r, i) => {
    const y = 40 + i * rowH;
    out += `<text x="${padL - 12}" y="${y + 5}" text-anchor="end" font-size="11.5"
      fill="var(--text-secondary)" font-family="var(--sans)">${V.esc(r.model)}</text>`;
    const xa = xOf(r.before, 12), xb = xOf(r.after, 12);
    out += `<line x1="${xa}" y1="${y}" x2="${xb}" y2="${y}" stroke="#2a3646" stroke-width="2"/>`;
    out += V.dot(xa, y, 5.5, V.BEFORE, `data-tip="<b>${V.esc(r.model)}</b> at normal severity: <b>${r.before}/12</b>"`);
    out += V.dot(xb, y, 5.5, r.after > r.before ? V.AFTER : V.BAD,
      `data-tip="<b>${V.esc(r.model)}</b> at 2.5× severity: <b>${r.after}/12</b>"`);
    out += `<text x="${w - padR + 14}" y="${y + 4}" font-size="11.5" font-family="var(--mono)"
      fill="${r.after > r.before ? "var(--after)" : "var(--text-muted)"}">${r.before} → ${r.after}</text>`;
  });
  V.paint(V.$("dg-severity"), V.svg(w, h, out));

  const improved = rows.filter((r) => r.after > r.before).map((r) => r.tier);
  const flat = rows.filter((r) => r.after <= r.before).map((r) => r.tier);
  V.$("dg-severity-meta").innerHTML =
    `${improved.length ? `<b>${improved.join(", ")}</b> improved when the fault got stronger, which is what
      reading the evidence looks like. ` : ""}
     ${flat.length ? `<b>${flat.join(", ")}</b> did not move, which is what a fixed answer looks like. ` : ""}
     The practical reading: this question belongs on a larger model, it is asked once a shift rather than
     once a part, and the cost of putting it there is a few seconds a shift.`;
}

function picker() {
  const sel = V.$("dg-model");
  sel.innerHTML = ev.drift.runs.map((r) =>
    `<option value="${r.tier}">${V.esc(r.model)}</option>`).join("");
  const best = [...ev.drift.runs].sort((a, b) => b.accuracy - a.accuracy)[0];
  sel.value = best.tier;
  sel.onchange = () => cases(sel.value);
  cases(best.tier);
}

function cases(tier) {
  const run = ev.drift.runs.find((r) => r.tier === tier);
  V.$("dg-cases").innerHTML = `<table>
    <thead><tr>
      <th>The run was</th><th>Seed</th><th>It said</th><th class="num">Confidence</th>
      <th class="num">Tilt mm/part</th><th class="num">Shift mm/part</th><th>What it pointed at</th>
    </tr></thead>
    <tbody>${run.cases.map((c) => `<tr>
      <td>${V.esc(SCENARIO_LABEL[c.scenario] || c.scenario)}</td>
      <td class="mono muted">${c.seed}</td>
      <td><span class="badge ${c.correct ? "good" : "bad"}">${V.esc(c.said)}</span></td>
      <td class="num">${c.confidence != null ? V.pct(c.confidence, 0) : "–"}</td>
      <td class="num">${V.fmt(c.tilt_slope, 3)}</td>
      <td class="num">${V.fmt(c.shift_slope, 3)}</td>
      <td class="quote" style="border:none;padding-left:0;max-width:440px">${V.esc(c.evidence || "")}</td>
    </tr>`).join("")}</tbody></table>`;
}
