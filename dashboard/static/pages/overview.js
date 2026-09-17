/* The headline page: what the cell claims, and the evidence for it, above the fold. */

import * as V from "../viz.js";
import { state, loadEvidence } from "../store.js";

export const meta = {
  id: "overview",
  title: "Findings",
  crumb: "Summary",
  group: "Result",
  tag: "50",
  icon: "M3 13h4l3-8 4 16 3-8h4",
};

export function render() {
  return `
    <p class="lede">
      A robot welding a folded tray runs a path that was taught once. The press does
      not hand it the same part twice: springback, camber and fit-up move the corner
      the seam lives on. These are the results of running <b>fifty parts</b> through a
      cell that looks at each one first and welds where the joint actually is.
    </p>
    <div class="tiles" id="ov-tiles"></div>
    <div class="grid">
      <div class="card w7">
        <header><h2>What a taught path would have missed, against what the cell found</h2></header>
        <div class="hint">One row per part. The left dot is the taught path's miss, the right dot is the
          tracked seam's residual error. Log scale, so both ends stay readable.</div>
        <div id="ov-dumbbell"></div>
        <div class="legend">
          <span><i style="background:var(--before)"></i>taught path, no correction</span>
          <span><i style="background:var(--after)"></i>tracked seam, mean of 20 stations</span>
        </div>
      </div>

      <div class="card w5">
        <header><h2>Residual error across all fifty parts</h2></header>
        <div class="hint">Every part's mean station error. The wire is 1.0 mm across, so
          anything left of the dashed line is inside a wire diameter.</div>
        <div id="ov-hist"></div>
        <footer id="ov-hist-meta"></footer>
      </div>

      <div class="card w12">
        <header><h2>Every claim on this dashboard, and what it was scored against</h2></header>
        <div class="hint">Nothing here is a demo reading of itself. Each study has a ground truth
          that was decided before the model was asked.</div>
        <div id="ov-studies"></div>
      </div>
    </div>`;
}

export async function mount() {
  const ev = await loadEvidence();
  tiles(ev);
  dumbbell(ev);
  hist(ev);
  studies(ev);
}

function tiles(ev) {
  const s = ev.seam, l = ev.locating, r = ev.router, i = ev.inspection;
  const ratio = s.available ? s.taught.mean_mm / s.tracked.mean_mm : null;
  const parts = s.available ? s.parts : [];
  const html = [];

  html.push(tile({
    label: "Seam error after correction",
    value: V.fmt(s.tracked.mean_mm, 3), unit: "mm",
    tone: "after",
    delta: ratio ? `${ratio.toFixed(0)}× better` : null,
    sub: `mean over ${s.stations_measured.toLocaleString()} measured stations, ${s.n_parts} parts`,
    spark: V.spark(parts.map((p) => p.mean_mm), "var(--after)"),
  }));
  html.push(tile({
    label: "A taught path would miss by",
    value: V.fmt(s.taught.mean_mm, 1), unit: "mm",
    tone: "before",
    sub: `worst part ${V.fmt(s.taught.max_mm, 1)} mm`,
    spark: V.spark(parts.map((p) => p.taught_miss_mm), "var(--before)"),
  }));
  html.push(tile({
    label: "Parts under 1 mm, every station",
    value: `${s.tracked.sub_mm_parts}/${s.n_parts}`,
    tone: s.tracked.sub_mm_parts / s.n_parts > 0.8 ? "good" : "warn",
    sub: `worst single station ${V.fmt(s.tracked.worst_station_mm, 2)} mm`,
    spark: V.spark(parts.map((p) => p.max_mm), "var(--series-1)"),
  }));
  html.push(tile({
    label: "Look-again calls scored right",
    value: r.available ? `${r.arms[0].correct}/${r.arms[0].n}` : "–",
    tone: "good",
    sub: r.available
      ? `${r.arms[0].looks_per_decision.toFixed(2)} looks per part, ${r.arms[0].total_latency_s.toFixed(0)} s total`
      : "",
  }));
  html.push(tile({
    label: "Post-weld verdicts",
    value: V.pct(i.binary_accuracy),
    tone: "warn",
    sub: `pass or defect on ${i.n} beads, ${i.false_alarms} false alarms, ${i.missed} missed`,
  }));
  html.push(tile({
    label: "Joint locating, blind to guided",
    value: l.available ? V.fmt(l.routed.mean_mm, 2) : "–", unit: "mm",
    tone: "after",
    delta: l.available ? `from ${V.fmt(l.blind.mean_mm, 1)} mm` : null,
    sub: l.available ? `${l.routed.sub_mm}/${l.n_parts} parts under 1 mm on a single look` : "",
    spark: l.available ? V.spark(l.parts.map((p) => p.routed_mm), "var(--series-5)") : "",
  }));

  V.$("ov-tiles").innerHTML = html.join("");
}

function tile({ label, value, unit = "", sub = "", tone = "", delta = null, spark = "", cls = "" }) {
  return `<div class="tile ${cls}">
    <div class="label">${V.esc(label)}</div>
    <div class="value ${tone}">${value}${unit ? `<span class="unit">${unit}</span>` : ""}
      ${delta ? `<span class="delta up">${V.esc(delta)}</span>` : ""}</div>
    <div class="sub">${V.esc(sub)}</div>${spark}</div>`;
}

function dumbbell(ev) {
  if (!ev.seam.available) return;
  const rows = [...ev.seam.parts]
    .sort((a, b) => b.taught_miss_mm - a.taught_miss_mm)
    .map((p) => ({
      a: p.taught_miss_mm,
      b: p.mean_mm,
      tip: `<b>part ${p.trial}</b><br>
        <span class="k">taught path misses</span> <b>${V.fmt(p.taught_miss_mm, 2)} mm</b><br>
        <span class="k">tracked to</span> <b>${V.fmt(p.mean_mm, 3)} mm</b>
        <span class="k">(worst station ${V.fmt(p.max_mm, 2)} mm)</span><br>
        <span class="k">seam bows</span> <b>${V.fmt(p.bow_mm, 2)} mm</b> <span class="k">off its own straight line</span>`,
    }));
  V.paint(V.$("ov-dumbbell"), V.dumbbell(rows, { width: 700, rowH: 11 }));
}

function hist(ev) {
  if (!ev.seam.available) return;
  const values = ev.seam.parts.map((p) => p.mean_mm);
  V.paint(V.$("ov-hist"), V.histogram(values, {
    width: 420, height: 210, bins: 14, max: 1.2,
    colour: "var(--series-1)", unit: "mm", noun: "parts",
    mark: 1.0, markLabel: "wire ø", markColour: "var(--warning)",
    xTitle: "mean station error, mm",
  }));
  const s = ev.seam;
  V.$("ov-hist-meta").innerHTML =
    `Median <b>${V.fmt(s.tracked.median_mm, 3)} mm</b>, worst part <b>${V.fmt(s.tracked.max_mm, 3)} mm</b>.
     The seam itself is not straight: it bows <b>${V.fmt(s.bow.mean_mm, 2)} mm</b> on average and
     <b>${V.fmt(s.bow.max_mm, 2)} mm</b> at worst, which is the part of the error a straight
     corrected line could not remove either.`;
}

function studies(ev) {
  const rows = [
    ["Seam tracking", `${ev.seam.n_parts} parts × ${ev.seam.stations_per_part} stations`,
     "the seam's true position, known from the part it was folded into",
     `${V.fmt(ev.seam.taught.mean_mm, 1)} mm → ${V.fmt(ev.seam.tracked.mean_mm, 3)} mm`, "good", "#/seam"],
    ["Joint locating", `${ev.locating.n_parts} parts, blind / one look / routed`,
     "the corner's true coordinates in the cell",
     `${V.fmt(ev.locating.blind.mean_mm, 1)} mm → ${V.fmt(ev.locating.routed.mean_mm, 2)} mm`, "good", "#/seam"],
    ["Look-again routing", ev.router.available ? `${ev.router.arms[0].n} decisions × 5 arms` : "–",
     "whether a second look actually improved the fit",
     ev.router.available ? `routed ${ev.router.arms[0].correct}/${ev.router.arms[0].n}, Super ${ev.router.arms.find((a) => a.arm === "super").correct}/${ev.router.arms[0].n}` : "–",
     "good", "#/routing"],
    ["Drift diagnosis", ev.drift.available ? `${ev.drift.runs[0].n} runs × ${ev.drift.runs.length} models` : "–",
     "the fault that was injected into the run",
     ev.drift.available ? ev.drift.runs.map((r) => `${r.tier} ${r.correct}/${r.n}`).join(", ") : "–",
     "warn", "#/diagnosis"],
    ["Post-weld inspection", `${ev.inspection.n} beads`,
     "the defect that was injected on purpose",
     `${V.pct(ev.inspection.binary_accuracy)} pass/defect, ${V.pct(ev.inspection.named_accuracy)} named`,
     "warn", "#/inspection"],
  ];
  V.$("ov-studies").innerHTML = `<table>
    <thead><tr><th>Study</th><th>Scale</th><th>Scored against</th><th>Result</th><th></th></tr></thead>
    <tbody>${rows.map(([a, b, c, d, tone, href]) => `<tr>
      <td><b>${V.esc(a)}</b></td>
      <td class="muted">${V.esc(b)}</td>
      <td class="muted">${V.esc(c)}</td>
      <td class="mono"><span class="badge ${tone}">${V.esc(d)}</span></td>
      <td style="text-align:right"><a href="${href}" class="badge">open →</a></td>
    </tr>`).join("")}</tbody></table>`;
}
