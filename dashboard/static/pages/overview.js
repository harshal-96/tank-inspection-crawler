/* The headline page: what the cell claims, and the evidence for it, above the fold. */

import * as V from "../viz.js";
import { state, loadEvidence, refreshLive } from "../store.js";

export const meta = {
  id: "overview",
  title: "Findings",
  crumb: "Summary",
  group: "Result",
  tag: "50",
  icon: "M3 13h4l3-8 4 16 3-8h4",
};

/** Proof the thing runs, not just that it once produced a table.

  A summary page can be written without a working system behind it, and a reader
  has no way to tell from the numbers alone. This band reads the live database, so
  it is empty until something has actually been through the cell.
*/
function liveBand() {
  const host = V.$("ov-live");
  if (!host) return;
  const live = state.live;
  if (!live || !state.connected || !live.parts.length) {
    host.innerHTML = `<div class="hint" style="margin:0 0 18px">
      Nothing has been through the cell yet. <code>python scripts/demo.py</code>
      runs six parts through it in about three seconds.</div>`;
    return;
  }
  const latest = live.parts[0];
  const slot = live.byPart.get(latest.id) || {};
  const seam = slot.seam;
  const bits = [
    ["parts through the cell", live.parts.length],
    ["last part", V.esc(latest.serial)],
  ];
  if (seam && seam.error_mm != null) {
    bits.push(["its seam, measured", `${V.fmt(seam.error_mm, 2)} mm over `
      + `${seam.stations_measured} stations`]);
  }
  if (slot.inspection) {
    bits.push(["post-weld", V.esc(slot.inspection.verdict)]);
  }
  host.innerHTML = `<div class="card w12" style="margin-bottom:18px">
    <header><h2>Running now</h2>
      <span style="flex:1"></span>
      <a class="hint" href="#/cell" style="text-decoration:none">open the cell &rarr;</a>
    </header>
    <div class="hint">Read from the live database rather than from the study files, so
      this is the cell as it stands and not a result recorded earlier.</div>
    <div class="kv">${bits.map(([k, v]) =>
      `<span class="k">${k}</span> <b>${v}</b>`).join("<span class='sep'></span>")}</div>
  </div>`;
}

export function render() {
  return `
    <p class="lede">
      A welding robot follows the same path every time. The parts do not arrive the
      same every time. Sheet metal is folded in a press, and every fold springs back by a
      slightly different amount, so the joint that needs welding sits in a different place
      on every part. This cell photographs each part first, works out where its joint
      actually is, and welds there. Below are the results from <b>fifty parts</b>.
    </p>
    <div id="ov-live"></div>
    <div class="tiles" id="ov-tiles"></div>

    <div class="grid">
      <div class="card w12">
        <header><h2>What happens to a part, in order</h2></header>
        <div class="hint">Six stages. Each one links to the page that measured it, with the
          number that page reports. This is the same order as the rail on the left.</div>
        <div id="ov-pipeline"></div>
      </div>
    </div>

    <div class="grid">
      <div class="card w12" id="bg-honest">
        <header><h2>What these numbers leave out</h2></header>
        <div class="hint">Two things worth knowing before you read any of the numbers below.</div>
        <div id="ov-caveats"></div>
      </div>
    </div>

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
        <div class="hint">Every study is scored against a correct answer that was fixed before the
          model was asked. None of this is the cell marking its own work.</div>
        <div id="ov-studies"></div>
      </div>
    </div>`;
}

export function update() { liveBand(); }

export async function mount() {
  const [ev] = await Promise.all([loadEvidence(), refreshLive()]);
  liveBand();
  pipeline(ev);
  tiles(ev);
  dumbbell(ev);
  hist(ev);
  studies(ev);
  caveats(ev);
}

/** The six stages a part goes through, each with the one number that page reports.

  The studies were a flat list of seven, numbered in the order they were built
  rather than the order they happen in, so nothing on this dashboard said what
  the cell actually does from one end to the other.
*/
function pipeline(ev) {
  const host = V.$("ov-pipeline");
  if (!host) return;
  const s = ev.seam, r = ev.router, i = ev.inspection, c = ev.compensation, h = ev.hard;
  const stages = [
    ["seam", "Find the joint",
     "A depth camera fits a plane to each folded face and intersects them, at twenty points along the seam.",
     s.available ? `${V.fmt(s.tracked.mean_mm, 3)} mm` : null,
     s.available ? `from ${V.fmt(s.taught.mean_mm, 1)} mm for a taught path` : ""],
    ["routing", "Decide whether to trust it",
     "The measurement carries signals about its own quality. A model reads them and either accepts it or asks for another look.",
     r.available ? `${r.arms[0].correct}/${r.arms[0].n}` : null,
     "calls scored against what a second look actually achieved"],
    ["compensation", "Aim ahead of the heat",
     "The arc rotates both folded sides as it goes, so the path is shifted by the distortion expected when the torch arrives.",
     c && c.available ? "0.55 → 0.03 mm" : null,
     "residual at nominal, over 120 trials"],
    ["inspection", "Judge the finished bead",
     "The weld is photographed and scored against the defect that was injected before the picture was taken.",
     i.available ? V.pct(i.binary_accuracy) : null,
     i.available ? `pass or defect on ${i.n} beads` : ""],
    ["diagnosis", "Name the machine that is drifting",
     "The same measurements are read again across a whole run, to say which machine upstream needs touching.",
     h && h.available ? V.pct(h.arms[0].rate) : null,
     h && h.available ? `best of ${h.arms.length} arms on the boundary test` : ""],
    ["briefs", "Say what to do about it",
     "The question is written out of the cell's own numbers, answered from published welding guidance, and cited.",
     null, "every statement carries its source"],
  ];
  host.innerHTML = `<ol class="pipeline">${stages.map(([id, title, body, value, note], n) => `
    <li>
      <a href="#/${id}">
        <span class="n">${n + 1}</span>
        <span class="text">
          <span class="t">${V.esc(title)}</span>
          <span class="d">${V.esc(body)}</span>
        </span>
        <span class="fig">
          ${value ? `<b>${value}</b>` : ""}
          <span class="k">${V.esc(note)}</span>
        </span>
      </a>
    </li>`).join("")}</ol>`;
}

/** The two things a reader should have before over-reading the headline. */
function caveats(ev) {
  const host = V.$("ov-caveats");
  if (!host) return;
  const cards = [];

  const b = ev.error_budget;
  if (b && b.available) {
    const realistic = b.budgets.find((x) => x.label.startsWith("Typical stereo"));
    const best = b.budgets.find((x) => x.label.startsWith("Laser"));
    cards.push([
      "The seam figure is one link in a chain",
      `The tracker finds the joint to ${V.fmt(b.budgets[0].total_mm, 2)} mm. But that is `
      + `one link in a chain. Add hand-eye calibration, the robot's own repeatability, `
      + `standoff drift and what the heat of the arc does, and the whole chain comes to `
      + `${V.fmt(realistic.total_mm, 2)} mm on a realistic sensor, against a requirement of `
      + `${b.industry_requirement_mm} mm. Swap in the laser profiler that the sensor sweep `
      + `says this needs, and it falls to ${V.fmt(best.total_mm, 2)} mm. The biggest single `
      + `term is calibration. It is the same error on every part, so averaging more parts `
      + `does not remove it, and this cell has no way to measure it.`,
      "#/budget",
    ]);
  }

  const c = ev.confound;
  if (c && c.available) {
    cards.push([
      "A drifting camera looks exactly like a drifting fixture",
      `A fixture slipping moves the part. A camera slipping moves the measurement. Both `
      + `shift the whole joint by the same amount top and bottom, so on the seam alone they `
      + `produce identical numbers. The cell answers "fixture" to both, and sends a fitter `
      + `to spend a shift on a machine that was fine. Bolting one reference marker to the `
      + `machine base fixes it, because only the camera can appear to move it: drifting `
      + `cameras identified went from ${c.camera_without.correct}/${c.camera_without.n} to `
      + `${c.camera_with.correct}/${c.camera_with.n}, same model, one extra column of data.`,
      "#/budget",
    ]);
  }

  if (!cards.length) {
    host.innerHTML = `<div class="empty">Not yet measured.</div>`;
    return;
  }
  host.innerHTML = `<div class="steps">${cards.map(([title, body, href]) => `
    <div class="step">
      <div class="t" style="margin-top:0">${V.esc(title)}</div>
      <div class="d">${V.esc(body)}</div>
      <div class="o"><a href="${href}" style="text-decoration:none">read the study \u2192</a></div>
    </div>`).join("")}</div>`;
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
  // This tile used to lead with the edge model's 12 of 12 on the three-cause set.
  // That set has no hard case in it and four lines of arithmetic score the same,
  // so the number was real and meant nothing. The boundary test is what the
  // summary should carry, including the part where nothing passes it.
  const hard = ev.hard;
  if (hard && hard.available && hard.arms.length) {
    const best = hard.arms[0];
    const rule = hard.arms.find((a) => a.label.includes("threshold as arithmetic"));
    const sig = hard.arms.filter((a) => a.vs_rule && a.vs_rule.p < 0.05).length;
    html.push(tile({
      label: "Best score on the boundary test",
      value: V.pct(best.rate),
      tone: "warn",
      delta: rule ? `vs ${V.pct(rule.rate)} for a fixed rule` : null,
      sub: `${best.label}, on ${hard.cases} cases built to sit on the decision `
         + `boundary. Only ${sig} arms beat the rule by a margin that survives a `
         + `paired test. Nothing is close to answering it.`,
    }));
  }
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
  if (ev.error_budget && ev.error_budget.available) {
    const realistic = ev.error_budget.budgets.find((b) => b.label.startsWith("Typical stereo"));
    rows.push(["Full error budget", "5 terms, 2 measured here",
      "published figures for calibration, repeatability and standoff",
      `${V.fmt(ev.error_budget.budgets[0].total_mm, 2)} mm quoted \u2192 `
      + `${V.fmt(realistic.total_mm, 2)} mm for the chain`,
      "bad", "#/budget"]);
  }
  if (ev.confound && ev.confound.available) {
    rows.push(["Camera against fixture", "4 causes \u00d7 4 runs \u00d7 2 arms",
      "the injected fault, with and without a reference feature",
      `${ev.confound.arms[0].correct}/${ev.confound.arms[0].n} \u2192 `
      + `${ev.confound.arms[1].correct}/${ev.confound.arms[1].n} with the instrument`,
      "good", "#/budget"]);
  }
  const trainedRun = (ev.drift.edge || []).find((r) => r.tier === "edge");
  const untrainedRun = (ev.drift.edge || []).find((r) => r.tier === "edge_base");
  if (trainedRun) {
    rows.push(["Trained diagnosis, at the edge",
      `600 simulated runs, ${trainedRun.n} held out`,
      "the same injected faults, on seeds the training set excluded",
      untrainedRun
        ? `${untrainedRun.correct}/${untrainedRun.n} untrained → ${trainedRun.correct}/${trainedRun.n} trained`
        : `${trainedRun.correct}/${trainedRun.n}`,
      // Not "good". The number is real and the test it was measured on is not
      // hard, which the row below is here to say.
      "warn", "#/diagnosis"]);
  }
  if (ev.hard && ev.hard.available) {
    const best = ev.hard.arms[0];
    const rule = ev.hard.arms.find((a) => a.label.includes("threshold as arithmetic"));
    const edge = ev.hard.arms.find((a) => a.label.includes("on the cell"));
    rows.push(["The same diagnosis, on hard cases",
      `${ev.hard.arms.length} arms × ${ev.hard.cases} cases, passes pooled`,
      "cases sampled onto the decision boundary, where a fixed rule fails",
      `best ${V.pct(best.rate)}, a fixed rule ${rule ? V.pct(rule.rate) : "–"}`
      + (edge ? `, the edge model ${V.pct(edge.rate)}` : "")
      + (best.vs_rule ? ` (p ${best.vs_rule.p.toFixed(3)})` : ""),
      "bad", "#/diagnosis"]);
  }
  if (ev.vision && ev.vision.available && ev.vision.arms.length > 1) {
    const quiet = ev.vision.arms.find((a) => a.never_says_defect);
    rows.push(["Two vision models", `${ev.vision.arms.length} models × ${ev.vision.arms[0].n} beads`,
      "the injected defect, scored the same way for both",
      quiet ? `equal scores, ${quiet.missed} misses from the one that never says defect`
            : "compared on the same beads",
      "warn", "#/inspection"]);
  }
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
