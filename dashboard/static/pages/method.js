/* How the cell works, and what it would take to point it at a real one. */

import * as V from "../viz.js";
import { loadEvidence } from "../store.js";

export const meta = {
  id: "method",
  title: "Method",
  crumb: "Reference",
  group: "Reference",
  icon: "M6 4h9l3 3v13H6zM9 10h6M9 14h6",
};

const STAGES = [
  { n: 1, name: "Look", what: "One fixed camera returns colour, depth, normals and a mask of the corner.",
    why: "A camera on the wrist has to be recalibrated against every pose, and the torch gets in front of it.",
    out: "one frame" },
  { n: 2, name: "Fit", what: "The two folded faces are fitted as planes and intersected; the line they share is the joint.",
    why: "Geometry, not a learned keypoint. It gives a line with a residual you can read, so failure is visible rather than silent.",
    out: "a joint line and its fit quality" },
  { n: 3, name: "Judge", what: "A small model reads the fit signals and says whether this is good enough to weld on.",
    why: "The question is cheap and asked once a part, so it goes to the smallest model that can answer it.",
    out: "accept, or take another look" },
  { n: 4, name: "Track", what: "Twenty stations are cut along the seam and each is located on its own.",
    why: "The seam is not straight. A single corrected point still leaves the bow.",
    out: "a path, not an offset" },
  { n: 5, name: "Schedule", what: "Each station gets its own current, voltage and travel speed from the gap measured there.",
    why: "A gap needs less current and slower travel, not more heat. Past 1.6 times the sheet, one pass cannot bridge it.",
    out: "per-station weld parameters, or a refusal" },
  { n: 6, name: "Predict", what: "Shrinkage and angular distortion from the heat about to go in are carried forward.",
    why: "The arc moves the joint ahead of itself. Measuring before the weld and ignoring that is measuring the wrong part.",
    out: "expected drift along the seam" },
  { n: 7, name: "Inspect", what: "The finished bead is photographed and judged against the defect that was injected.",
    why: "A verdict with nothing to score it against is a demo reading of itself.",
    out: "pass or a named defect" },
  { n: 8, name: "Diagnose", what: "Once a shift, the run is read as a whole to name what is going out of adjustment.",
    why: "Correcting every part hides the machine that made it wrong. This is the question the small model fails.",
    out: "press, fixture, or nothing" },
];

export function render() {
  return `
    <p class="lede">
      The robot is accurate. The problem is that the part is not where the program
      assumes it is. Sheet metal is cut flat, folded, and springs back by a slightly
      different amount every time. A fixed taught path welds thin air on one part and into
      the corner on the next. Everything here exists to close that gap, without touching
      the robot's own accuracy.
    </p>

    <div class="grid">
      <div class="card w12">
        <header><h2>Eight stages, and what each one is for</h2></header>
        <div id="me-stages"></div>
      </div>

      <div class="card w6">
        <header><h2>Ground truth, per study</h2></header>
        <div class="hint">Every number on this dashboard means something only because the correct answer
          was fixed before the model was asked.</div>
        <div id="me-truth"></div>
      </div>

      <div class="card w6">
        <header><h2>What this does not show</h2></header>
        <div id="me-limits"></div>
      </div>

      <div class="card w12">
        <header><h2>Does this only work for welding?</h2></header>
        <div class="hint">Worth answering plainly. The honest answer is not a simple yes, and not the flat
          no a salesman would give. What generalises is the loop. What does not is every number in it.</div>
        <div id="me-general"></div>
        <footer id="me-general-meta"></footer>
      </div>

      <div class="card w12">
        <header><h2>One question per tier, chosen by how often it is asked</h2></header>
        <div class="hint">Not a cascade by difficulty. A cascade escalates when it is unsure, which
          means paying for the large model on exactly the cases where it is slowest. This picks the
          tier by what the question is worth, and the routing study is the test of whether that holds.</div>
        <div id="me-tiers"></div>
      </div>
    </div>`;
}

export async function mount() {
  const ev = await loadEvidence();
  stages();
  truth(ev);
  limits();
  generality();
  tiers(ev);
}

function stages() {
  V.$("me-stages").innerHTML = `<div class="steps">${STAGES.map((st) => `
    <div class="step">
      <div class="n">${String(st.n).padStart(2, "0")}</div>
      <div class="t">${V.esc(st.name)}</div>
      <div class="d">${V.esc(st.what)}</div>
      <div class="w">${V.esc(st.why)}</div>
      <div class="o">→ ${V.esc(st.out)}</div>
    </div>`).join("")}</div>`;
}

function truth(ev) {
  const rows = [
    ["Seam tracking", "the seam's position in the part that was folded, known exactly",
     `${ev.seam.n_parts} parts × ${ev.seam.stations_per_part} stations`],
    ["Joint locating", "the corner's coordinates in the cell", `${ev.locating.n_parts} parts, three policies`],
    ["Look-again routing", "whether a second look measurably improved the fit",
     ev.router.available ? `${ev.router.arms[0].n} decisions × 5 arms` : "–"],
    ["Drift diagnosis", "the fault injected into the run, with a control that was verified flat first",
     ev.drift.available ? `${ev.drift.runs[0].n} runs × ${ev.drift.runs.length} models` : "–"],
    ["Post-weld inspection", "the defect injected into the bead on purpose", `${ev.inspection.n} beads`],
  ];
  V.$("me-truth").innerHTML = `<table>
    <thead><tr><th>Study</th><th>Scored against</th><th>Scale</th></tr></thead>
    <tbody>${rows.map(([a, b, c]) => `<tr><td><b>${V.esc(a)}</b></td>
      <td class="muted">${V.esc(b)}</td><td class="mono muted">${V.esc(c)}</td></tr>`).join("")}</tbody></table>`;
}

function limits() {
  const rows = [
    ["It runs in simulation", "The depth, the normals and the mask come from a rendered cell, not a real sensor. Real depth is noisier at a bright arc and on a specular sheet, and that noise lands directly on the plane fits."],
    ["The arc is modelled, not struck", "Shrinkage and angular distortion come from a calibrated model of heat input, not from a thermocouple. The numbers are the right shape and the right order; they are not a measurement of this joint."],
    ["The defects are injected", "That is what makes the inspection score meaningful, and it also means the defect population is the one that was written, not the one a real line produces."],
    ["Fifty parts is a run, not a process", "It is enough to show the correction holds and enough to see drift. It is not enough to quote a capability index."],
  ];
  V.$("me-limits").innerHTML = rows.map(([a, b]) => `
    <div class="step" style="padding:13px 0 12px">
      <div class="t" style="margin-top:0">${V.esc(a)}</div>
      <div class="d">${V.esc(b)}</div>
    </div>`).join("");
}

const MOVES = [
  { n: 1, name: "Measure the part, not the program",
    here: "Twenty stations along the corner where two folded edges meet.",
    elsewhere: "Whatever the tool is about to touch, in the pose the part actually "
             + "arrived in, at the place it will be touched rather than at a datum "
             + "somewhere else on it." },
  { n: 2, name: "Schedule the process to what was measured",
    here: "Current, voltage and travel speed per station, from the gap measured there, "
        + "and a refusal where one pass cannot bridge it.",
    elsewhere: "The process parameter that the local condition decides, and the "
             + "willingness to stop rather than run the process outside where it works." },
  { n: 3, name: "Read the run to name the machine",
    here: "Which is drifting, the press brake or the fixture, from how the seam moves "
        + "at the bottom against the top.",
    elsewhere: "Correcting every part hides the machine that made them wrong. Something "
             + "has to look across parts rather than at one." },
];

const ELSEWHERE = [
  ["Press braking", "The same springback this cell corrects for is measurable on the part "
   + "the brake just made. Measure the achieved angle, correct the next bend. High-end "
   + "brakes already do this with angle sensors, which is the strongest evidence the loop "
   + "is the right shape."],
  ["Self-pierce riveting", "Hole position moves with the same forming variation, and the "
   + "die and force depend on the local stack thickness, which is a measurement rather than "
   + "a lookup. Stage two is nearly identical; the feature is a point rather than a line."],
  ["Adhesive and sealant dispensing", "The bead has to follow a flange that moved, and flow "
   + "rate should follow the gap it is filling. This is the closest relative: the same three "
   + "moves with the arc swapped for a nozzle."],
  ["Deburring and linishing", "Contact force set from the local material condition rather "
   + "than from a taught constant, and a refusal where there is more stock than one pass "
   + "should take."],
];

function generality() {
  V.$("me-general").innerHTML = `<div class="grid">
    <div class="w5" style="min-width:0">
      <div class="eyebrow">The loop, stated without the welding</div>
      ${MOVES.map((m) => `
        <div class="step">
          <div class="n">${String(m.n).padStart(2, "0")}</div>
          <div class="t">${V.esc(m.name)}</div>
          <div class="d">${V.esc(m.elsewhere)}</div>
          <div class="w"><b>Here:</b> ${V.esc(m.here)}</div>
        </div>`).join("")}
    </div>
    <div class="w7" style="min-width:0">
      <div class="eyebrow">Processes with the same shape</div>
      ${ELSEWHERE.map(([a, b]) => `
        <div class="step">
          <div class="t" style="margin-top:0">${V.esc(a)}</div>
          <div class="d">${V.esc(b)}</div>
        </div>`).join("")}
    </div>
  </div>`;

  V.$("me-general-meta").innerHTML =
    `What carries across is the architecture and the discipline: measure before acting, decide
     per station, refuse rather than guess, score everything against a truth fixed in advance,
     and route the question by how often it is asked. What does not carry across is any number
     on this dashboard. Each process needs its own feature to find, its own process model in
     place of <b>weld_params</b>, its own defect vocabulary, and its own calibration.
     Claiming otherwise would be the same mistake as quoting an accuracy figure measured on
     exact depth, which is the thing the sensor study exists to stop.`;
}

function tiers(ev) {
  const rows = [
    { tier: "nano", question: "Is this measurement good enough to weld on?",
      cadence: "once per part", evidence: ev.router.available
        ? `${ev.router.arms.find((a) => a.arm === "nano").correct}/${ev.router.arms[0].n} correct against ground truth` : "" },
    { tier: "lightning", question: "Held in reserve for the same question under load",
      cadence: "as needed", evidence: ev.router.available
        ? `${ev.router.arms.find((a) => a.arm === "lightning").correct}/${ev.router.arms[0].n} on the accept call, too cautious` : "" },
    { tier: "super", question: "What is going out of adjustment across this run?",
      cadence: "once per shift", evidence: ev.drift.available
        ? `${ev.drift.runs.find((r) => r.tier === "super").correct}/12 causes named, and it improves when the fault gets stronger` : "" },
    { tier: "ultra", question: "Reserved for a disagreement worth the wall clock",
      cadence: "rare", evidence: ev.router.available
        ? `${ev.router.arms.find((a) => a.arm === "ultra").correct}/${ev.router.arms[0].n} on the accept call at ${ev.router.arms.find((a) => a.arm === "ultra").total_latency_s.toFixed(0)} s` : "" },
  ];
  V.$("me-tiers").innerHTML = `<table>
    <thead><tr><th>Tier</th><th>The question it owns</th><th>How often it is asked</th><th>What the study found</th></tr></thead>
    <tbody>${rows.map((r) => `<tr>
      <td><span class="badge ${V.ARM_CLASS[r.tier]}">${V.esc(r.tier)}</span></td>
      <td>${V.esc(r.question)}</td>
      <td class="muted">${V.esc(r.cadence)}</td>
      <td class="muted">${V.esc(r.evidence)}</td>
    </tr>`).join("")}</tbody></table>`;
}
