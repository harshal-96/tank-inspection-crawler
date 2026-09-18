/* The post-weld eye: what it got right, and the two ways it can be wrong. */

import * as V from "../viz.js";
import { loadEvidence, refreshLive, state } from "../store.js";

export const meta = {
  id: "inspection",
  title: "Post-weld inspection",
  crumb: "Stage 4",
  group: "What happens to a part",
  tag: "50",
  icon: "M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Zm10 2.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5Z",
};

const DEFECT_LABEL = {
  none: "clean", skip: "skip", undercut: "undercut",
  porosity: "porosity", spatter: "spatter", burn_through: "burn through",
};

export function render() {
  return `
    <p class="lede">
      After the arc stops, the weld bead is photographed and judged. Every defect
      was put there deliberately, so each verdict can be checked against a record written
      before the model saw the picture. Two kinds of mistake matter differently: a false
      alarm throws away a good part, and a miss ships a bad one.
    </p>

    <div class="tiles" id="in-tiles"></div>

    <div class="grid">
      <div class="card w6">
        <header><h2>Where the verdicts land</h2></header>
        <div class="hint">Rows are what was put into the bead, columns are what came back. Green on
          the diagonal is agreement; everything off it is a named defect the model got wrong.</div>
        <div id="in-matrix"></div>
      </div>

      <div class="card w6">
        <header><h2>Spotting a defect against naming it</h2></header>
        <div class="hint">Both numbers are real, but they are not the same claim. The cell uses the
          pass/defect call to route a part; the named defect is advice for the person who fixes it.</div>
        <div id="in-split"></div>
        <footer id="in-split-meta"></footer>
      </div>

      <div class="card w5">
        <header><h2>False alarms against misses</h2></header>
        <div id="in-errors"></div>
        <footer id="in-errors-meta"></footer>
      </div>

      <div class="card w7">
        <header><h2>Which defects get confused for which</h2></header>
        <div id="in-classes"></div>
      </div>

      <div class="card w12">
        <header><h2>Every model that will look at a bead, on the same fifty beads</h2></header>
        <div class="hint">Each model that accepts a picture was given the same fifty photos and the
          same question. A model may also refuse to judge a bead, which is neither right nor wrong:
          it is a part someone has to go and look at. Those are the grey squares.</div>
        <div id="in-vision"></div>
        <footer id="in-vision-meta"></footer>
      </div>

      <div class="card w12">
        <header><h2>Why the inspector is not a frontier model</h2></header>
        <div class="hint">Every model the account serves was sent one clean bead and one with a
          hole burnt through it, along the real inspection path. Most refuse the picture before
          any model runs. This table is the whole field; the comparison above is the models that
          got past it.</div>
        <div id="in-field"></div>
      </div>

      <div class="card w12">
        <header>
          <h2>Every bead, and the reason given</h2>
          <span class="grow" style="flex:1"></span>
          <div class="seg" id="in-filter">
            <button data-f="all" class="on">All</button>
            <button data-f="wrong">Got it wrong</button>
            <button data-f="alarm">False alarms</button>
            <button data-f="miss">Missed</button>
          </div>
        </header>
        <div id="in-cases"></div>
      </div>
    </div>`;
}

let ev = null;
let filter = "all";

export async function mount() {
  [ev] = await Promise.all([loadEvidence(), refreshLive()]);
  if (!ev.inspection.available) return;
  tiles();
  matrix();
  split();
  errors();
  classes();
  vision();
  field();
  V.$("in-filter").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      filter = b.dataset.f;
      V.$("in-filter").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      cases();
    };
  });
  cases();
}

function tiles() {
  const i = ev.inspection;
  V.$("in-tiles").innerHTML = [
    tile("Pass or defect", V.pct(i.binary_accuracy), "good",
      `${i.binary_correct} of ${i.n} beads routed correctly`),
    tile("Defect named exactly", V.pct(i.named_accuracy), "warn",
      `${i.named_correct} of ${i.n}, which is the harder question`),
    tile("False alarms", `${i.false_alarms}`, "warn",
      `good parts called defective, out of ${i.clean} clean beads`),
    tile("Missed", `${i.missed}`, "bad",
      `faulty parts called clean, out of ${i.faulty} with a defect`),
    tile("Median verdict time", V.fmt(i.median_latency_ms / 1000, 1), "",
      `seconds, on ${(i.model || "").split("/").pop()}`, "s"),
  ].join("");
}

const tile = (label, value, tone, sub, unit = "") => `<div class="tile">
  <div class="label">${V.esc(label)}</div>
  <div class="value ${tone}">${value}${unit ? `<span class="unit">${unit}</span>` : ""}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

function matrix() {
  const i = ev.inspection;
  const labels = i.labels.map((l) => DEFECT_LABEL[l] || l);
  V.paint(V.$("in-matrix"), V.heat(labels, labels, i.matrix, {
    cell: 54, padL: 96, padT: 58,
    xTitle: "rows: injected · columns: called",
    tip: (r, c, v) => v === 0 ? `no beads`
      : `<b>${v}</b> bead${v > 1 ? "s" : ""} with <b>${labels[r]}</b> injected
         ${r === c ? "were named correctly" : `were called <b>${labels[c]}</b>`}`,
  }));
}

function split() {
  const i = ev.inspection;
  const rows = [
    { label: "Pass or defect", value: i.binary_accuracy, colour: V.GOOD,
      display: `${i.binary_correct}/${i.n}`,
      tip: "Did it decide correctly whether this bead has a problem at all" },
    { label: "Defect named exactly", value: i.named_accuracy, colour: V.WARN,
      display: `${i.named_correct}/${i.n}`,
      tip: "Did it also name the right defect out of five classes" },
  ];
  V.paint(V.$("in-split"), V.ranked(rows, { width: 420, rowH: 46, padL: 156, max: 1 }));
  V.$("in-split-meta").innerHTML =
    `The gap between the two is where the model sees something wrong but calls it by the wrong
     name, most often reaching for <b>skip</b> when the bead has undercut or burn through.
     Both numbers are quoted because only one of them is the production decision: the cell routes
     on the <b>${V.pct(i.binary_accuracy)}</b> figure, and the named defect is a hint, not a gate.`;
}

function errors() {
  const i = ev.inspection;
  const w = 340, h = 190, padL = 12, padT = 34;
  const cellW = (w - padL - 12) / 2, cellH = 62;
  const cells = [
    { r: 0, c: 0, n: i.clean - i.false_alarms, label: "clean, called clean", tone: "var(--good)" },
    { r: 0, c: 1, n: i.false_alarms, label: "clean, called defective", tone: "var(--warning)" },
    { r: 1, c: 0, n: i.missed, label: "defect, called clean", tone: "var(--critical)" },
    { r: 1, c: 1, n: i.faulty - i.missed, label: "defect, called defective", tone: "var(--good)" },
  ];
  let out = `<text class="axis-label" x="${padL + cellW / 2}" y="${padT - 12}" text-anchor="middle">called clean</text>
    <text class="axis-label" x="${padL + cellW * 1.5}" y="${padT - 12}" text-anchor="middle">called defective</text>`;
  cells.forEach((c) => {
    const x = padL + c.c * cellW, y = padT + c.r * cellH;
    out += `<rect x="${x + 2}" y="${y + 2}" width="${cellW - 4}" height="${cellH - 4}" rx="9"
      fill="color-mix(in oklab, ${c.tone} ${c.n ? 13 + Math.min(34, c.n * 2) : 3}%, var(--surface-1))"
      stroke="var(--bg)" stroke-width="2"
      data-tip="<b>${c.n}</b> beads: ${V.esc(c.label)}"/>
      <text x="${x + cellW / 2}" y="${y + cellH / 2 + 2}" text-anchor="middle" font-size="24"
        font-weight="600" fill="${c.n ? c.tone : "var(--text-muted)"}">${c.n}</text>
      <text x="${x + cellW / 2}" y="${y + cellH - 10}" text-anchor="middle" font-size="9"
        fill="var(--text-muted)">${V.esc(c.label)}</text>`;
  });
  out += `<text class="axis-label" x="${padL}" y="${padT + cellH * 2 + 18}">
    a false alarm scraps a good part · a miss ships a bad one</text>`;
  V.paint(V.$("in-errors"), V.svg(w, h, out));
  V.$("in-errors-meta").innerHTML =
    `On this run the eye leans toward calling a clean bead defective
     (<b>${i.false_alarms}</b>) slightly more than letting a bad one through (<b>${i.missed}</b>),
     which is the safer way round for a cell that feeds a downstream process, and the wrong way
     round for scrap cost. Which way it should lean is a shop decision, and it is a threshold,
     not a retrain.`;
}

function classes() {
  const i = ev.inspection;
  const rows = i.labels.filter((l) => l !== "none").map((label, k) => {
    const idx = i.labels.indexOf(label);
    const injected = i.matrix[idx].reduce((a, b) => a + b, 0);
    const right = i.matrix[idx][idx];
    const calledClean = i.matrix[idx][i.labels.indexOf("none")] || 0;
    const wrongName = injected - right - calledClean;
    return { label, injected, right, wrongName, calledClean };
  }).filter((r) => r.injected > 0);

  const w = 480, rowH = 34, padL = 116, padR = 92;
  const h = rows.length * rowH + 34;
  const hi = Math.max(...rows.map((r) => r.injected));
  const xOf = (v) => (v / hi) * (w - padL - padR);
  let out = V.axisTitle(padL, 14, "beads with this defect injected");
  rows.forEach((r, k) => {
    const y = 26 + k * rowH, bh = rowH - 14;
    out += `<text x="${padL - 10}" y="${y + bh / 2 + 4}" text-anchor="end" font-size="11"
      fill="var(--text-secondary)" font-family="var(--sans)">${V.esc(DEFECT_LABEL[r.label] || r.label)}</text>`;
    let x = padL;
    const segs = [
      { n: r.right, colour: V.GOOD, what: "named correctly" },
      { n: r.wrongName, colour: V.WARN, what: "seen but named wrong" },
      { n: r.calledClean, colour: V.BAD, what: "called clean" },
    ];
    segs.forEach((s) => {
      if (s.n > 0) {
        out += `<rect x="${x + 1}" y="${y}" width="${Math.max(0, xOf(s.n) - 2)}" height="${bh}" rx="4"
          fill="${s.colour}" opacity=".88"
          data-tip="<b>${s.n}</b> of ${r.injected} <b>${V.esc(DEFECT_LABEL[r.label] || r.label)}</b> beads ${s.what}"/>`;
      }
      x += xOf(s.n);
    });
    out += `<text x="${padL + xOf(r.injected) + 10}" y="${y + bh / 2 + 4}" font-size="11"
      font-family="var(--mono)" fill="var(--text-secondary)">${r.right}/${r.injected}</text>`;
  });
  V.paint(V.$("in-classes"), V.svg(w, h, out));
  V.$("in-classes").insertAdjacentHTML("afterend", `<div class="legend">
    <span><i style="background:var(--good)"></i>named correctly</span>
    <span><i style="background:var(--warning)"></i>seen, named wrong</span>
    <span><i style="background:var(--critical)"></i>called clean</span></div>`);
}

function vision() {
  const host = V.$("in-vision");
  if (!ev.vision || !ev.vision.available || ev.vision.arms.length < 2) {
    if (host) host.innerHTML = `<div class="empty">Only one vision model has been scored.</div>`;
    return;
  }
  const arms = ev.vision.arms;
  const w = 720, rowH = 74, padL = 190, padR = 24;
  const h = arms.length * rowH + 44;
  const track = w - padL - padR;
  let out = V.axisTitle(padL, 16, "fifty beads: clean called clean, defect caught, "
                                  + "false alarm, missed, declined to judge");

  arms.forEach((a, i) => {
    const y = 34 + i * rowH;
    const isCell = a.model === ev.vision.cell_model;
    out += `<text x="${padL - 12}" y="${y + 13}" text-anchor="end" font-size="12"
      fill="var(--text-primary)" font-family="var(--sans)"
      font-weight="${isCell ? 700 : 400}">${V.esc(a.model)}${isCell ? " ●" : ""}</text>`;
    out += `<text x="${padL - 12}" y="${y + 29}" text-anchor="end" font-size="10.5"
      fill="var(--text-muted)" font-family="var(--sans)">${a.named_correct} right of the ${a.judged} it judged</text>`;

    const segments = [
      { n: a.clean - a.false_alarms, colour: "var(--good)", what: "clean, called clean" },
      { n: a.faulty - a.missed, colour: "var(--series-1)", what: "defect, caught" },
      { n: a.false_alarms, colour: "var(--warning)", what: "false alarm, a good part scrapped" },
      { n: a.missed, colour: "var(--critical)", what: "missed, a bad part shipped" },
      { n: a.declined, colour: "var(--band)", what: "declined to judge, a part someone must look at" },
    ];
    let x = padL;
    segments.forEach((seg) => {
      if (seg.n > 0) {
        const sw = (seg.n / a.n) * track;
        out += `<rect x="${x + 1}" y="${y}" width="${Math.max(0, sw - 2)}" height="24" rx="5"
          fill="${seg.colour}" opacity=".9"
          data-tip="<b>${seg.n}</b> of ${a.n}: ${V.esc(seg.what)}"/>`;
        if (sw > 34) out += `<text x="${x + sw / 2}" y="${y + 16}" text-anchor="middle"
          font-size="11" fill="${seg.colour === "var(--band)" ? "var(--text-primary)" : "#0b0e13"}"
          font-weight="600">${seg.n}</text>`;
        x += sw;
      }
    });
    const verdict = a.judged === 0 ? "judged nothing"
      : a.never_says_defect ? "never once used the word defect"
      : `named ${a.defects_named} different defect type${a.defects_named === 1 ? "" : "s"}`;
    out += `<text x="${padL}" y="${y + 44}" font-size="11" fill="var(--text-muted)"
      font-family="var(--sans)">${V.esc(verdict)} · median
      ${V.fmt(a.median_latency_ms / 1000, 1)} s per bead</text>`;
  });
  V.paint(host, V.svg(w, h, out));

  const cell = arms.find((a) => a.model === ev.vision.cell_model) || arms[0];
  const quiet = arms.find((a) => a !== cell && a.never_says_defect && a.judged === a.n);
  // The nearest rival is the one that got the most right while declining some.
  const shy = arms.filter((a) => a !== cell && a.declined > 0)
    .sort((p, q) => q.named_correct - p.named_correct)[0];
  const parts = [`<b>${V.esc(cell.model)}</b> is the one the cell uses: it judged every bead,
    named <b>${cell.named_correct}</b> of ${cell.n} right, with <b>${cell.false_alarms}</b> false
    alarms and <b>${cell.missed}</b> misses.`];
  if (quiet) parts.push(`<b>${V.esc(quiet.model)}</b> reaches ${quiet.named_correct} of
    ${quiet.n} by calling every bead a pass, and ships <b>${quiet.missed}</b> bad parts.`);
  if (shy) parts.push(`<b>${V.esc(shy.model)}</b> got ${shy.named_correct} right, but only of
    the ${shy.judged} it would judge: it declined <b>${shy.declined}</b> beads, missed
    <b>${shy.missed}</b>, and took ${V.fmt(shy.median_latency_ms / 1000, 0)} seconds a bead.
    A declined bead is a part someone has to go and look at.`);
  V.$("in-vision-meta").innerHTML = parts.join(" ");
}

function field() {
  const host = V.$("in-field");
  const f = ev.vision_field;
  if (!host) return;
  if (!f || !f.available) {
    host.innerHTML = `<div class="empty">The account has not been probed yet:
      <code>python scripts/probe_vision_field.py</code></div>`;
    return;
  }
  const say = {
    judged: ["judged both beads", "var(--good)"],
    declined: ["took the picture, would not judge a rendering", "var(--warning)"],
    error: ["failed", "var(--critical)"],
    refused: ["refuses any picture", "var(--text-muted)"],
  };
  const c = f.counts;
  const rows = f.models.map((m) => {
    const [label, colour] = say[m.outcome];
    return `<tr>
      <td><span class="dot" style="display:inline-block;margin-right:7px;background:${colour}"></span>${V.esc(m.raw_model)}</td>
      <td>${label}</td>
      <td class="num">${m.outcome === "refused" ? "" : V.esc(m.clean)}</td>
      <td class="num">${m.outcome === "refused" ? "" : V.esc(m.defect)}</td>
      <td class="num">${V.fmt(m.seconds, 1)}</td>
    </tr>`;
  }).join("");
  host.innerHTML = `
    <div class="kv" style="margin-bottom:10px">
      <span><b>${f.n}</b> <span class="k">models on the account</span></span><span class="sep"></span>
      <span><b>${c.refused}</b> <span class="k">refuse a picture</span></span><span class="sep"></span>
      <span><b>${c.declined}</b> <span class="k">take it, will not judge</span></span><span class="sep"></span>
      <span><b>${c.judged}</b> <span class="k">judge it</span></span>
    </div>
    <table class="plain">
      <thead><tr><th>Model</th><th>What happened</th><th>Clean bead</th>
        <th>Burnt-through bead</th><th>seconds</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function cases() {
  const rows = ev.inspection.cases.filter((c) => {
    if (filter === "wrong") return !c.correct;
    if (filter === "alarm") return c.injected === "none" && c.called !== "none";
    if (filter === "miss") return c.injected !== "none" && c.called === "none";
    return true;
  });
  const host = V.$("in-cases");
  if (!rows.length) { host.innerHTML = `<div class="empty">Nothing in this group.</div>`; return; }
  host.innerHTML = `<table>
    <thead><tr><th>Bead</th><th>Injected</th><th>Called</th><th class="num">Severity</th>
      <th class="num">ms</th><th>Reasoning</th></tr></thead>
    <tbody>${rows.map((c) => `<tr>
      <td class="mono muted">#${c.trial}</td>
      <td><span class="badge">${V.esc(DEFECT_LABEL[c.injected] || c.injected)}</span></td>
      <td><span class="badge ${c.correct ? "good" : (c.injected === "none" ? "warn" : "bad")}">${V.esc(DEFECT_LABEL[c.called] || c.called)}</span></td>
      <td class="num">${V.fmt(c.severity, 2)}</td>
      <td class="num muted">${V.fmt(c.latency_ms, 0)}</td>
      <td class="quote" style="border:none;padding-left:0;max-width:520px">${V.esc(c.reasoning || "")}</td>
    </tr>`).join("")}</tbody></table>`;
}
