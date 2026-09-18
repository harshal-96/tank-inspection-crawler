/* Everything between the joint and the torch, and the one term the cell cannot see. */

import * as V from "../viz.js";
import { loadEvidence } from "../store.js";

export const meta = {
  id: "budget",
  title: "Error budget",
  crumb: "Study 7",
  group: "Studies",
  icon: "M4 19V5m0 14h16M8 15V9m4 6V6m4 9v-4",
};

export function render() {
  return `
    <p class="lede">
      The seam study says the cell finds the joint to 0.68 mm. That number is true,
      and it covers one link in a chain. Quoting it as the accuracy of the cell would be
      misleading. Published work on vision guided welding is clear that what you actually
      get depends on the whole chain: the sensor, the calibration, the standoff, the part
      surface, the robot's own repeatability. This project measured two of those carefully
      and said nothing about the rest. Here is the whole chain, with a source for every
      term in it.
    </p>

    <div class="tiles" id="bg-tiles"></div>

    <div class="grid">
      <div class="card w12">
        <header><h2>The chain, under four sets of assumptions</h2></header>
        <div class="hint">Systematic terms add, because an offset that is the same on every
          part is not averaged away by making more parts. Independent random terms combine in
          quadrature. Whether a term is systematic or random decides how much it ends up mattering.</div>
        <div id="bg-chain"></div>
        <div class="legend">
          <span><i style="background:var(--series-1)"></i>measured in this repository</span>
          <span><i style="background:var(--warning)"></i>published range, not measured here</span>
          <span><i style="background:var(--before)"></i>manufacturer specification</span>
          <span><i class="line" style="background:var(--critical)"></i>what industrial weld guidance is asked for</span>
        </div>
      </div>

      <div class="card w7">
        <header><h2>Every term, and where its number came from</h2></header>
        <div id="bg-terms"></div>
      </div>

      <div class="card w5">
        <header><h2>The term the cell cannot see</h2></header>
        <div class="hint">Hand eye calibration is both the largest unmeasured contribution
          and, on the seam alone, undetectable. A camera whose calibration has crept moves
          every measurement equally at the bottom and the top, which is exactly what a
          creeping fixture does.</div>
        <div id="bg-confound"></div>
        <footer id="bg-confound-meta"></footer>
      </div>
    </div>`;
}

const SOURCE_COLOUR = {
  "measured here": "var(--series-1)",
  "published range": "var(--warning)",
  "specification": "var(--before)",
};

export async function mount() {
  const ev = await loadEvidence();
  const b = ev.error_budget;
  if (!b || !b.available) {
    V.$("bg-chain").innerHTML = `<div class="empty">The budget is unavailable.</div>`;
    return;
  }
  tiles(b);
  chain(b);
  terms(b);
  confound(ev.confound, b);
}

function tiles(b) {
  const quoted = b.budgets[0];
  const realistic = b.budgets.find((x) => x.label.startsWith("Typical stereo"));
  const best = b.budgets.find((x) => x.label.startsWith("Laser"));
  const handEye = realistic.terms.find((t) => t.name === "Hand eye calibration");

  V.$("bg-tiles").innerHTML = [
    tile("What the seam study quotes", V.fmt(quoted.total_mm, 2), "after",
      "mm, and it is one term of five", " mm"),
    tile("The whole chain, realistically", V.fmt(realistic.total_mm, 2), "bad",
      `mm on a stereo camera, against a ${b.industry_requirement_mm} mm requirement`, " mm"),
    tile("With the sensor this needs", V.fmt(best.total_mm, 2), "good",
      `mm on a laser profiler, which does meet ${b.industry_requirement_mm} mm`, " mm"),
    tile("Largest unmeasured term", V.fmt(handEye.mm, 2), "warn",
      "mm of hand eye calibration, bigger than the tracking error itself", " mm"),
    tile("Terms measured here", "2", "", "of five. The other three are published figures"),
  ].join("");
}

const tile = (label, value, tone, sub, unit = "") => `<div class="tile">
  <div class="label">${V.esc(label)}</div>
  <div class="value ${tone}">${value}${unit ? `<span class="unit">${unit}</span>` : ""}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

function chain(b) {
  const rows = b.budgets;
  const w = 900, rowH = 62, padL = 250, padR = 96, padT = 30;
  const h = padT + rows.length * rowH + 30;
  const hi = Math.max(...rows.map((r) => r.total_mm)) * 1.12;
  const xOf = (v) => (v / hi) * (w - padL - padR);

  let out = V.axisTitle(padL, 16, "millimetres between the torch and the joint");
  for (let tick = 0; tick <= hi; tick += 0.5) {
    const x = padL + xOf(tick);
    out += `<line class="gridline" x1="${x}" y1="${padT - 8}" x2="${x}" y2="${h - 26}"/>
      <text class="axis-label" x="${x}" y="${h - 10}" text-anchor="middle">${tick.toFixed(1)}</text>`;
  }
  const req = padL + xOf(b.industry_requirement_mm);
  out += `<line x1="${req}" y1="${padT - 12}" x2="${req}" y2="${h - 26}"
    stroke="var(--critical)" stroke-width="2" stroke-dasharray="4 3"/>
    <text class="axis-label" x="${req + 6}" y="${padT - 14}" fill="var(--critical)">
      ${b.industry_requirement_mm} mm asked of industrial weld guidance</text>`;

  rows.forEach((budget, i) => {
    const y = padT + i * rowH;
    out += `<text x="${padL - 14}" y="${y + 14}" text-anchor="end" font-size="12"
      fill="var(--text-primary)" font-family="var(--sans)">${V.esc(budget.label)}</text>`;

    // Stacked by contribution, which is not how they combine but is how they are
    // read. The total drawn is the correct combination, marked separately.
    let x = padL;
    const scale = budget.total_mm > 0
      ? xOf(budget.total_mm) / budget.terms.reduce((a, t) => a + t.mm, 0) : 0;
    budget.terms.forEach((t) => {
      const width = t.mm * scale;
      if (width > 0.4) {
        out += `<rect x="${x + 0.8}" y="${y + 22}" width="${Math.max(0, width - 1.6)}"
          height="20" rx="4" fill="${SOURCE_COLOUR[t.source] || "var(--surface-3)"}"
          opacity=".92"
          data-tip="<b>${V.esc(t.name)}</b> <b>${V.fmt(t.mm, 3)} mm</b><br>
            <span class='k'>${V.esc(t.kind)}, ${V.esc(t.source)}</span><br>
            ${V.esc(t.note)}${t.cite ? `<br><span class='k'>${V.esc(t.cite)}</span>` : ""}"/>`;
      }
      x += width;
    });
    const end = padL + xOf(budget.total_mm);
    out += `<text x="${end + 10}" y="${y + 36}" font-size="12.5" font-family="var(--mono)"
      fill="${budget.total_mm < b.industry_requirement_mm ? "var(--after)" : "var(--critical)"}">
      ${V.fmt(budget.total_mm, 2)} mm</text>`;
    out += `<text x="${padL}" y="${y + 56}" font-size="10.5" fill="var(--text-muted)"
      font-family="var(--sans)">${V.fmt(budget.systematic_mm, 2)} mm systematic, adds
      &#183; ${V.fmt(budget.random_mm, 2)} mm random, in quadrature</text>`;
  });
  V.paint(V.$("bg-chain"), V.svg(w, h, out));
}

function terms(b) {
  const realistic = b.budgets.find((x) => x.label.startsWith("Typical stereo"));
  V.$("bg-terms").innerHTML = `<table>
    <thead><tr><th>Term</th><th class="num">mm</th><th>Kind</th><th>Where the number is from</th>
      <th>Can this cell fix it?</th></tr></thead>
    <tbody>${realistic.terms.map((t) => `<tr>
      <td><b>${V.esc(t.name)}</b><div class="muted" style="font-size:11.5px;margin-top:3px">
        ${V.esc(t.note)}</div></td>
      <td class="num">${V.fmt(t.mm, 3)}</td>
      <td><span class="badge ${t.kind === "systematic" ? "warn" : ""}">${V.esc(t.kind)}</span>
        <div class="muted" style="font-size:10.5px">${
          t.averages_down ? "averages down with parts" : "does not average down"}</div></td>
      <td class="muted">${V.esc(t.source)}${t.cite
        ? `<div style="font-size:10.5px;margin-top:3px">${V.esc(t.cite)}</div>` : ""}</td>
      <td><span class="badge ${t.fixable ? "good" : "bad"}">${t.fixable ? "yes" : "no"}</span></td>
    </tr>`).join("")}</tbody></table>`;
}

function confound(c, b) {
  const host = V.$("bg-confound");
  if (!c || !c.available) {
    host.innerHTML = `<div class="empty">The confound study has not been run.</div>`;
    return;
  }
  const rows = c.arms.map((a) => ({
    label: a.label,
    value: a.correct,
    display: `${a.correct}/${a.n}`,
    colour: a.reference ? "var(--after)" : "var(--before)",
    tip: `<b>${a.label}</b><br>` + Object.entries(a.by_scenario)
      .map(([k, v]) => `<span class='k'>${k}</span> <b>${v.correct}/${v.n}</b>`).join("<br>"),
  }));
  V.paint(host, V.ranked(rows, { width: 380, rowH: 40, padL: 136,
                                 max: Math.max(...c.arms.map((a) => a.n)) }));

  V.$("bg-confound-meta").innerHTML =
    `Asked to name one of four causes, <b>${V.esc((c.model || "").split("/").pop())}</b> scored
     <b>${c.arms[0].correct}/${c.arms[0].n}</b> on the seam alone and
     <b>${c.arms[1].correct}/${c.arms[1].n}</b> with a reference feature bolted to the machine
     base. The drifting camera went from <b>${c.camera_without.correct}/${c.camera_without.n}</b>
     to <b>${c.camera_with.correct}/${c.camera_with.n}</b>: without the instrument it was never
     identified once, and the answer given instead was always the fixture. Same model, same
     prompt structure, one extra column. What was missing was not intelligence, it was an
     instrument, and it is the same instrument that would catch the
     <b>${V.fmt(b.budgets[2].terms.find((t) => t.name === "Hand eye calibration").mm, 2)} mm</b>
     calibration term above going out.`;
}
