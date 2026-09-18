/* The headline page: what the cell claims, and the evidence for it, above the fold. */

import * as V from "../viz.js";
import { state, loadEvidence, refreshLive, loadCell, chosen } from "../store.js";

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
      + `${seam.stations_measured} points`]);
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
      A welding robot follows the same path every time, but folded sheet metal never comes
      out of the press the same shape twice, so the joint is never quite where the path
      says. This cell finds the joint on every part and welds there. Fifty parts, measured.
    </p>

    <div class="stage brief">
      <canvas id="ov-view"></canvas>
      <div class="overlay-tl" id="ov-chips"></div>
      <div class="overlay-bl">drag to orbit &middot; scroll to zoom &middot; true scale</div>
      <div class="overlay-br">
        <div class="glass legend" style="margin:0">
                    <span><i class="line" style="background:var(--series-1)"></i>where the robot was taught to weld</span>
          <span><i class="line" style="background:var(--good)"></i>the corrected path the cell welds along</span>
          <span><i style="background:var(--warning)"></i>gap needs attention</span>
          <span><i style="background:var(--before)"></i>too wide, not welded</span>
        </div>
      </div>
    </div>

    <div class="photos">
      <div class="photo">
        <img src="/figures/seam_tracked.png" alt="One part seen by the depth camera, with the taught path and the measured joint drawn on it">
        <div class="cap"><b>What the camera sees.</b> One part, straight from the depth frame.
          Blue is where the robot was taught to weld, red is where the joint really is, and
          green is the path the cell measured and welded along, at its twenty points. On this
          part the taught path is 34 mm off.</div>
      </div>
      <div class="photo">
        <div class="beads">
          <figure><img src="/demo/beads/rgb_0009.png" alt="A clean weld bead">
            <figcaption class="ok">clean &middot; called pass</figcaption></figure>
          <figure><img src="/demo/beads/rgb_0039.png" alt="A weld bead with burn-through">
            <figcaption class="bad">burn-through &middot; called defect</figcaption></figure>
        </div>
        <div class="cap"><b>What the inspection camera sees.</b> Two beads from the shipped
          run. The defect was put there on purpose before the picture was taken, so the
          verdict has an answer to be marked against.</div>
      </div>
    </div>

    <div class="grid">
      <div class="card w12">
        <header><h2>What happens to a part</h2></header>
        <div class="hint">Every box links to the page that measured it. The two dashed lines
          are what makes the cell self-correcting: a measurement it does not trust goes round
          for another look, and a machine it catches slipping gets adjusted before it spoils
          the next part.</div>
        <div id="ov-flow"></div>
        <div id="ov-machines"></div>
      </div>
    </div>

    <div id="ov-live"></div>
    <div class="tiles" id="ov-tiles"></div>

    <div class="grid">
      <div class="card w12">
        <header><h2>Which model won each job</h2></header>
        <div class="hint"><span id="ov-models-n">Every model on the account was</span> scored.
          They are not interchangeable and bigger is not better, so each question goes to
          whichever one measured best on it.</div>
        <div id="ov-models"></div>
      </div>
    </div>

    <div class="grid">
      <div class="card w12">
        <header><h2>What this cell does that a demonstration usually does not</h2></header>
        <div id="ov-new"></div>
      </div>
    </div>

    <div class="grid">
      <div class="card w7">
        <header><h2>Every part: where a taught path would have welded, and where the cell did</h2></header>
        <div class="hint">One row per part. Left dot is the taught path's miss, right dot is
          the measured joint's error. Log scale, so both ends stay readable.</div>
        <div id="ov-dumbbell"></div>
        <div class="legend">
          <span><i style="background:var(--before)"></i>taught path, no correction</span>
          <span><i style="background:var(--after)"></i>measured joint, mean of 20 points</span>
        </div>
      </div>

      <div class="card w5">
        <header><h2>How close every part came</h2></header>
        <div class="hint">The welding wire is 1.0 mm across, so anything left of the dashed
          line is inside a wire diameter.</div>
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

export function update() { liveBand(); drawPart(); }

let view = null;

export async function mount() {
  const [ev, geo] = await Promise.all([loadEvidence(), loadCell(), refreshLive()]);
  liveBand();
  flow(ev);
  tiles(ev);
  dumbbell(ev);
  hist(ev);
  studies(ev);
  models(ev);
  novelty(ev);
  await viewport(geo);
}

/** The cell in three dimensions, framed on the joint, exaggerated so the miss reads.

    A number says the taught path is 22.8 mm off. A picture of the taught path
  running beside the real joint says it without a number, which is why this sits
  at the top of the page rather than in a tile. It is true scale, and says so: an
  earlier version exaggerated it tenfold and put the part through its own fixture.
*/
async function viewport(geo) {
  const canvas = V.$("ov-view");
  if (!canvas || !geo) return;
  try {
    const { Cell3D } = await import("../cell3d.js");
        view = new Cell3D(canvas);
    view.build(geo);
  } catch (err) {
    canvas.insertAdjacentHTML("afterend",
      `<div class="empty">3D view unavailable: ${V.esc(err.message)}</div>`);
    view = null;
    return;
  }
  // Draw first, then frame: the camera aims between the taught line and the
  // seam as shown, and it cannot know where that is until the seam is drawn.
  drawPart();
  view.frameSeam();
}

function drawPart() {
  const slot = chosen();
  if (!slot || !view) return;
  view.showSeam(slot.seam, slot.schedule);
  const t = slot.seam, chips = [];
  if (t) {
    chips.push(glass("Where the robot was taught", `${V.fmt(t.taught_miss_mm, 1)} mm off`,
      "from the real joint", "var(--series-1)"));
    chips.push(glass("Where the cell welded", `${V.fmt(t.error_mm, 2)} mm off`,
      `worst of ${t.stations_measured} points ${V.fmt(t.worst_station_mm, 2)} mm`, "var(--good)"));
  }
    chips.push(glass("Display", "true scale", "nothing exaggerated", "var(--text-muted)"));
  V.$("ov-chips").innerHTML = chips.join("");
}

const glass = (label, value, sub, colour) => `<div class="glass">
  <div class="label">${V.esc(label)}</div>
  <div class="value" style="color:${colour}">${value}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

export function retheme() { view?.retheme?.(); }

export function unmount() {
  view?.dispose?.();
  view = null;
}

/** The process, drawn. Two loops are the whole point of the cell.

  This was a list of six numbered rows, which is a table of contents pretending to
  be a process. What a list cannot show is that the flow turns back on itself
  twice: a measurement the planner does not trust goes round for another look, and
  a drift the cell names goes back to the machine upstream that caused it. Those
  loops are why it is called self-correcting.

  The layout reserves lanes before it places anything: a return lane down the left
  for the upstream loop, a band above the first row for the look-again loop, and a
  band below the last row for the caption. The first version drew the return path
  straight through two of the boxes.
*/
function flow(ev) {
  const host = V.$("ov-flow");
  if (!host) return;
  const s = ev.seam, r = ev.router, i = ev.inspection, h = ev.hard;

  const LANE = 34;                 // the return path down the left
  const X0 = 78;                   // where boxes start, clear of the lane
  const bw = 174, bh = 74, gap = 30;
  const LOOP = 92;                 // the look-again band, above row one
  const row1 = 146, row2 = 300;
  const BACK = row2 + bh + 40;     // the upstream return, below row two
  const W = X0 + 4 * (bw + gap) + 34, H = BACK + 46;

  const xOf = (n) => X0 + n * (bw + gap);
  const mid1 = row1 + bh / 2, mid2 = row2 + bh / 2;

  const top = [
    ["1", "Find the joint", s.available ? `${V.fmt(s.tracked.mean_mm, 3)} mm` : "", false, "seam"],
    ["2", "Trust it?", r.available ? `${r.arms[0].correct}/${r.arms[0].n}` : "", true, "routing"],
    ["3", "Aim ahead of heat", "0.55 \u2192 0.03 mm", false, "compensation"],
  ];
  const bottom = [
    ["4", "Judge the bead", i.available ? V.pct(i.binary_accuracy) : "", true, "inspection"],
    ["5", "Which machine slipped", h && h.available ? V.pct(h.arms[0].rate) : "", true, "diagnosis"],
    ["6", "Tell the operator", "with sources", true, "briefs"],
  ];

  const node = (x, y, [no, title, figure, usesModel, page]) => `
    <a href="#/${page}" class="flow-node">
      <rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="9"
        fill="var(--surface-1)" stroke="${usesModel ? "var(--series-1)" : "var(--line)"}"
        stroke-width="${usesModel ? 1.6 : 1}"/>
      <text x="${x + 14}" y="${y + 22}" font-size="10.5" font-weight="600"
        fill="var(--after)">${no}</text>
      <text x="${x + 31}" y="${y + 22}" font-size="13" font-weight="600"
        fill="var(--text-primary)">${V.esc(title)}</text>
      <text x="${x + 14}" y="${y + 45}" font-size="14.5" font-weight="500"
        fill="var(--text-primary)">${V.esc(figure)}</text>
      <text x="${x + 14}" y="${y + 63}" font-size="9.5"
        fill="var(--text-muted)">${usesModel ? "a model answers this" : "geometry only"}</text>
    </a>`;

  const arrow = (x1, y1, x2, y2) => `<path d="M ${x1} ${y1} L ${x2} ${y2}"
    stroke="var(--line)" stroke-width="1.6" fill="none" marker-end="url(#ar)"/>`;

  let out = `<defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"
      orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-muted)"/>
    </marker>
    <marker id="ar-look" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"
      orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--series-1)"/>
    </marker>
    <marker id="ar-back" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"
      orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--after)"/>
    </marker>
  </defs>`;

  // Row one, left to right, into the arc.
  top.forEach((stage, n) => { out += node(xOf(n), row1, stage); });
  for (let n = 0; n < top.length - 1; n++) {
    out += arrow(xOf(n) + bw, mid1, xOf(n + 1) - 7, mid1);
  }
  const arcX = xOf(3) + 34;
  out += arrow(xOf(2) + bw, mid1, arcX - 34, mid1);
  out += `<circle cx="${arcX}" cy="${mid1}" r="28" fill="var(--surface-1)"
    stroke="var(--warning)" stroke-width="1.6"/>`;
  out += `<text x="${arcX}" y="${mid1 + 4}" text-anchor="middle" font-size="11.5"
    font-weight="600" fill="var(--warning)">WELD</text>`;

  // The arc drops into row two, round the right-hand side.
  out += `<path d="M ${arcX} ${mid1 + 28} L ${arcX} ${row2 - 30}
    L ${xOf(0) + bw / 2} ${row2 - 30} L ${xOf(0) + bw / 2} ${row2 - 7}"
    stroke="var(--line)" stroke-width="1.6" fill="none" marker-end="url(#ar)"/>`;

  bottom.forEach((stage, n) => { out += node(xOf(n), row2, stage); });
  for (let n = 0; n < bottom.length - 1; n++) {
    out += arrow(xOf(n) + bw, mid2, xOf(n + 1) - 7, mid2);
  }

  // Loop one, in its own band above row one.
  out += `<path d="M ${xOf(1) + bw / 2} ${row1} L ${xOf(1) + bw / 2} ${LOOP}
    L ${xOf(0) + bw / 2} ${LOOP} L ${xOf(0) + bw / 2} ${row1 - 7}"
    stroke="var(--series-1)" stroke-width="1.5" fill="none" stroke-dasharray="4 3"
    marker-end="url(#ar-look)"/>`;
  out += `<text x="${xOf(1) + bw / 2 + 12}" y="${LOOP + 4}" font-size="11"
    fill="var(--series-1)">not good enough: look again</text>`;

  // Loop two, down the reserved lane on the left.
  out += `<path d="M ${xOf(2) + bw / 2} ${row2 + bh} L ${xOf(2) + bw / 2} ${BACK}
    L ${LANE} ${BACK} L ${LANE} ${row1 - 46}"
    stroke="var(--after)" stroke-width="1.7" fill="none" stroke-dasharray="5 4"
    marker-end="url(#ar-back)"/>`;
  out += `<text x="${LANE + 14}" y="${BACK + 19}" font-size="11" font-weight="600"
    fill="var(--after)">go and adjust that machine, before it spoils the next part
    the same way</text>`;

  // What arrives, at the top left where the part enters, not beside the torch.
  out += `<text x="${LANE}" y="24" font-size="10.5" font-weight="600"
    fill="var(--text-muted)" letter-spacing="1">A PART ARRIVES</text>`;
  out += `<text x="${LANE}" y="42" font-size="12" fill="var(--before)">its joint is
    ${s.available ? V.fmt(s.taught.mean_mm, 1) : "22.8"} mm from where the program expects
    it, and bowed</text>`;
  out += `<text x="${arcX + 42}" y="${mid1 + 4}" font-size="11"
    fill="var(--text-muted)">the arc runs</text>`;

  V.paint(host, V.svg(W, H, out));
  machines();
}

/** Which machine, in plain words.

  "Drift" and "the drifting machine" told a reader nothing. There are exactly
  three machines that can put the weld in the wrong place, they fail in different
  ways, and a fitter has to walk to a different part of the shop for each one.
  That is the whole reason the cell names which.
*/
function machines() {
  const host = V.$("ov-machines");
  if (!host) return;
  const rows = [
    ["The press brake", "var(--series-2)",
     "Folds the flat sheet up into a tray. If its angle setting slowly shifts, every part "
     + "comes out bent a little differently, and the joint tilts: the top of it moves while "
     + "the bottom stays put."],
    ["The fixture", "var(--series-3)",
     "The jig that clamps the part down for welding. If it works loose or gets knocked, the "
     + "part sits in the wrong place, and the whole joint slides over together, top and "
     + "bottom by the same amount."],
    ["The camera mount", "var(--series-4)",
     "Nothing is wrong with the part at all. The camera has crept on its post, so the "
     + "measurement is wrong. This looks identical to a loose fixture, which is why the cell "
     + "has a fixed reference marker to tell them apart."],
  ];
  host.innerHTML = `
    <div class="hint" style="margin:18px 0 10px">The three machines that can put a weld in
      the wrong place. Each fails in its own way and each is a different walk across the
      shop, which is why naming the right one is worth a model call.</div>
    <div class="machines">${rows.map(([name, colour, what]) => `
      <div class="machine">
        <span class="n"><i style="background:${colour}"></i>${V.esc(name)}</span>
        <p>${V.esc(what)}</p>
      </div>`).join("")}</div>`;
}

/** Three jobs, three winners, and the loser that makes each point.

  Reading this dashboard used to mean opening four pages and holding the numbers
  in your head. The comparison that actually shaped the architecture is small
  enough to sit on the summary.
*/
function models(ev) {
  const host = V.$("ov-models");
  if (!host) return;
  const n = V.$("ov-models-n");
  if (n && ev.models && ev.models.available) n.textContent = `${ev.models.n} models were`;
  const r = ev.router, v = ev.vision, h = ev.hard;
  const blocks = [];

  if (r && r.available) {
    const order = ["nano", "ultra", "lightning", "super"];
    const rows = order.map((k) => r.arms.find((a) => a.arm === k)).filter(Boolean);
    blocks.push(["Per part: is this measurement good enough to weld on?",
      "Asked fifty times a run, so it goes to the cheapest model that is right.",
      ["Model", "Correct", "Wasted robot moves"],
      rows.map((a, i) => [
        a.arm === "nano" ? "Nemotron 3 Nano 30B" : a.arm === "ultra" ? "Nemotron 3 Ultra 550B"
          : a.arm === "lightning" ? "Nemotron 3.5 Lightning" : "Nemotron 3 Super 120B",
        `${a.correct}/${a.n}`, String(a.n - a.correct), i === 0])]);
  }
  if (v && v.available && v.arms.length > 1) {
    // Every model that will take an image, on the same fifty beads. Judged is
    // how many beads it gave a verdict on at all; the rest it declined, and a
    // declined bead is a part a person has to look at.
    const f = ev.vision_field;
    const seen = f && f.available ? `${f.counts.judged + f.counts.declined} of the ${f.n} models on the account will take a picture. ` : "";
    const looking = v.arms.filter((a) => a.judged === a.n && a.defects_named > 0).length;
    blocks.push(["After welding: is this bead good?",
      seen + `Under the prompt the cell uses, ${looking} of ${v.arms.length} judge every bead and `
      + "name defects. The one in bold is the one the cell uses.",
      ["Model", "Beads it judged", "Named right", "Bad parts shipped"],
      v.arms.map((a) => [a.model, `${a.judged}/${a.n}`, `${a.named_correct}/${a.judged}`,
        String(a.missed), a.model === v.cell_model])]);
  }
  if (h && h.available) {
    const rule = h.arms.find((a) => a.label.includes("arithmetic"));
    const top = h.arms.filter((a) => a !== rule).slice(0, 3);
    const worst = h.arms[h.arms.length - 1];
    blocks.push(["Once a shift: which machine slipped?",
      "Hard cases only, against a fixed rule with no model in it.",
      ["Model", "Correct", "Beats the rule at"],
      [...top.map((a, i) => [a.label, `${a.correct}/${a.n}`,
        a.vs_rule ? `p ${a.vs_rule.p < 0.05 ? a.vs_rule.p.toFixed(3) : a.vs_rule.p.toFixed(2)}` : "", i === 0]),
       [rule.label, `${rule.correct}/${rule.n}`, "the baseline", false],
       [worst.label, `${worst.correct}/${worst.n}`, "loses to it", false]]]);
  }

  host.innerHTML = `<div class="jobs">${blocks.map(([title, note, head, rows]) => `
    <div class="job">
      <div class="jt">${V.esc(title)}</div>
      <div class="jn">${V.esc(note)}</div>
      <table><thead><tr>${head.map((h2, i) =>
        `<th${i ? ' class="num"' : ""}>${V.esc(h2)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => `<tr${row[row.length - 1] ? ' class="won"' : ""}>
        ${row.slice(0, -1).map((v, i) => `<td${i ? ' class="num"' : ""}>${V.esc(v)}</td>`).join("")}
      </tr>`).join("")}</tbody></table>
    </div>`).join("")}</div>`;
}

/** What is actually new here, as opposed to what merely works. */
function novelty(ev) {
  const host = V.$("ov-new");
  if (!host) return;
  const h = ev.hard, c = ev.confound, b = ev.error_budget, r = ev.router;
  const items = [];
  if (h && h.available) {
    const rule = h.arms.find((a) => a.label.includes("arithmetic"));
    items.push(["A test a fixed rule cannot pass",
      "Evaluations usually leave out the trivial baseline. When one was written here, four "
      + "lines of arithmetic scored full marks on the original drift test. So a harder set was "
      + "built, on cases that sit right on the decision boundary.",
      `best model ${V.pct(h.arms[0].rate)} against ${V.pct(rule.rate)} for the rule`]);
  }
  if (c && c.available) {
    items.push(["A blind spot fixed with an instrument, not a bigger model",
      "A loose fixture moves the part. A creeping camera moves the measurement. In the weld "
      + "seam they produce identical numbers, so no model can separate them. One reference "
      + "marker bolted to the machine base can, because only the camera can appear to move it.",
      `drifting cameras found: ${c.camera_without.correct}/${c.camera_without.n} `
      + `to ${c.camera_with.correct}/${c.camera_with.n}`]);
  }
  if (r && r.available) {
    items.push(["Routing by how often a question is asked",
      "Not a ladder that escalates when unsure, which pays for the biggest model on exactly "
      + "the slowest cases. The per-part question goes to the smallest model because it "
      + "measured best on it. The once-a-shift question can afford a large one.",
      "44% fewer robot moves, half the model time"]);
  }
  items.push(["A model small enough to run on the cell itself",
    "The simulator knows which fault it injected, so runs with a known answer cost nothing to "
    + "make and arrive already labelled. Eight hundred of them trained a 1.7B model that now "
    + "answers beside the cell with no network at all.",
    "answers in 4 seconds, offline"]);
  if (b && b.available) {
    const real = b.budgets.find((x) => x.label.startsWith("Typical stereo"));
    const best = b.budgets.find((x) => x.label.startsWith("Laser"));
    items.push(["The whole chain, not just the camera",
      "Most demonstrations quote the measurement and stop. Add hand-eye calibration, the "
      + "robot's own repeatability, standoff drift and what the arc does to the metal, and the "
      + "answer changes which sensor a shop would have to buy.",
      `${V.fmt(real.total_mm, 2)} mm on a stereo camera, `
      + `${V.fmt(best.total_mm, 2)} mm on a laser profiler, against `
      + `${b.industry_requirement_mm} mm required`]);
  }
  host.innerHTML = `<div class="novel">${items.map(([t, d, f]) => `
    <div class="nv"><span class="t">${V.esc(t)}</span>
      <p>${V.esc(d)}</p><span class="f">${V.esc(f)}</span></div>`).join("")}</div>`;
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
    sub: `mean over ${s.stations_measured.toLocaleString()} measured points, ${s.n_parts} parts`,
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
    label: "Parts under 1 mm, every point",
    value: `${s.tracked.sub_mm_parts}/${s.n_parts}`,
    tone: s.tracked.sub_mm_parts / s.n_parts > 0.8 ? "good" : "warn",
    sub: `worst single point ${V.fmt(s.tracked.worst_station_mm, 2)} mm`,
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
         + `boundary. Only ${sig} models beat the rule by a margin that survives a `
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
        <span class="k">(worst point ${V.fmt(p.max_mm, 2)} mm)</span><br>
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
    xTitle: "mean point error, mm",
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
    ["Seam tracking", `${ev.seam.n_parts} parts × ${ev.seam.stations_per_part} points`,
     "the seam's true position, known from the part it was folded into",
     `${V.fmt(ev.seam.taught.mean_mm, 1)} mm → ${V.fmt(ev.seam.tracked.mean_mm, 3)} mm`, "good", "#/seam"],
    ["Joint locating", `${ev.locating.n_parts} parts, blind / one look / routed`,
     "the corner's true coordinates in the cell",
     `${V.fmt(ev.locating.blind.mean_mm, 1)} mm → ${V.fmt(ev.locating.routed.mean_mm, 2)} mm`, "good", "#/seam"],
    ["Look-again routing", ev.router.available ? `${ev.router.arms[0].n} decisions × 5 models` : "–",
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
    rows.push(["Camera against fixture", "4 causes \u00d7 4 runs, with and without the marker",
      "the injected fault, with and without a reference feature",
      `${ev.confound.arms[0].correct}/${ev.confound.arms[0].n} \u2192 `
      + `${ev.confound.arms[1].correct}/${ev.confound.arms[1].n} with the instrument`,
      "good", "#/budget"]);
  }
  const trainedRun = (ev.drift.edge || []).find((r) => r.tier === "edge");
  const untrainedRun = (ev.drift.edge || []).find((r) => r.tier === "edge_base");
  if (trainedRun) {
    rows.push(["Trained diagnosis, at the edge",
      `600 simulated runs, ${trainedRun.n} never trained on`,
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
      `${ev.hard.arms.length} contenders × ${ev.hard.cases} cases, passes pooled`,
      "cases sampled onto the decision boundary, where a fixed rule fails",
      `best ${V.pct(best.rate)}, a fixed rule ${rule ? V.pct(rule.rate) : "–"}`
      + (edge ? `, the edge model ${V.pct(edge.rate)}` : "")
      + (best.vs_rule ? ` (p ${best.vs_rule.p.toFixed(3)})` : ""),
      "bad", "#/diagnosis"]);
  }
  if (ev.vision && ev.vision.available && ev.vision.arms.length > 1) {
    const cell = ev.vision.arms.find((a) => a.model === ev.vision.cell_model) || ev.vision.arms[0];
    rows.push(["Every model that will take a picture",
      `${ev.vision.arms.length} models × ${ev.vision.arms[0].n} beads`,
      "the injected defect, scored the same way for every model",
      `only ${cell.model} judged every bead and named defects; the rest decline or never say defect`,
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
