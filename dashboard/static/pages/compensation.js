/* Welding where the joint is going to be, not where it was. */

import * as V from "../viz.js";
import { loadEvidence } from "../store.js";

export const meta = {
  id: "compensation",
  title: "Thermal compensation",
  crumb: "Study 6",
  group: "Studies",
  icon: "M12 3v10m0 0a3 3 0 100 6 3 3 0 000-6ZM9 8h6",
};

export function render() {
  return `
    <p class="lede">
      The measurement is taken cold. The weld is not. Heat from the arc makes
      both folded sides rotate about their fold lines, so the top of the joint moves inward
      before the torch ever reaches it. A cell that measures a point and drives straight to
      it is aiming at where that point used to be. So the path is shifted in advance, by
      the amount of distortion expected at the moment the torch arrives at each point.
    </p>

    <div class="tiles" id="cp-tiles"></div>

    <div class="grid">
      <div class="card w7">
        <header><h2>How wrong the model can be and still pay</h2></header>
        <div class="hint">There is an obvious way to cheat this: predict the distortion with
          the same model that produces it, and the correction is exact by construction. So the
          cell compensates with nominal constants while the metal moves by perturbed ones,
          which is the situation a real machine is in.</div>
        <div id="cp-curve"></div>
        <div class="legend">
          <span><i class="line" style="background:var(--before)"></i>welding the cold measurement</span>
          <span><i class="line" style="background:var(--after)"></i>welding where the joint will be</span>
          <span><i class="line" style="background:var(--warning)"></i>worst station, compensated</span>
        </div>
        <footer id="cp-verdict"></footer>
      </div>

      <div class="card w5">
        <header><h2>Why it fails on one side and not the other</h2></header>
        <div id="cp-shape"></div>
      </div>

      <div class="card w12">
        <header><h2>Every level tested</h2></header>
        <div id="cp-table"></div>
      </div>
    </div>`;
}

const SHAPE = [
  ["Under-modelled", "The correction is too small, so some of the walk survives and the torch "
   + "lands behind the joint. The residual is bounded by the distortion itself, which is why "
   + "this side of the curve has no cliff: at worst you are back to not compensating."],
  ["Over-modelled", "The correction overshoots and pushes the torch past the joint, and now the "
   + "error is the correction rather than the distortion. Nothing bounds it. This is the side "
   + "with the edge on it, and it is where the method stops paying."],
  ["Which is why the gain is one", "Applying the model as calibrated, rather than scaling it up "
   + "for safety, sits on the side of the curve that fails gently. A cell that is unsure how hot "
   + "its own weld runs should under-correct on purpose."],
];

export async function mount() {
  const ev = await loadEvidence();
  const c = ev.compensation;
  if (!c || !c.available) {
    V.$("cp-curve").innerHTML = `<div class="empty">The study has not been run.</div>`;
    return;
  }
  tiles(c);
  curve(c);
  table(c);
  V.$("cp-shape").innerHTML = SHAPE.map(([a, b]) => `
    <div class="step" style="padding:13px 0 12px">
      <div class="t" style="margin-top:0">${V.esc(a)}</div>
      <div class="d">${V.esc(b)}</div>
    </div>`).join("");
}

function tiles(c) {
  const n = c.nominal;
  const cut = n.uncompensated_mm > 0
    ? 1 - n.compensated_mm / n.uncompensated_mm : 0;
  V.$("cp-tiles").innerHTML = [
    tile("Miss at nominal", V.fmt(n.compensated_mm, 2), "after",
      `mm, from ${V.fmt(n.uncompensated_mm, 2)} mm welding the cold measurement`, " mm"),
    tile("Of the walk removed", V.pct(cut), "good",
      "by aiming where the arc will have left the joint"),
    tile("Model error it survives", `${(c.helps_from * 100).toFixed(0)} to +${(c.helps_to * 100).toFixed(0)}%`,
      "good", "over which compensating still beats not compensating"),
    tile("Fails by", "over-predicting", "warn",
      "how far the metal moves. Under-predicting only leaves some walk behind"),
    tile("Trials per level", String(c.trials ?? "–"), "",
      "each with its own draw of the metal's constants"),
  ].join("");
}

const tile = (label, value, tone, sub, unit = "") => `<div class="tile">
  <div class="label">${V.esc(label)}</div>
  <div class="value ${tone}">${value}${unit ? `<span class="unit">${unit}</span>` : ""}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

function curve(c) {
  const rows = c.curve;
  const w = 560, h = 300, padL = 50, padR = 18, padT = 24, padB = 42;
  const xs = rows.map((r) => r.model_error);
  const lo = Math.min(...xs), hi = Math.max(...xs);
  const xOf = (v) => padL + ((v - lo) / (hi - lo)) * (w - padL - padR);

  const all = rows.flatMap((r) => [r.uncompensated_mm, r.compensated_mm, r.worst_compensated_mm]);
  const [top, ticks] = V.nice(Math.max(...all) * 1.1, 4);
  const yOf = (v) => h - padB - (v / top) * (h - padB - padT);

  let out = V.gridY(padL, w - padR, yOf, ticks, "", (v) => v.toFixed(1));
  out += `<line x1="${padL}" y1="${yOf(1)}" x2="${w - padR}" y2="${yOf(1)}"
    stroke="var(--critical)" stroke-width="1.5" stroke-dasharray="4 3" opacity=".75"/>
    <text class="axis-label" x="${w - padR}" y="${yOf(1) - 6}" text-anchor="end"
      fill="var(--critical)">wire ø 1.0 mm</text>`;
  // Where the cell's own model sits.
  out += `<line x1="${xOf(0)}" y1="${padT}" x2="${xOf(0)}" y2="${h - padB}"
    stroke="var(--line)" stroke-width="1.5"/>
    <text class="axis-label" x="${xOf(0)}" y="${padT - 8}" text-anchor="middle">model as calibrated</text>`;

  const line = (key, colour, width, dash = "") => {
    const pts = rows.map((r) => `${xOf(r.model_error).toFixed(1)},${yOf(r[key]).toFixed(1)}`).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${colour}" stroke-width="${width}"
      stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`;
  };
  out += line("worst_compensated_mm", "var(--warning)", 1.5, "3 3");
  out += line("uncompensated_mm", "var(--before)", 2);
  out += line("compensated_mm", "var(--after)", 2.4);

  rows.forEach((r) => {
    const colour = r.helped ? "var(--after)" : "var(--critical)";
    out += V.dot(xOf(r.model_error), yOf(r.compensated_mm), 3.6, colour,
      `data-tip="<b>${(r.model_error * 100).toFixed(0)}% model error</b><br>
        <span class='k'>welding the cold measurement</span> <b>${V.fmt(r.uncompensated_mm, 3)} mm</b><br>
        <span class='k'>compensated</span> <b>${V.fmt(r.compensated_mm, 3)} mm</b><br>
        <span class='k'>worst station</span> <b>${V.fmt(r.worst_compensated_mm, 2)} mm</b><br>
        ${r.helped ? "compensating helped here" : "<b>compensating made it worse here</b>"}"`);
    out += V.dot(xOf(r.model_error), yOf(r.uncompensated_mm), 2.6, "var(--before)");
  });

  [lo, 0, hi].forEach((v) => {
    out += `<text class="axis-label" x="${xOf(v)}" y="${h - padB + 15}" text-anchor="middle">${
      (v * 100).toFixed(0)}%</text>`;
  });
  out += V.axisTitle(padL, h - 8, "how wrong the distortion model is, against the metal's own constants");
  out += V.axisTitle(padL - 42, padT - 6, "mm");
  V.paint(V.$("cp-curve"), V.svg(w, h, out));

  const n = c.nominal;
  V.$("cp-verdict").innerHTML =
    `At nominal the miss drops from <b>${V.fmt(n.uncompensated_mm, 2)} mm</b> to
     <b>${V.fmt(n.compensated_mm, 2)} mm</b>. More usefully, compensating still beats not
     compensating from <b>${(c.helps_from * 100).toFixed(0)}%</b> to
     <b>+${(c.helps_to * 100).toFixed(0)}%</b> model error, which is a wide enough band that a
     calibrated model on a real machine would sit inside it. The curve is not symmetric, and
     that asymmetry is the practical result: the failure is on the over-modelled side.`;
}

function table(c) {
  V.$("cp-table").innerHTML = `<table>
    <thead><tr><th>Model error</th><th class="num">Welding the cold measurement</th>
      <th class="num">Welding where it will be</th><th class="num">Change</th>
      <th class="num">Worst station</th><th>Verdict</th></tr></thead>
    <tbody>${c.curve.map((r) => `<tr>
      <td class="mono">${(r.model_error * 100).toFixed(0)}%</td>
      <td class="num">${V.fmt(r.uncompensated_mm, 3)}</td>
      <td class="num">${V.fmt(r.compensated_mm, 3)}</td>
      <td class="num" style="color:${r.helped ? "var(--after)" : "var(--critical)"}">
        ${r.compensated_mm - r.uncompensated_mm > 0 ? "+" : ""}${V.fmt(r.compensated_mm - r.uncompensated_mm, 3)}</td>
      <td class="num">${V.fmt(r.worst_compensated_mm, 2)}</td>
      <td><span class="badge ${r.helped ? "good" : "bad"}">${
        r.helped ? "helps" : "overshoots"}</span></td>
    </tr>`).join("")}</tbody></table>`;
}
