/* Where the perception stops working, and what moving that edge cost.
 *
 * Every other page reports a number measured on exact rendered depth. This is the
 * only one that asks what those numbers are worth, by degrading the frame the way
 * a real sensor does until the method fails.
 */

import * as V from "../viz.js";
import { loadEvidence } from "../store.js";

export const meta = {
  id: "limits",
  title: "Sensor limits",
  crumb: "Limits",
  group: "What it leaves out",
  tag: "28",
  icon: "M12 3v4M12 17v4M3 12h4M17 12h4M6.3 6.3l2.8 2.8M14.9 14.9l2.8 2.8M17.7 6.3l-2.8 2.8M9.1 14.9l-2.8 2.8",
};

const AXIS_TITLE = {
  axial: "Range noise",
  dropout: "Shiny surface dropout",
  flying: "Flying pixels at the joint edge",
};

const AXIS_UNIT = {
  axial: (v) => `${v} mm`,
  dropout: (v) => `${Math.round(v * 100)}%`,
  flying: (v) => `${Math.round(v * 100)}%`,
};

const AXIS_WHAT = {
  axial: "How far a reading sits from the truth at one metre. Triangulation error grows "
       + "with the square of distance, so a camera quoted at a single millimetre figure "
       + "with no range beside it has told you nothing.",
  dropout: "How much of the frame comes back empty. Folded sheet steel is shiny and "
         + "returns nothing from a face angled away, in patches rather than speckle, "
         + "because a whole region shares an orientation.",
  flying: "Pixels straddling a depth step report a range belonging to neither surface. "
        + "This matters here more than anywhere else, because the step IS the joint.",
};

export function render() {
  return `
    <p class="lede">
      Every seam number on this dashboard was measured on perfect depth data. That
      makes them best cases rather than results, because a joint finder tested on perfect
      input has not really been tested. So the fifty-part run was repeated at twenty-eight
      levels of sensor degradation, to find where it breaks. The pass mark was fixed before
      the sweep started: the welding wire is 1.0 mm across, so a path further than that
      from the joint misses it.
    </p>

    <div class="tiles" id="lm-tiles"></div>

    <div class="grid">
      <div class="card w12">
        <header><h2>Each failure mode, taken until it breaks</h2></header>
        <div class="hint">One axis at a time, everything else clean, so each curve shows what
          that one failure costs rather than a mixture. The pale line is the worst point on any part. It degrades far faster than the mean, and it is the one that burns through.</div>
        <div id="lm-curves"></div>
        <div class="legend">
          <span><i class="line" style="background:var(--after)"></i>mean error, faces fitted from all their points</span>
          <span><i class="line" style="background:var(--before)"></i>mean error, slices alone</span>
          <span><i class="line" style="background:var(--warning)"></i>worst point on any part</span>
          <span><i style="background:var(--critical)"></i>past the wire diameter</span>
        </div>
      </div>

      <div class="card w7">
        <header><h2>What the sweep says the sensor has to be</h2></header>
        <div id="lm-limits"></div>
        <footer id="lm-verdict"></footer>
      </div>

      <div class="card w5">
        <header><h2>Real sensors, every failure mode at once</h2></header>
        <div class="hint">The curves above change one thing at a time. These are the combinations a real camera actually produces.</div>
        <div id="lm-named"></div>
      </div>
    </div>`;
}

let ev = null;

export async function mount() {
  ev = await loadEvidence();
  const r = ev.robustness;
  if (!r || !r.available) {
    V.$("lm-curves").innerHTML = `<div class="empty">The sweep has not been run.</div>`;
    return;
  }
  tiles(r);
  curves(r);
  limits(r);
  named(r);
}

function arm(r, key) { return r.arms.find((a) => a.arm === key); }

function tiles(r) {
  const best = arm(r, "refined") || arm(r, "slices");
  const base = arm(r, "slices");
  const axial = best.limits.axial || {};
  const dropout = best.limits.dropout || {};
  const flying = best.limits.flying || {};

  const stereo = (best.named || []).find((n) => n.value === "typical_stereo");
  const baseStereo = (base?.named || []).find((n) => n.value === "typical_stereo");

  V.$("lm-tiles").innerHTML = [
    tile("Range accuracy needed", `${axial.holds_to ?? "–"}`, "warn",
      `mm at a metre. It fails at ${axial.fails_at ?? "–"} mm, and this is the binding limit`, " mm"),
    tile("Frame it can lose", `${Math.round((dropout.holds_to ?? 0) * 100)}%`, "good",
      `and still weld. Missing data is not what kills this`),
    tile("Flying pixels tolerated", `${Math.round((flying.holds_to ?? 0) * 100)}%`, "good",
      flying.fails_at == null ? "weldable across the whole sweep" : `fails at ${flying.fails_at}`),
    stereo ? tile("On a typical stereo camera", V.fmt(stereo.mean_mm, 2), "bad",
      baseStereo && baseStereo.mean_mm !== stereo.mean_mm
        ? `mm, improved from ${V.fmt(baseStereo.mean_mm, 2)} mm and still past the wire`
        : "mm, which is past the wire diameter", " mm") : "",
    tile("Sensor levels run", "28", "", "each one the whole fifty-part campaign, twice over"),
  ].filter(Boolean).join("");
}

const tile = (label, value, tone, sub, unit = "") => `<div class="tile">
  <div class="label">${V.esc(label)}</div>
  <div class="value ${tone}">${value}${unit ? `<span class="unit">${unit}</span>` : ""}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

function curves(r) {
  const refined = arm(r, "refined");
  const slices = arm(r, "slices");
  const source = refined || slices;
  const axes = Object.keys(source.axes);

  const html = axes.map((axis) => {
    const a = source.axes[axis];
    const b = refined && slices ? slices.axes[axis] : null;
    return `<div class="w4" style="min-width:0">
      <div class="eyebrow">${V.esc(AXIS_TITLE[axis] || axis)}</div>
      ${curve(axis, a, b)}
      <div class="hint" style="margin:10px 0 0">${V.esc(AXIS_WHAT[axis] || "")}</div>
    </div>`;
  }).join("");
  V.paint(V.$("lm-curves"), `<div class="grid">${html}</div>`);
}

function curve(axis, primary, secondary) {
  const w = 360, h = 230, padL = 44, padR = 14, padT = 20, padB = 34;
  const values = primary.map((r) => r.value);
  const lo = Math.min(...values), hi = Math.max(...values);
  const xOf = (v) => padL + ((v - lo) / Math.max(1e-9, hi - lo)) * (w - padL - padR);

  // Log on the error axis: the mean and the worst point are an order apart and
  // a linear scale shows one of them as a flat line along the floor.
  const errs = [
    ...primary.map((r) => r.mean_mm), ...primary.map((r) => r.worst_mm),
    ...(secondary || []).map((r) => r.mean_mm),
  ].filter((v) => v != null && v > 0);
  const eLo = Math.min(0.4, Math.min(...errs)), eHi = Math.max(...errs) * 1.2;
  const yOf = (v) => {
    const t = (Math.log10(Math.max(v, eLo)) - Math.log10(eLo)) /
              (Math.log10(eHi) - Math.log10(eLo));
    return h - padB - t * (h - padB - padT);
  };

  let out = "";
  for (const tick of [0.5, 1, 2, 5, 10, 20]) {
    if (tick < eLo || tick > eHi) continue;
    out += `<line class="gridline" x1="${padL}" y1="${yOf(tick)}" x2="${w - padR}" y2="${yOf(tick)}"/>
      <text class="axis-label" x="${padL - 7}" y="${yOf(tick) + 3.5}" text-anchor="end">${tick}</text>`;
  }
  // The declared threshold.
  out += `<line x1="${padL}" y1="${yOf(1)}" x2="${w - padR}" y2="${yOf(1)}"
    stroke="var(--critical)" stroke-width="1.5" stroke-dasharray="4 3" opacity=".8"/>
    <text class="axis-label" x="${w - padR}" y="${yOf(1) - 6}" text-anchor="end"
      fill="var(--critical)">wire ø 1.0 mm</text>`;

  const line = (rows, key, colour, width, dash = "") => {
    const pts = rows.filter((r) => r[key] != null)
      .map((r) => `${xOf(r.value).toFixed(1)},${yOf(r[key]).toFixed(1)}`).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${colour}" stroke-width="${width}"
      stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`;
  };

  out += line(primary, "worst_mm", "var(--warning)", 1.5, "3 3");
  if (secondary) out += line(secondary, "mean_mm", "var(--before)", 1.8);
  out += line(primary, "mean_mm", secondary ? "var(--after)" : "var(--series-1)", 2.2);

  primary.forEach((row) => {
    if (row.mean_mm == null) return;
    const colour = row.weldable ? "var(--after)" : "var(--critical)";
    out += V.dot(xOf(row.value), yOf(row.mean_mm), 3.2, colour,
      `data-tip="<b>${AXIS_UNIT[axis] ? AXIS_UNIT[axis](row.value) : row.value}</b><br>
        <span class='k'>mean</span> <b>${V.fmt(row.mean_mm, 3)} mm</b><br>
        <span class='k'>worst point</span> <b>${V.fmt(row.worst_mm, 2)} mm</b><br>
        <span class='k'>parts tracked</span> <b>${row.parts_tracked}</b><br>
        <span class='k'>points returned</span> <b>${V.pct(row.station_yield, 0)}</b>"`);
  });

  [lo, (lo + hi) / 2, hi].forEach((v) => {
    out += `<text class="axis-label" x="${xOf(v)}" y="${h - padB + 15}" text-anchor="middle">${
      AXIS_UNIT[axis] ? AXIS_UNIT[axis](Number(v.toFixed(2))) : v}</text>`;
  });
  out += V.axisTitle(padL - 38, padT - 4, "mm");
  return V.svg(w, h, out);
}

function limits(r) {
  const best = arm(r, "refined") || arm(r, "slices");
  const rows = Object.entries(best.limits).map(([axis, lim]) => `<tr>
    <td><b>${V.esc(AXIS_TITLE[axis] || axis)}</b></td>
    <td class="mono">${lim.holds_to == null ? "–"
      : (AXIS_UNIT[axis] ? AXIS_UNIT[axis](lim.holds_to) : lim.holds_to)}</td>
    <td class="mono">${lim.fails_at == null ? "never, across the sweep"
      : (AXIS_UNIT[axis] ? AXIS_UNIT[axis](lim.fails_at) : lim.fails_at)}</td>
    <td class="muted">${V.esc(lim.fails_by || "–")}</td>
    <td class="num">${V.fmt(lim.worst_mm_at_best, 2)}</td>
  </tr>`).join("");

  V.$("lm-limits").innerHTML = `<table>
    <thead><tr><th>Failure mode</th><th>Holds to</th><th>Fails at</th>
      <th>Fails by</th><th class="num">Worst point when clean</th></tr></thead>
    <tbody>${rows}</tbody></table>`;

  const axial = best.limits.axial || {};
  V.$("lm-verdict").innerHTML =
    `The binding constraint is range accuracy, and it is tighter than it looks:
     <b>${axial.holds_to} mm</b> at a metre. A general purpose stereo depth camera is
     around 2.5 mm at that range, so this method does not run on one. Production seam trackers use laser line triangulation at a tenth of a millimetre. Losing most of the frame barely matters by comparison.`;
}

// The sensor presets are stored by their code names. A reader does not know
// what "specular" means, so the rows say what each one is in plain words and
// fall back to the code name only for a preset this map has not met.
const SENSOR_LABEL = {
  clean: "clean, no noise at all",
  typical_stereo: "typical stereo camera",
  harsh_specular: "harsh: shiny sheet, worst case",
};
const sensorName = (v) => SENSOR_LABEL[v] ?? String(v).replace(/_/g, " ");

function named(r) {
  const refined = arm(r, "refined");
  const slices = arm(r, "slices");
  const source = refined || slices;
  if (!source.named?.length) return;

  const rows = source.named.map((n) => {
    const before = (slices?.named || []).find((x) => x.value === n.value);
    return {
      label: sensorName(n.value),
      value: n.mean_mm ?? 0,
      display: `${V.fmt(n.mean_mm, 2)} mm`,
      colour: n.weldable ? "var(--after)" : "var(--critical)",
      tip: `<b>${sensorName(n.value)}</b><br>
        <span class='k'>mean</span> <b>${V.fmt(n.mean_mm, 3)} mm</b><br>
        <span class='k'>worst point</span> <b>${V.fmt(n.worst_mm, 2)} mm</b><br>
        <span class='k'>points returned</span> <b>${V.pct(n.station_yield, 0)}</b>
        ${before && refined && before.mean_mm !== n.mean_mm
          ? `<br><span class='k'>slices alone gave</span> <b>${V.fmt(before.mean_mm, 2)} mm</b>` : ""}`,
    };
  });
  V.paint(V.$("lm-named"), V.ranked(rows, { width: 380, rowH: 40, padL: 128 }));
}
