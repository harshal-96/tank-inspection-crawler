/* The cell itself, drawn from the same constants the simulation runs on. */

import * as V from "../viz.js";
import { state, chosen, choose, loadCell, refreshLive } from "../store.js";

export const meta = {
  id: "cell",
  title: "The cell",
  crumb: "Live",
  group: "Cell",
  icon: "M4 20V8l8-5 8 5v12M9 20v-6h6v6",
};

let view = null;
let exaggeration = 1;

export function render() {
  return `
    <div class="stage">
      <canvas id="view"></canvas>
      <div class="overlay-tl" id="cell-chips"></div>
      <div class="overlay-tr">
        <button id="v-cell" class="on">Cell</button>
        <button id="v-seam">Seam</button>
        <button id="v-sensor">Sensor</button>
        <button id="v-spin">Spin</button>
        <button id="v-exag">Exaggerate ×1</button>
      </div>
      <div class="overlay-bl">drag to orbit · scroll to zoom</div>
      <div class="overlay-br">
        <div class="glass legend" style="margin:0">
          <span><i class="line" style="background:var(--before)"></i>taught path</span>
          <span><i class="line" style="background:var(--series-1)"></i>tracked seam</span>
          <span><i style="background:var(--good)"></i>station ok</span>
          <span><i style="background:var(--warning)"></i>needs attention</span>
        </div>
      </div>
    </div>

    <div class="grid">
      <div class="card w4">
        <header><h2>What the cell is made of</h2></header>
        <div id="cell-parts"></div>
      </div>
      <div class="card w4">
        <header><h2>The one look that starts the weld</h2></header>
        <div class="hint">The camera is fixed to its own post, not the arm: a camera that rides
          the wrist has to be recalibrated against every pose it takes, and its view is the first
          thing the torch gets in the way of.</div>
        <div id="cell-sensor"></div>
      </div>
      <div class="card w4">
        <header><h2>How far the corner moved</h2></header>
        <div id="cell-part"></div>
      </div>
    </div>`;
}

export async function mount() {
  const [geo] = await Promise.all([loadCell(), refreshLive()]);
  const { Cell3D } = await import("../cell3d.js");

  const canvas = V.$("view");
  if (!canvas) return;
  try {
    view = new Cell3D(canvas);
    view.build(geo);
  } catch (err) {
    canvas.insertAdjacentHTML("afterend",
      `<div class="empty">3D view unavailable: ${V.esc(err.message)}</div>`);
    view = null;
  }

  V.$("v-cell").onclick = () => { view?.frameCell(); active("v-cell"); };
  V.$("v-seam").onclick = () => { view?.frameSeam(); active("v-seam"); };
  V.$("v-sensor").onclick = () => { view?.frameSensor(); active("v-sensor"); };
  V.$("v-spin").onclick = () => {
    if (!view) return;
    view.spinning = !view.spinning;
    V.$("v-spin").classList.toggle("on", view.spinning);
  };
  V.$("v-exag").onclick = () => {
    exaggeration = exaggeration >= 40 ? 1 : (exaggeration === 1 ? 10 : exaggeration * 2);
    view?.setExaggeration(exaggeration);
    V.$("v-exag").textContent = `Exaggerate ×${exaggeration}`;
    V.$("v-exag").classList.toggle("on", exaggeration > 1);
    draw();
  };
  function active(id) {
    for (const b of ["v-cell", "v-seam", "v-sensor"]) V.$(b)?.classList.toggle("on", b === id);
  }

  fixed(geo);
  draw();
}

export function update() { draw(); }

/** The viewport is a lit scene, not CSS, so the theme switch has to reach it. */
export function retheme() { view?.retheme?.(); }

export function unmount() {
  view?.dispose?.();
  view = null;
  exaggeration = 1;
}

function draw() {
  const slot = chosen();
  if (!slot) return;
  view?.showSeam(slot.seam, slot.schedule);
  chips(slot);
  partCard(slot);
}

function chips(slot) {
  const t = slot.seam, s = slot.schedule;
  const cards = [];
  if (t) {
    cards.push(glass("Seam found to", V.fmt(t.error_mm, 2) + " mm",
      `worst station ${V.fmt(t.worst_station_mm, 2)} mm`, "var(--after)"));
    cards.push(glass("Taught path misses", V.fmt(t.taught_miss_mm, 1) + " mm",
      `${t.stations_measured}/${t.stations_asked} stations measured`, "var(--before)"));
  }
  if (s) {
    cards.push(glass("Weld schedule", `${s.weldable_stations}/${s.weldable_stations + s.refused_stations}`,
      `hottest ${V.fmt(s.heat_j_per_mm_max, 0)} J/mm`,
      s.refused_stations ? "var(--warning)" : "var(--good)"));
  }
  if (exaggeration > 1) {
    cards.push(glass("Display", `×${exaggeration}`, "exaggerated for visibility", "var(--warning)"));
  }
  V.$("cell-chips").innerHTML = cards.join("");
}

const glass = (label, value, sub, colour) => `<div class="glass">
  <div class="label">${V.esc(label)}</div>
  <div class="value" style="color:${colour}">${value}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

function fixed(geo) {
  const mm = (m) => (m * 1000).toFixed(0);
  V.$("cell-parts").innerHTML = `<table class="plain"><tbody>
    ${row("Tray", `${mm(geo.part.width_m)} × ${mm(geo.part.length_m)} mm, sides ${mm(geo.part.side_height_m)} mm`)}
    ${row("Sheet", `${geo.part.thickness_m * 1000} mm, folded from a notched blank`)}
    ${row("Seam", `the ${mm(geo.part.side_height_m)} mm corner where two folded sides meet`)}
    ${row("Fixture", `table top at ${mm(geo.fixture.table_top_m)} mm, two locating pads clear of the joint`)}
    ${row("Robot", "six axis arm, torch on the flange")}
    ${row("Torch", `${geo.torch.stickout_m * 1000} mm stickout at the wire tip`)}
  </tbody></table>`;

  const eye = geo.camera.eye_m, aim = geo.camera.aim_m;
  const standoff = Math.hypot(eye[0] - aim[0], eye[1] - aim[1], eye[2] - aim[2]);
  V.$("cell-sensor").innerHTML = `<table class="plain"><tbody>
    ${row("Mount", "fixed post, off the arm")}
    ${row("Standoff", `${(standoff * 1000).toFixed(0)} mm to the corner`)}
    ${row("Looks at", `x ${aim[0].toFixed(3)} m, y ${aim[1].toFixed(3)} m`)}
    ${row("Returns", "colour, depth, surface normals and a semantic mask")}
    ${row("Used for", "two plane fits, intersected to give the joint line")}
    ${row("Stations", "20 slices along the seam, each measured on its own")}
  </tbody></table>`;
}

const row = (k, v) => `<tr><td class="muted" style="width:88px">${V.esc(k)}</td><td>${V.esc(v)}</td></tr>`;

function partCard(slot) {
  const t = slot.seam, m = slot.measurement, d = slot.decision;
  if (!t) { V.$("cell-part").innerHTML = `<div class="empty">No seam for this part.</div>`; return; }
  const dev = deviations(t.path || []);
  const w = 300, h = 120, pad = 18;
  const span = Math.max(1.0, Math.max(...dev.map(Math.abs)) * 1.5);
  const xOf = (i) => pad + (i / Math.max(1, dev.length - 1)) * (w - pad * 2);
  const yOf = (v) => h / 2 - (v / span) * (h / 2 - 12);
  const pts = dev.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const chart = V.svg(w, h, `
    <line class="gridline" x1="${pad}" y1="${h / 2}" x2="${w - pad}" y2="${h / 2}"/>
    ${V.line(pts, "var(--series-1)", 2)}
    ${dev.map((v, i) => V.dot(xOf(i), yOf(v), 2.6, "var(--series-1)",
      `data-tip="station ${i + 1}: <b>${V.fmt(v, 2)} mm</b> off the straight line"`)).join("")}
    ${V.axisTitle(pad, 12, `±${span.toFixed(1)} mm about the seam's own chord`)}`);

  V.$("cell-part").innerHTML = `
    ${chart}
    <table class="plain" style="margin-top:10px"><tbody>
      ${row("Part", `#${t.part_id}`)}
      ${row("Bow", `${V.fmt(t.seam_bow_mm, 2)} mm across the ${t.stations_measured} stations`)}
      ${row("Gap", `mean ${V.fmt(t.gap_mm_mean, 2)} mm, worst ${V.fmt(t.gap_mm_max, 2)} mm`)}
      ${m ? row("Corner", `x ${m.detected_x.toFixed(4)} m, y ${m.detected_y.toFixed(4)} m`) : ""}
      ${d ? row("Looks", `${d.after_looks}, decided by ${(d.model || "").split("/").pop()}`) : ""}
    </tbody></table>`;
  V.bindTips(V.$("cell-part"));
}

/** How far each station sits off the straight line between the seam's own ends. */
export function deviations(path) {
  if (!path || path.length < 2) return (path || []).map(() => 0);
  const a = path[0], b = path[path.length - 1];
  const dx = b.x_mm - a.x_mm, dy = b.y_mm - a.y_mm;
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return path.map(() => 0);
  const ux = dx / len, uy = dy / len;
  return path.map((pt) => {
    const rx = pt.x_mm - a.x_mm, ry = pt.y_mm - a.y_mm;
    const along = rx * ux + ry * uy;
    return (rx - along * ux) * uy - (ry - along * uy) * ux;
  });
}
