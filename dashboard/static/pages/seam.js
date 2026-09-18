/* The seam study: fifty parts, twenty points each, and what happened at each one. */

import * as V from "../viz.js";
import { loadEvidence, refreshLive, state, chosen, choose } from "../store.js";
import { deviations } from "./cell.js";

export const meta = {
  id: "seam",
  title: "Seam tracking",
  crumb: "Stage 1",
  group: "What happens to a part",
  tag: "1k",
  icon: "M3 17c4-10 14 4 18-6",
};

export function render() {
  return `
    <p class="lede">
      The joint is the line where two folded panels meet. It is not where the program
      says it is, and it is not straight. The cell measures it at twenty points along its
      length, all from a single camera view, and each point is compared against where the
      joint truly is.
    </p>

    <div class="grid">
      <div class="tiles" id="sm-tiles"></div>

      <div class="card w8">
        <header><h2>Point error, part by part</h2></header>
        <div class="hint">Each column is a part; each mark is one of its twenty points. The dark
          band is where the middle half of that part's points landed.</div>
        <div id="sm-strip"></div>
        <div class="legend">
          <span><i class="line" style="background:var(--series-1)"></i>the part's mean</span>
          <span><i style="background:var(--warning)"></i>a point over 1 mm</span>
        </div>
      </div>

      <div class="card w4">
        <header><h2>What the correction cannot remove</h2></header>
        <div class="hint">Correcting the path removes the offset, but it cannot remove the bow, because
          the seam itself is curved. That bow is the limit on how accurate this method
          can get.</div>
        <div id="sm-decomp"></div>
        <footer id="sm-decomp-meta"></footer>
      </div>

      <div class="card w12">
        <header>
          <h2>Point detail</h2>
          <span class="grow"></span>
          <select id="sm-part"></select>
        </header>
        <div class="grid" style="margin-top:6px">
          <div class="w4" style="min-width:0">
            <div id="sm-path"></div>
          </div>
          <div class="w4" style="min-width:0">
            <div id="sm-gap"></div>
          </div>
          <div class="w4" style="min-width:0">
            <div id="sm-heat"></div>
          </div>
        </div>
        <footer id="sm-detail-meta"></footer>
      </div>

      <div class="card w6">
        <header><h2>Where the joint opens too far to weld</h2></header>
        <div class="hint">A gap past 1.6 times the sheet cannot be bridged in one pass without
          backing. The cell refuses those points rather than laying a bead it cannot hold.</div>
        <div id="sm-refused"></div>
        <footer id="sm-refused-meta"></footer>
      </div>

      <div class="card w6">
        <header><h2>Blind, one fixed look, and look-then-decide</h2></header>
        <div class="hint">The same fifty parts under three policies, scored against the corner's
          true coordinates.</div>
        <div id="sm-locating"></div>
        <footer id="sm-locating-meta"></footer>
      </div>
    </div>`;
}

let ev = null;

export async function mount() {
  [ev] = await Promise.all([loadEvidence(), refreshLive()]);
  tiles();
  strip();
  decomp();
  refused();
  locating();
  picker();
  detail();
}

export function update() { picker(); detail(); }

/* --------------------------------------------------------------------- charts */

function tiles() {
  const s = ev.seam;
  if (!s.available) return;
  const perception = s.perception_ms || {};
  V.$("sm-tiles").innerHTML = [
    tile("Tracked to", V.fmt(s.tracked.mean_mm, 3), "after",
      `mean over ${s.stations_measured.toLocaleString()} points`, "mm"),
    tile("A taught path misses by", V.fmt(s.taught.mean_mm, 1), "before",
      `worst part ${V.fmt(s.taught.max_mm, 1)} mm`, "mm"),
    tile("Parts under 1 mm, every point", `${s.tracked.sub_mm_parts}/${s.n_parts}`, "good",
      `worst single point ${V.fmt(s.tracked.worst_station_mm, 2)} mm`),
    tile("The seam's own bow", V.fmt(s.bow.mean_mm, 2), "",
      `worst ${V.fmt(s.bow.max_mm, 2)} mm, and no straight path removes it`, "mm"),
    perception.median
      ? tile("Depth frame to weld path", V.fmt(perception.median, 0), "",
          `median, worst ${V.fmt(perception.max, 0)} ms, on the cell's own CPU`, "ms")
      : "",
    tile("Points refused", String(s.refused_stations), s.refused_stations ? "warn" : "good",
      `of ${s.stations_measured.toLocaleString()}, where the joint was too open to bridge`),
  ].filter(Boolean).join("");
}

const tile = (label, value, tone, sub, unit = "") => `<div class="tile">
  <div class="label">${V.esc(label)}</div>
  <div class="value ${tone}">${value}${unit ? `<span class="unit">${unit}</span>` : ""}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

function strip() {
  if (!ev.seam.available) return;
  const live = state.live;
  const parts = [...ev.seam.parts].sort((a, b) => a.trial - b.trial);
  const w = 720, h = 260, padL = 40, padB = 28, padT = 16, padR = 8;
  const hiRaw = Math.max(...parts.map((p) => p.max_mm));
  const [top, ticks] = V.nice(hiRaw * 1.05, 4);
  const yOf = (v) => h - padB - (v / top) * (h - padB - padT);
  const cw = (w - padL - padR) / parts.length;

  let out = V.gridY(padL, w - padR, yOf, ticks, "", (v) => v.toFixed(1));
  out += `<line class="gridline" x1="${padL}" y1="${yOf(1)}" x2="${w - padR}" y2="${yOf(1)}"
    stroke="var(--warning)" stroke-dasharray="3 3" opacity=".6"/>`;

  parts.forEach((p, i) => {
    const x = padL + i * cw + cw / 2;
    // Per-point errors live on the live seam rows; fall back to the summary box.
    const track = live?.seams.find((s) => s.part_id === p.trial + 1);
    const devs = track ? deviations(track.path || []).map(Math.abs) : null;
    const lo = p.mean_mm - (p.max_mm - p.mean_mm) * 0.6;
    out += `<line x1="${x}" y1="${yOf(Math.max(0, lo))}" x2="${x}" y2="${yOf(p.max_mm)}"
      stroke="var(--band)" stroke-width="${Math.max(2.5, cw * 0.52)}" stroke-linecap="round"/>`;
    if (devs && devs.length) {
      devs.forEach((d) => {
        out += `<circle cx="${x}" cy="${yOf(Math.min(d, top))}" r="1.5"
          fill="${d > 1 ? "var(--warning)" : "var(--text-muted)"}" opacity=".75"/>`;
      });
    }
    out += `<circle cx="${x}" cy="${yOf(p.mean_mm)}" r="${Math.min(3.1, Math.max(2, cw * 0.28))}"
      fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="1.2"
      data-tip="<b>part ${p.trial}</b><br><span class='k'>mean</span> <b>${V.fmt(p.mean_mm, 3)} mm</b><br>
        <span class='k'>worst point</span> <b>${V.fmt(p.max_mm, 2)} mm</b><br>
        <span class='k'>bow</span> <b>${V.fmt(p.bow_mm, 2)} mm</b><br>
        <span class='k'>taught path would miss</span> <b>${V.fmt(p.taught_miss_mm, 1)} mm</b>
        ${p.refused ? `<br><span class='k'>refused</span> <b>${p.refused}</b> <span class='k'>points</span>` : ""}"/>`;
  });
  out += V.axisTitle(padL, h - 5, "part number, in the order they were welded");
  out += V.axisTitle(padL - 30, padT - 4, "mm");
  V.paint(V.$("sm-strip"), V.svg(w, h, out));
}

/**
 * The residual is two percent of the miss, so three bars on one scale would be one
 * bar and two hairlines. This draws the miss at full scale and then magnifies what
 * is left, with the magnification stated, because the two halves are three orders
 * apart and one scale would hide the smaller one.
 */
function decomp() {
  const s = ev.seam;
  if (!s.available) return;
  const w = 420, h = 236, padL = 8, padR = 8;
  const track = w - padL - padR;
  const factor = Math.round(s.taught.mean_mm / s.tracked.mean_mm);
  const keptFrac = s.tracked.mean_mm / s.taught.mean_mm;

  let out = "";
  // The miss, full width.
  out += `<text class="axis-label" x="${padL}" y="14">a taught path misses by</text>`;
  out += `<text x="${w - padR}" y="15" text-anchor="end" font-size="15" font-family="var(--mono)"
    fill="var(--before)">${V.fmt(s.taught.mean_mm, 1)} mm</text>`;
  out += V.barH(padL, 24, track, 22, "var(--before)", 'opacity=".85"');
  const kept = Math.max(2, track * keptFrac);
  out += V.barH(padL, 24, kept, 22, "var(--after)",
    `data-tip="what is left after the correction: <b>${V.fmt(s.tracked.mean_mm, 3)} mm</b>"`);

  // The bracket down to the magnified view.
  out += `<path d="M${padL},${52} L${padL},${62} L${padL + kept},${62} L${padL + kept},${52}"
    fill="none" stroke="var(--line)" stroke-width="1"/>`;
  out += `<path d="M${padL + kept / 2},${62} L${padL + kept / 2},${70} L${padL},${70}
    L${padL},${84} M${padL + kept / 2},${70} L${w - padR},${70} L${w - padR},${84}"
    fill="none" stroke="var(--line)" stroke-width="1" stroke-dasharray="2 3"/>`;
  out += `<text class="axis-label" x="${w / 2}" y="${80}" text-anchor="middle">that remainder, magnified ×${factor}</text>`;

  // What the remainder is made of, at its own scale.
  const top = s.tracked.mean_mm * 1.12;
  const xOf = (v) => (v / top) * track;
  const rows = [
    { label: "the seam's own bow", value: s.bow.mean_mm, colour: "var(--series-1)",
      note: "no straight corrected line removes this",
      tip: `the corner is curved by <b>${V.fmt(s.bow.mean_mm, 2)} mm</b> on average, <b>${V.fmt(s.bow.max_mm, 2)} mm</b> at worst` },
    { label: "everything else", value: Math.max(0, s.tracked.mean_mm - s.bow.mean_mm),
      colour: "var(--after)", note: "measurement, fit and point spacing",
      tip: `what the measurement itself contributes: <b>${V.fmt(Math.max(0, s.tracked.mean_mm - s.bow.mean_mm), 2)} mm</b>` },
  ];
  rows.forEach((r, i) => {
    const y = 100 + i * 50;
    out += `<text x="${padL}" y="${y}" font-size="11.5" fill="var(--text-secondary)"
      font-family="var(--sans)">${V.esc(r.label)}</text>`;
    out += `<text x="${w - padR}" y="${y}" text-anchor="end" font-size="12" font-family="var(--mono)"
      fill="var(--text-primary)">${V.fmt(r.value, 2)} mm</text>`;
    out += V.barH(padL, y + 7, Math.max(2, xOf(r.value)), 12, r.colour,
      `opacity=".9" data-tip="${V.esc(r.tip)}"`);
    out += `<text x="${padL}" y="${y + 33}" font-size="10.5" fill="var(--text-muted)"
      font-family="var(--sans)">${V.esc(r.note)}</text>`;
  });
  V.paint(V.$("sm-decomp"), V.svg(w, h, out));

  V.$("sm-decomp-meta").innerHTML =
    `The correction removes <b>${(100 * (1 - keptFrac)).toFixed(1)}%</b> of the miss. Of what remains,
     roughly <b>${(100 * s.bow.mean_mm / s.tracked.mean_mm).toFixed(0)}%</b> is the seam being curved
     rather than the measurement being wrong: a corrected straight line still has to cut a corner that
     bows <b>${V.fmt(s.bow.max_mm, 2)} mm</b> at worst.`;
}

function refused() {
  const s = ev.seam;
  if (!s.available) return;
  const counts = new Map();
  s.parts.forEach((p) => counts.set(p.refused, (counts.get(p.refused) || 0) + 1));
  const keys = [...counts.keys()].sort((a, b) => a - b);
  const rows = keys.map((k) => ({
    label: k === 0 ? "no refusals" : `${k} point${k > 1 ? "s" : ""} refused`,
    value: counts.get(k),
    display: `${counts.get(k)} parts`,
    colour: k === 0 ? V.GOOD : k <= 1 ? V.WARN : V.BAD,
    tip: `<b>${counts.get(k)}</b> parts had <b>${k}</b> point${k === 1 ? "" : "s"} where the joint was too open to weld in one pass`,
  }));
  V.paint(V.$("sm-refused"), V.ranked(rows, { width: 400, rowH: 32, padL: 132 }));
  const clean = counts.get(0) || 0;
  V.$("sm-refused-meta").innerHTML =
    `<b>${s.refused_stations}</b> of <b>${s.stations_measured.toLocaleString()}</b> points refused across the run,
     on <b>${s.n_parts - clean}</b> parts. A refused point means the gap there is wider than one pass can bridge, so the fit-up needs correcting at the press.`;
}

function locating() {
  const l = ev.locating;
  if (!l.available) { V.$("sm-locating").innerHTML = `<div class="empty">Not run.</div>`; return; }
  const rows = [
    { label: "Blind, taught only", value: l.blind.mean_mm, colour: V.BEFORE,
      display: V.fmt(l.blind.mean_mm, 1) + " mm", tip: `${l.blind.sub_mm}/${l.n} under 1 mm` },
    { label: "One fixed look", value: l.one_look.mean_mm, colour: "var(--series-1)",
      display: V.fmt(l.one_look.mean_mm, 2) + " mm",
      tip: `<b>${l.one_look.sub_mm}</b>/${l.n_parts} parts under 1 mm, worst ${V.fmt(l.one_look.max_mm, 2)} mm` },
    { label: "Look then decide", value: l.routed.mean_mm, colour: "var(--series-5)",
      display: V.fmt(l.routed.mean_mm, 2) + " mm",
      tip: `<b>${l.routed.sub_mm}</b>/${l.n_parts} parts under 1 mm, worst ${V.fmt(l.routed.max_mm, 2)} mm` },
  ];
  V.paint(V.$("sm-locating"), V.ranked(rows, { width: 400, rowH: 32, padL: 132 }));
  V.$("sm-locating-meta").innerHTML =
    `The gap that matters is the first one: <b>${V.fmt(l.blind.mean_mm, 1)} mm</b> down to
     <b>${V.fmt(l.one_look.mean_mm, 2)} mm</b>, bought by looking once. Deciding whether to look again
     is worth <b>${V.fmt(Math.abs(l.one_look.mean_mm - l.routed.mean_mm), 3)} mm</b> on the mean:
     on this part mix the first look was almost always good enough.`;
}

/* --------------------------------------------------------------------- detail */

function picker() {
  const sel = V.$("sm-part");
  if (!sel || !state.live) return;
  const ids = state.live.ids;
  if (sel.options.length !== ids.length) {
    sel.innerHTML = ids.map((id) => `<option value="${id}">part #${id}</option>`).join("");
    sel.onchange = (e) => { choose(e.target.value); detail(); };
  }
  sel.value = String(state.chosenPart);
}

function detail() {
  const slot = chosen();
  if (!slot || !slot.seam) return;
  const t = slot.seam, sch = slot.schedule;
  const path = t.path || [];
  const dev = deviations(path);

  const W = 300, H = 250, padX = 42, padY = 18;
  const alongs = path.map((p) => p.along_mm);
  const loA = Math.min(...alongs), hiA = Math.max(...alongs);
  const yOf = (a) => H - padY - ((a - loA) / Math.max(1e-6, hiA - loA)) * (H - padY * 2);

  // --- the path against the straight line it was taught to follow
  const span = Math.max(1.0, Math.max(...dev.map(Math.abs)) * 1.7);
  const xOf = (d) => padX + ((d + span) / (2 * span)) * (W - padX - 14);
  let out = "";
  for (let mm = -Math.floor(span); mm <= Math.floor(span); mm++) {
    out += `<line class="gridline" x1="${xOf(mm)}" y1="${padY}" x2="${xOf(mm)}" y2="${H - padY}"/>
      <text class="axis-label" x="${xOf(mm)}" y="${H - 4}" text-anchor="middle">${mm}</text>`;
  }
  out += `<line x1="${xOf(0)}" y1="${padY}" x2="${xOf(0)}" y2="${H - padY}"
    stroke="var(--before)" stroke-width="2"/>`;
  out += V.line(path.map((p, i) => `${xOf(dev[i])},${yOf(p.along_mm)}`).join(" "), "var(--series-1)", 2);
  path.forEach((p, i) => {
    out += V.dot(xOf(dev[i]), yOf(p.along_mm), 3, "var(--series-1)",
      `data-tip="<b>${p.along_mm.toFixed(0)} mm</b> up the seam<br><span class='k'>off the straight line by</span> <b>${V.fmt(dev[i], 2)} mm</b>"`);
  });
  out += V.axisTitle(padX, 11, "mm off the straight line between the seam's ends");
  V.paint(V.$("sm-path"), V.svg(W, H, out));

  // --- how open the joint is at each point
  const stations = (sch && sch.stations) || [];
  out = "";
  if (stations.length) {
    const hiG = Math.max(1.0, Math.max(...stations.map((s) => s.gap_mm)) * 1.2);
    const xG = (g) => padX + (g / hiG) * (W - padX - 14);
    for (let mm = 0; mm <= Math.ceil(hiG); mm++) {
      out += `<line class="gridline" x1="${xG(mm)}" y1="${padY}" x2="${xG(mm)}" y2="${H - padY}"/>
        <text class="axis-label" x="${xG(mm)}" y="${H - 4}" text-anchor="middle">${mm}</text>`;
    }
    out += V.line(stations.map((s) => `${xG(s.gap_mm)},${yOf(s.along_mm)}`).join(" "),
      "var(--series-1)", 2, 'opacity=".55"');
    stations.forEach((s) => {
      const colour = !s.weldable ? V.BAD : s.needs_attention ? V.WARN : V.GOOD;
      out += V.dot(xG(s.gap_mm), yOf(s.along_mm), 3.4, colour,
        `data-tip="<b>${s.along_mm.toFixed(0)} mm</b> up the seam<br>
          <span class='k'>gap</span> <b>${V.fmt(s.gap_mm, 2)} mm</b><br>
          <span class='k'>travel</span> <b>${V.fmt(s.travel_mm_s, 1)} mm/s</b>
          <span class='k'>at</span> <b>${V.fmt(s.current_a, 0)} A</b><br>
          ${V.esc(s.reason || "")}"`);
    });
    out += V.axisTitle(padX, 11, "joint opening, mm");
  }
  V.paint(V.$("sm-gap"), V.svg(W, H, out));

  // --- what the schedule asks the arc for
  out = "";
  if (stations.length) {
    const hiH = Math.max(...stations.map((s) => s.heat_j_per_mm), 1) * 1.15;
    const xH = (v) => padX + (v / hiH) * (W - padX - 14);
    const bh = (H - padY * 2) / stations.length;
    [0, hiH / 2, hiH].forEach((v) => {
      out += `<line class="gridline" x1="${xH(v)}" y1="${padY}" x2="${xH(v)}" y2="${H - padY}"/>
        <text class="axis-label" x="${xH(v)}" y="${H - 4}" text-anchor="middle">${v.toFixed(0)}</text>`;
    });
    stations.forEach((s, i) => {
      const y = H - padY - (i + 1) * bh;
      const colour = !s.weldable ? V.BAD : s.needs_attention ? V.WARN : V.AFTER;
      out += V.barH(padX, y + 1, Math.max(1, xH(s.heat_j_per_mm)), Math.max(2, bh - 2), colour,
        `opacity=".9" data-tip="<b>${s.along_mm.toFixed(0)} mm</b> up the seam<br>
          <span class='k'>heat</span> <b>${V.fmt(s.heat_j_per_mm, 0)} J/mm</b><br>
          ${V.esc(s.reason || "")}"`);
    });
    out += V.axisTitle(padX, 11, "heat input, J/mm, bottom of the seam upward");
  }
  V.paint(V.$("sm-heat"), V.svg(W, H, out));

  const bits = [];
  bits.push(`Part <b>#${t.part_id}</b> tracked to <b>${V.fmt(t.error_mm, 2)} mm</b>`);
  bits.push(`a taught path would have missed by <b>${V.fmt(t.taught_miss_mm, 1)} mm</b>`);
  bits.push(`the seam bows <b>${V.fmt(t.seam_bow_mm, 2)} mm</b>`);
  if (sch) {
    bits.push(`<b>${sch.weldable_stations}</b> points weldable, <b>${sch.refused_stations}</b> refused`);
    bits.push(`travel <b>${V.fmt(sch.travel_mm_s_min, 1)}</b> mm/s upward, hottest <b>${V.fmt(sch.heat_j_per_mm_max, 0)}</b> J/mm`);
    if (sch.predicted_drift_mm) {
      bits.push(`the arc is expected to walk the corner <b>${V.fmt(sch.predicted_drift_mm, 2)} mm</b> while welding`);
    }
  }
  V.$("sm-detail-meta").innerHTML = bits.join(" &middot; ");
}
