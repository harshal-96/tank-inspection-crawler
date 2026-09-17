/* The run, as parts rather than as lines.
 *
 * This is the page that used to be a log. A log makes you read fifty rows to learn
 * one thing; a card carries its own shape, so the odd one out is visible before you
 * have read a single number.
 */

import * as V from "../viz.js";
import { refreshLive, state, chosen, choose } from "../store.js";
import { deviations } from "./cell.js";

export const meta = {
  id: "parts",
  title: "The run",
  crumb: "Live",
  group: "Cell",
  icon: "M4 5h16M4 12h16M4 19h10",
};

const SORTS = {
  recent: { label: "Newest first", key: (s) => -s.part_id },
  worst: { label: "Worst tracked", key: (s) => -(s.seam?.error_mm ?? -1) },
  moved: { label: "Moved furthest", key: (s) => -(s.seam?.taught_miss_mm ?? -1) },
  gap: { label: "Widest gap", key: (s) => -(s.seam?.gap_mm_max ?? -1) },
};

let sort = "recent";
let onlyFlagged = false;

export function render() {
  return `
    <p class="lede">
      Fifty parts came off the press and through the cell. Each card is one of them:
      how far a taught path would have been off, where the measurement landed, how the
      seam ran, and what the arc was asked to do about it.
    </p>

    <div class="grid">
      <div class="card w12">
        <header>
          <h2>Run profile</h2>
          <span class="grow" style="flex:1"></span>
          <span class="muted" style="font-size:12px" id="pt-count"></span>
        </header>
        <div class="hint">Every part in build order. The bar is how far the taught path would have
          missed; the line is what the cell tracked to.</div>
        <div id="pt-profile"></div>
        <div class="legend">
          <span><i style="background:var(--before)"></i>taught path miss, mm</span>
          <span><i class="line" style="background:var(--after)"></i>tracked error, mm (right scale)</span>
          <span><i style="background:var(--warning)"></i>a station the cell refused to weld</span>
        </div>
      </div>

      <div class="card w12">
        <header>
          <h2>Parts</h2>
          <span class="grow" style="flex:1"></span>
          <div class="seg" id="pt-sort"></div>
          <button id="pt-flagged">Only flagged</button>
        </header>
        <div class="partlist" id="pt-list"></div>
      </div>

      <div class="card w12" id="pt-detail-card">
        <header><h2 id="pt-detail-title">–</h2></header>
        <div id="pt-detail"></div>
      </div>
    </div>`;
}

export async function mount() {
  await refreshLive();
  V.$("pt-sort").innerHTML = Object.entries(SORTS).map(([k, s]) =>
    `<button data-s="${k}" class="${k === sort ? "on" : ""}">${s.label}</button>`).join("");
  V.$("pt-sort").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      sort = b.dataset.s;
      V.$("pt-sort").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      list();
    };
  });
  V.$("pt-flagged").onclick = () => {
    onlyFlagged = !onlyFlagged;
    V.$("pt-flagged").classList.toggle("on", onlyFlagged);
    list();
  };
  update();
}

export function update() { profile(); list(); detail(); }

function slots() {
  if (!state.live) return [];
  return state.live.ids.map((id) => state.live.byPart.get(id)).filter(Boolean);
}

function profile() {
  const rows = slots().filter((s) => s.seam).sort((a, b) => a.part_id - b.part_id);
  if (!rows.length) return;
  const w = 960, h = 220, padL = 44, padR = 44, padT = 20, padB = 30;
  const hiMiss = Math.max(...rows.map((s) => s.seam.taught_miss_mm));
  const hiErr = Math.max(...rows.map((s) => s.seam.error_mm ?? 0), 1);
  const [topMiss, ticksMiss] = V.nice(hiMiss, 4);
  const yMiss = (v) => h - padB - (v / topMiss) * (h - padB - padT);
  const yErr = (v) => h - padB - (v / (hiErr * 1.35)) * (h - padB - padT);
  const bw = (w - padL - padR) / rows.length;

  let out = V.gridY(padL, w - padR, yMiss, ticksMiss, "", (v) => v.toFixed(0));
  rows.forEach((s, i) => {
    const x = padL + i * bw;
    const refused = s.schedule?.refused_stations || 0;
    out += V.bar(x + bw * 0.16, yMiss(s.seam.taught_miss_mm), bw * 0.68,
      h - padB - yMiss(s.seam.taught_miss_mm),
      refused ? V.WARN : V.BEFORE,
      `opacity=".8" data-tip="${tipFor(s)}"`);
  });
  out += V.line(rows.map((s, i) =>
    `${padL + i * bw + bw / 2},${yErr(s.seam.error_mm ?? 0)}`).join(" "), V.AFTER, 2);
  rows.forEach((s, i) => {
    out += V.dot(padL + i * bw + bw / 2, yErr(s.seam.error_mm ?? 0), 2.6, V.AFTER,
      `data-tip="${tipFor(s)}"`);
  });
  [0, hiErr * 0.675, hiErr * 1.35].forEach((v) => {
    out += `<text class="axis-label" x="${w - padR + 8}" y="${yErr(v) + 3.5}" fill="var(--after)">${v.toFixed(1)}</text>`;
  });
  out += V.axisTitle(padL - 36, padT - 6, "mm");
  out += V.axisTitle(padL, h - 6, "part number, in build order");
  V.paint(V.$("pt-profile"), V.svg(w, h, out));
  V.$("pt-count").textContent = `${rows.length} parts · ${rows.length * 20} stations measured`;
}

function tipFor(s) {
  const t = s.seam, sch = s.schedule, ins = s.inspection;
  return V.esc(`<b>part #${s.part_id}</b><br>
    <span class="k">taught path misses</span> <b>${V.fmt(t.taught_miss_mm, 1)} mm</b><br>
    <span class="k">tracked to</span> <b>${V.fmt(t.error_mm, 2)} mm</b><br>
    <span class="k">seam bows</span> <b>${V.fmt(t.seam_bow_mm, 2)} mm</b><br>
    <span class="k">widest gap</span> <b>${V.fmt(t.gap_mm_max, 2)} mm</b>
    ${sch?.refused_stations ? `<br><span class="k">refused</span> <b>${sch.refused_stations}</b> <span class="k">stations</span>` : ""}
    ${ins ? `<br><span class="k">bead</span> <b>${ins.verdict}</b>` : ""}`);
}

function list() {
  const host = V.$("pt-list");
  if (!host) return;
  let rows = slots().filter((s) => s.seam);
  if (onlyFlagged) {
    rows = rows.filter((s) => (s.schedule?.refused_stations || 0) > 0
      || (s.seam.error_mm ?? 0) > 1 || (s.inspection && s.inspection.verdict !== "pass"));
  }
  rows.sort((a, b) => SORTS[sort].key(a) - SORTS[sort].key(b));
  if (!rows.length) { host.innerHTML = `<div class="empty">Nothing matches.</div>`; return; }

  host.innerHTML = rows.map((s) => {
    const t = s.seam;
    const dev = deviations(t.path || []);
    const flagged = (s.schedule?.refused_stations || 0) > 0;
    const tone = (t.error_mm ?? 0) < 1 ? "var(--after)" : "var(--warning)";
    const sw = 200, sh = 26;
    const span = Math.max(1, Math.max(...dev.map(Math.abs)) * 1.3);
    const pts = dev.map((v, i) =>
      `${(i / Math.max(1, dev.length - 1) * sw).toFixed(1)},${(sh / 2 - (v / span) * (sh / 2 - 2)).toFixed(1)}`).join(" ");
    return `<button class="partcard ${s.part_id === state.chosenPart ? "sel" : ""}" data-id="${s.part_id}">
      <div class="row">
        <span class="id">#${s.part_id}</span>
        ${flagged ? `<span class="badge warn">${s.schedule.refused_stations} refused</span>` : ""}
        ${s.inspection && s.inspection.verdict !== "pass" ? `<span class="badge bad">${V.esc(s.inspection.defect_type)}</span>` : ""}
        <span class="big" style="color:${tone}">${V.fmt(t.error_mm, 2)}</span>
      </div>
      <div class="meta">taught path off by ${V.fmt(t.taught_miss_mm, 1)} mm · bow ${V.fmt(t.seam_bow_mm, 2)} mm</div>
      <svg viewBox="0 0 ${sw} ${sh}" preserveAspectRatio="none">
        <line x1="0" y1="${sh / 2}" x2="${sw}" y2="${sh / 2}" stroke="var(--line)" stroke-width="1"/>
        <polyline points="${pts}" fill="none" stroke="var(--series-1)" stroke-width="1.6"/>
      </svg>
    </button>`;
  }).join("");

  host.querySelectorAll(".partcard").forEach((el) => {
    el.onclick = () => { choose(el.dataset.id); list(); detail(); };
  });
}

function detail() {
  const s = chosen();
  const host = V.$("pt-detail");
  if (!host) return;
  if (!s || !s.seam) { host.innerHTML = `<div class="empty">Pick a part.</div>`; return; }
  const t = s.seam, sch = s.schedule, d = s.decision, ins = s.inspection, m = s.measurement;
  V.$("pt-detail-title").textContent = `Part #${s.part_id}`;

  host.innerHTML = `<div class="grid">
    <div class="w3">
      <table class="plain"><tbody>
        ${kv("Taught path off by", `${V.fmt(t.taught_miss_mm, 2)} mm`)}
        ${kv("Tracked to", `${V.fmt(t.error_mm, 3)} mm`)}
        ${kv("Worst station", `${V.fmt(t.worst_station_mm, 2)} mm`)}
        ${kv("Seam bow", `${V.fmt(t.seam_bow_mm, 2)} mm`)}
        ${kv("Stations", `${t.stations_measured} of ${t.stations_asked}`)}
        ${m ? kv("Corner at", `${m.detected_x.toFixed(4)}, ${m.detected_y.toFixed(4)} m`) : ""}
      </tbody></table>
    </div>
    <div class="w3">
      ${sch ? `<table class="plain"><tbody>
        ${kv("Weldable", `${sch.weldable_stations} stations`)}
        ${kv("Refused", `${sch.refused_stations} stations`)}
        ${kv("Travel", `${V.fmt(sch.travel_mm_s_min, 1)} to ${V.fmt(sch.travel_mm_s_max, 1)} mm/s`)}
        ${kv("Hottest", `${V.fmt(sch.heat_j_per_mm_max, 0)} J/mm`)}
        ${kv("Gap", `mean ${V.fmt(t.gap_mm_mean, 2)}, worst ${V.fmt(t.gap_mm_max, 2)} mm`)}
        ${sch.predicted_drift_mm ? kv("Arc walks corner", `${V.fmt(sch.predicted_drift_mm, 2)} mm`) : ""}
      </tbody></table>` : `<div class="empty">No schedule.</div>`}
    </div>
    <div class="w3">
      ${d ? `<table class="plain"><tbody>
        ${kv("Looks", d.after_looks)}
        ${kv("Model", (d.model || "").split("/").pop())}
        ${kv("Tier", d.tier || "fixed")}
        ${kv("Verdict", d.accept ? "accept" : "look again")}
        ${kv("Latency", `${V.fmt(d.latency_ms, 0)} ms`)}
        ${kv("Routed because", d.route_why || "–")}
      </tbody></table>
      <div class="quote" style="margin-top:9px">${V.esc(d.reason || "")}</div>` : `<div class="empty">No decision.</div>`}
    </div>
    <div class="w3">
      ${ins ? `<table class="plain"><tbody>
        ${kv("Verdict", ins.verdict)}
        ${kv("Called", ins.defect_type)}
        ${kv("Injected", ins.injected_defect ?? "–")}
        ${kv("Scored", ins.correct ? "correct" : "wrong")}
        ${kv("Severity", V.fmt(ins.severity, 2))}
        ${kv("Latency", `${V.fmt(ins.latency_ms, 0)} ms`)}
      </tbody></table>
      <div class="quote" style="margin-top:9px">${V.esc(ins.reasoning || "")}</div>` : `<div class="empty">Not inspected.</div>`}
    </div>
  </div>`;
}

const kv = (k, v) => `<tr><td class="muted" style="width:140px">${V.esc(k)}</td>
  <td class="mono">${V.esc(String(v))}</td></tr>`;
