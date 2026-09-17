// What the cell knows, drawn so it can be read rather than stacked.
//
// Everything here comes from the service. Nothing is computed in the browser that
// the cell already measured, and nothing is drawn that the cell has not measured.

import { Cell3D } from "/static/cell3d.js";

const $ = (id) => document.getElementById(id);
const COL = {
  taught: "#6e7681", tracked: "#58a6ff", ok: "#3fb950",
  warn: "#e3a008", bad: "#e5534b", accent: "#76b900", muted: "#8b98a8",
  line: "#243040", dim: "#5d6b7c",
};

let cell = null;
let chosenPart = null;
const state = { seams: [], schedules: [], decisions: [], inspections: [], stats: null };

// ---------------------------------------------------------------------- helpers

function svgEl(id, markup) { $(id).innerHTML = markup; }
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function setConn(ok, note) {
  $("dot").className = "dot " + (ok ? "live" : "down");
  $("conn").textContent = note;
}
function classify(el, value, goodBelow, warnBelow) {
  el.classList.remove("good", "warn", "bad");
  if (value === null || value === undefined) return;
  el.classList.add(value < goodBelow ? "good" : value < warnBelow ? "warn" : "bad");
}

/** How far each station sits off the straight line between the seam's own ends.
 *  This is the whole claim: a taught program drives that line, and the seam does
 *  not follow it, because the press bowed it. */
function deviations(path) {
  if (path.length < 2) return path.map(() => 0);
  const a = path[0], b = path[path.length - 1];
  const dx = b.x_mm - a.x_mm, dy = b.y_mm - a.y_mm;
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return path.map(() => 0);
  const ux = dx / len, uy = dy / len;
  return path.map(p => {
    const rx = p.x_mm - a.x_mm, ry = p.y_mm - a.y_mm;
    const along = rx * ux + ry * uy;
    return (rx - along * ux) * uy - (ry - along * uy) * ux;
  });
}

// ------------------------------------------------------------------- the charts

function drawSeamProfile(track, schedule) {
  const path = (track && track.path) || [];
  $("seam-part").textContent = track ? `part #${track.part_id}` : "";
  if (!path.length) { svgEl("seam-path", ""); $("seam-meta").textContent = ""; return; }

  const W = 320, H = 250, padL = 40, padR = 14, padT = 12, padB = 26;
  const dev = deviations(path);
  const along = path.map(p => p.along_mm);
  const loA = Math.min(...along), hiA = Math.max(...along);
  const span = Math.max(0.8, Math.max(...dev.map(Math.abs)) * 1.55);
  const xOf = d => padL + ((d + span) / (2 * span)) * (W - padL - padR);
  const yOf = a => H - padB - ((a - loA) / Math.max(1e-6, hiA - loA)) * (H - padT - padB);

  let out = "";
  const step = span > 3 ? 2 : 1;
  for (let mm = -Math.ceil(span); mm <= Math.ceil(span); mm += step) {
    const x = xOf(mm);
    out += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${H - padB}" stroke="${COL.line}" stroke-width="1" opacity=".55"/>`;
    out += `<text x="${x}" y="${H - 10}" fill="${COL.dim}" font-size="9" text-anchor="middle">${mm}</text>`;
  }
  out += `<text x="${W / 2}" y="${H}" fill="${COL.muted}" font-size="10" text-anchor="middle">millimetres across the taught path</text>`;
  out += `<text x="6" y="${padT + 8}" fill="${COL.dim}" font-size="9">top</text>`;
  out += `<text x="6" y="${H - padB}" fill="${COL.dim}" font-size="9">base</text>`;
  out += `<line x1="${xOf(0)}" y1="${padT}" x2="${xOf(0)}" y2="${H - padB}" stroke="${COL.taught}" stroke-width="2"/>`;

  const pts = path.map((p, i) => `${xOf(dev[i]).toFixed(1)},${yOf(p.along_mm).toFixed(1)}`).join(" ");
  out += `<polyline points="${pts}" fill="none" stroke="${COL.tracked}" stroke-width="2"/>`;

  const stations = (schedule && schedule.stations) || [];
  path.forEach((p, i) => {
    const st = stations[i];
    const colour = !st ? COL.tracked : !st.weldable ? COL.bad
      : st.needs_attention ? COL.warn : COL.ok;
    out += `<circle cx="${xOf(dev[i]).toFixed(1)}" cy="${yOf(p.along_mm).toFixed(1)}" r="3.2"
      fill="${colour}" stroke="#121820" stroke-width="1">
      <title>${p.along_mm.toFixed(0)} mm up the seam: ${dev[i].toFixed(2)} mm off the straight line</title></circle>`;
  });
  svgEl("seam-path", out);

  const bits = [];
  if (track.error_mm != null) bits.push(`tracked to <b>${track.error_mm.toFixed(2)} mm</b> of the true seam`);
  if (track.taught_miss_mm != null) bits.push(`taught path would miss by <b>${track.taught_miss_mm.toFixed(1)} mm</b>`);
  if (track.wander_mm != null) bits.push(`seam wanders <b>${track.wander_mm.toFixed(2)} mm</b> off its own straight line`);
  $("seam-meta").innerHTML = bits.join(" &middot; ");
}

function drawSchedule(schedule) {
  const stations = (schedule && schedule.stations) || [];
  if (!stations.length) { svgEl("sched", ""); $("sched-meta").textContent = "No schedule for this part."; return; }

  const W = 320, H = 250, padL = 40, padR = 14, padT = 12, padB = 26;
  const along = stations.map(s => s.along_mm);
  const loA = Math.min(...along), hiA = Math.max(...along);
  const yOf = a => H - padB - ((a - loA) / Math.max(1e-6, hiA - loA)) * (H - padT - padB);
  const hiHeat = Math.max(...stations.map(s => s.heat_j_per_mm), 1) * 1.1;
  const xOf = v => padL + (v / hiHeat) * (W - padL - padR);

  let out = "";
  for (let v = 0; v <= hiHeat; v += 50) {
    const x = xOf(v);
    out += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${H - padB}" stroke="${COL.line}" stroke-width="1" opacity=".5"/>`;
    out += `<text x="${x}" y="${H - 10}" fill="${COL.dim}" font-size="9" text-anchor="middle">${v}</text>`;
  }
  out += `<text x="${W / 2}" y="${H}" fill="${COL.muted}" font-size="10" text-anchor="middle">heat input, joules per millimetre</text>`;

  const rowH = (H - padT - padB) / stations.length;
  stations.forEach(st => {
    const y = yOf(st.along_mm) - rowH * 0.34;
    const colour = !st.weldable ? COL.bad : st.needs_attention ? COL.warn : COL.accent;
    out += `<rect x="${padL}" y="${y.toFixed(1)}" width="${Math.max(1, xOf(st.heat_j_per_mm) - padL).toFixed(1)}"
      height="${(rowH * 0.68).toFixed(1)}" fill="${colour}" opacity=".85" rx="1.5">
      <title>${st.along_mm.toFixed(0)} mm up: ${st.heat_j_per_mm.toFixed(0)} J/mm at ${st.travel_mm_s.toFixed(1)} mm/s, ${st.current_a.toFixed(0)} A, gap ${st.gap_mm.toFixed(2)} mm
${esc(st.reason || "")}</title></rect>`;
  });
  svgEl("sched", out);

  const drift = schedule.predicted_drift_mm;
  $("sched-meta").innerHTML =
    `<b>${schedule.weldable_stations}</b> weldable &middot; <b>${schedule.flagged_stations}</b> flagged &middot; ` +
    `<b>${schedule.refused_stations}</b> refused &middot; travel ` +
    `${schedule.travel_mm_s_min?.toFixed(1)}&ndash;${schedule.travel_mm_s_max?.toFixed(1)} mm/s` +
    (drift ? ` &middot; the arc is expected to walk the corner <b>${drift.toFixed(2)} mm</b> while welding` : "");
}

function drawDumbbell(seams) {
  const rows = seams
    .filter(s => s.error_mm != null && s.taught_miss_mm != null)
    .sort((a, b) => b.taught_miss_mm - a.taught_miss_mm);
  if (!rows.length) { svgEl("dumbbell", ""); $("dumbbell-meta").textContent = "No scored parts yet."; return; }

  const W = 700, H = 270, padL = 46, padR = 16, padT = 26, padB = 26;
  const lo = 0.3, hi = Math.max(...rows.map(r => r.taught_miss_mm)) * 1.2;
  const xOf = v => padL + (Math.log10(Math.max(v, lo)) - Math.log10(lo)) /
    (Math.log10(hi) - Math.log10(lo)) * (W - padL - padR);
  const rowH = (H - padT - padB) / rows.length;

  let out = "";
  for (const tick of [0.5, 1, 2, 5, 10, 20, 50]) {
    if (tick > hi) continue;
    const x = xOf(tick);
    out += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${H - padB}" stroke="${COL.line}" stroke-width="1" opacity=".6"/>`;
    out += `<text x="${x}" y="${padT - 8}" fill="${COL.dim}" font-size="9" text-anchor="middle">${tick}</text>`;
  }
  out += `<text x="${W / 2}" y="${H - 6}" fill="${COL.muted}" font-size="10" text-anchor="middle">error against the true seam, millimetres</text>`;
  out += `<text x="4" y="${padT + 8}" fill="${COL.dim}" font-size="9">worst arrival</text>`;
  out += `<text x="4" y="${H - padB - 2}" fill="${COL.dim}" font-size="9">best arrival</text>`;

  rows.forEach((r, i) => {
    const y = padT + i * rowH + rowH / 2;
    const x0 = xOf(r.taught_miss_mm), x1 = xOf(r.error_mm);
    out += `<line x1="${x1.toFixed(1)}" y1="${y.toFixed(1)}" x2="${x0.toFixed(1)}" y2="${y.toFixed(1)}"
      stroke="${COL.taught}" stroke-width="1.6" opacity=".55"/>`;
    out += `<circle cx="${x0.toFixed(1)}" cy="${y.toFixed(1)}" r="3" fill="${COL.bad}" opacity=".9">
      <title>part #${r.part_id}: taught path misses by ${r.taught_miss_mm.toFixed(1)} mm</title></circle>`;
    out += `<circle cx="${x1.toFixed(1)}" cy="${y.toFixed(1)}" r="3" fill="${COL.ok}">
      <title>part #${r.part_id}: tracked to ${r.error_mm.toFixed(2)} mm</title></circle>`;
  });
  svgEl("dumbbell", out);

  const meanE = rows.reduce((a, r) => a + r.error_mm, 0) / rows.length;
  const meanT = rows.reduce((a, r) => a + r.taught_miss_mm, 0) / rows.length;
  const sub = rows.filter(r => r.error_mm < 1).length;
  $("dumbbell-meta").innerHTML =
    `<b>${meanT.toFixed(1)} mm</b> without correction, <b>${meanE.toFixed(2)} mm</b> with it, ` +
    `a <b>${(meanT / meanE).toFixed(0)}&times;</b> reduction &middot; ` +
    `<b>${sub} of ${rows.length}</b> parts under a millimetre, every station`;
}

function drawTiers(decisions) {
  if (!decisions.length) { svgEl("tiers", ""); $("tiers-meta").textContent = "No decisions yet."; return; }
  const order = ["nano", "lightning", "super", "ultra", "fixed"];
  const colour = { nano: COL.ok, lightning: COL.accent, super: COL.warn, ultra: COL.bad, fixed: COL.muted };
  const counts = {}, latency = {};
  for (const d of decisions) {
    const t = (d.tier || "fixed").toLowerCase();
    counts[t] = (counts[t] || 0) + 1;
    (latency[t] = latency[t] || []).push(d.latency_ms || 0);
  }
  const present = order.filter(t => counts[t]);
  const total = decisions.length;

  const W = 300, H = 160, padL = 74, padR = 40, padT = 14;
  const rowH = Math.min(30, (H - padT - 20) / Math.max(present.length, 1));
  let out = "";
  present.forEach((t, i) => {
    const y = padT + i * rowH;
    const w = (counts[t] / total) * (W - padL - padR);
    out += `<text x="${padL - 8}" y="${y + rowH * 0.62}" fill="${COL.muted}" font-size="11" text-anchor="end">${t}</text>`;
    out += `<rect x="${padL}" y="${y + rowH * 0.18}" width="${Math.max(2, w).toFixed(1)}" height="${(rowH * 0.5).toFixed(1)}"
      rx="3" fill="${colour[t]}" opacity=".9"/>`;
    const med = latency[t].slice().sort((a, b) => a - b)[Math.floor(latency[t].length / 2)];
    out += `<text x="${padL + Math.max(2, w) + 7}" y="${y + rowH * 0.62}" fill="${COL.dim}" font-size="10">${counts[t]} &middot; ${Math.round(med)}ms</text>`;
  });
  svgEl("tiers", out);

  const cheapest = counts.nano || 0;
  $("tiers-meta").innerHTML =
    `<b>${Math.round(100 * cheapest / total)}%</b> of decisions went to the smallest model in the family, ` +
    `which scoring against ground truth says is the one that is right about them.`;
}

function drawInspections(inspections) {
  const scored = inspections.filter(i => i.injected_defect != null);
  if (!scored.length) { svgEl("insp-chart", ""); $("insp-meta").textContent = "No scored inspections yet."; return; }

  const clean = scored.filter(i => i.injected_defect === "none");
  const faulty = scored.filter(i => i.injected_defect !== "none");
  const falseAlarm = clean.filter(i => i.verdict !== "pass").length;
  const missed = faulty.filter(i => i.verdict === "pass").length;
  const named = scored.filter(i => i.defect_type === i.injected_defect).length;

  const bars = [
    ["clean, passed", clean.length - falseAlarm, COL.ok],
    ["faulty, caught", faulty.length - missed, COL.accent],
    ["false alarm", falseAlarm, COL.warn],
    ["missed", missed, COL.bad],
  ];
  const W = 300, H = 140, padL = 92, padR = 34, padT = 10;
  const rowH = (H - padT - 12) / bars.length;
  const hi = Math.max(...bars.map(b => b[1]), 1);
  let out = "";
  bars.forEach(([label, n, colour], i) => {
    const y = padT + i * rowH;
    const w = (n / hi) * (W - padL - padR);
    out += `<text x="${padL - 8}" y="${y + rowH * 0.62}" fill="${COL.muted}" font-size="10" text-anchor="end">${label}</text>`;
    out += `<rect x="${padL}" y="${y + rowH * 0.2}" width="${Math.max(1.5, w).toFixed(1)}" height="${(rowH * 0.48).toFixed(1)}" rx="3" fill="${colour}" opacity=".9"/>`;
    out += `<text x="${padL + Math.max(1.5, w) + 6}" y="${y + rowH * 0.62}" fill="${COL.dim}" font-size="10">${n}</text>`;
  });
  svgEl("insp-chart", out);

  $("insp-meta").innerHTML =
    `A false alarm scraps a good part; a miss ships a bad one. ` +
    `Naming <i>which</i> defect is harder: right <b>${Math.round(100 * named / scored.length)}%</b> of the time.`;
}

function drawDecisions(decisions) {
  $("dec-empty").style.display = decisions.length ? "none" : "block";
  $("dec-rows").innerHTML = decisions.map(d => {
    const tier = (d.tier || "fixed").toLowerCase();
    return `<tr>
      <td>#${d.part_id}</td>
      <td>${d.after_looks}</td>
      <td><span class="tier ${tier}" title="${esc(d.model || "")}">${tier}</span></td>
      <td>${d.accept ? "accept" : "look again"}</td>
      <td>${d.latency_ms ? Math.round(d.latency_ms) : "&ndash;"}</td>
      <td>${esc((d.reason || "").slice(0, 96))}</td>
    </tr>`;
  }).join("");
}

// ------------------------------------------------------------------- the wiring

function selectedSeam() {
  if (chosenPart == null) return state.seams[0] || null;
  return state.seams.find(s => s.part_id === chosenPart) || state.seams[0] || null;
}

function render() {
  const seam = selectedSeam();
  const schedule = seam ? state.schedules.find(s => s.part_id === seam.part_id) || null : null;

  drawSeamProfile(seam, schedule);
  drawSchedule(schedule);
  drawDumbbell(state.seams);
  drawTiers(state.decisions);
  drawInspections(state.inspections);
  drawDecisions(state.decisions.slice(0, 12));
  if (view) view.showSeam(seam, schedule);

  const scored = state.seams.filter(s => s.error_mm != null);
  if (scored.length) {
    const mean = scored.reduce((a, s) => a + s.error_mm, 0) / scored.length;
    const sub = scored.filter(s => s.error_mm < 1).length;
    $("c-err").textContent = mean.toFixed(2) + " mm";
    classify($("c-err"), mean, 1, 3);
    $("c-err-sub").textContent = `${sub} of ${scored.length} parts under 1 mm, every station`;
  }
  const missed = state.seams.filter(s => s.taught_miss_mm != null);
  if (missed.length) {
    const avg = missed.reduce((a, s) => a + s.taught_miss_mm, 0) / missed.length;
    const worst = Math.max(...missed.map(s => s.taught_miss_mm));
    $("c-taught").textContent = avg.toFixed(1) + " mm";
    $("c-taught-sub").textContent = `worst ${worst.toFixed(1)} mm, with no correction`;
  }
  if (state.stats) {
    $("c-parts").textContent = state.stats.parts_total;
    $("c-parts-sub").textContent = `${state.seams.length} seams tracked`;
    const ins = state.stats.inspection;
    $("c-insp").textContent = ins.accuracy == null ? "–" : Math.round(ins.accuracy * 100) + "%";
    classify($("c-insp"), ins.accuracy == null ? null : 1 - ins.accuracy, 0.1, 0.25);
    $("c-insp-sub").textContent = ins.n_scored
      ? `${ins.false_pass} missed / ${ins.false_defect} false alarm`
      : "awaiting inspections";
  }
}

function fillPicker() {
  const pick = $("part-pick");
  const current = pick.value;
  pick.innerHTML = state.seams
    .map(s => `<option value="${s.part_id}">part #${s.part_id}${s.error_mm != null ? ` &mdash; ${s.error_mm.toFixed(2)} mm` : ""}</option>`)
    .join("");
  if (current && state.seams.some(s => String(s.part_id) === current)) pick.value = current;
  else if (state.seams.length) chosenPart = state.seams[0].part_id;
}

async function refresh() {
  try {
    const [stats, seams, schedules, decisions, inspections] = await Promise.all([
      fetch("/api/stats").then(r => r.json()),
      fetch("/api/seam-tracks?limit=80").then(r => r.json()),
      fetch("/api/weld-schedules?limit=80").then(r => r.json()),
      fetch("/api/view-decisions?limit=80").then(r => r.json()),
      fetch("/api/inspections?limit=80").then(r => r.json()),
    ]);
    Object.assign(state, { stats, seams, schedules, decisions, inspections });
    setConn(true, "live");
    fillPicker();
    render();
  } catch (err) {
    setConn(false, "service unreachable");
  }
}

// ------------------------------------------------------------------------ chat

async function send() {
  const input = $("input"), text = input.value.trim();
  if (!text) return;
  input.value = "";
  $("send").disabled = true;
  $("log").insertAdjacentHTML("beforeend", `<div class="msg user"></div>`);
  $("log").lastElementChild.textContent = text;
  $("log").scrollTop = $("log").scrollHeight;
  try {
    const res = await fetch("/api/agent/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    for (const call of (data.tool_calls || [])) {
      $("log").insertAdjacentHTML("beforeend", `<div class="msg tool"></div>`);
      $("log").lastElementChild.textContent = `${call.name}(${JSON.stringify(call.arguments || {})})`;
    }
    $("log").insertAdjacentHTML("beforeend", `<div class="msg assistant"></div>`);
    $("log").lastElementChild.textContent = data.reply;
  } catch (err) {
    $("log").insertAdjacentHTML("beforeend", `<div class="msg assistant"></div>`);
    $("log").lastElementChild.textContent =
      "Supervisor unavailable. Check that the cloud model is configured in .env.";
  } finally {
    $("send").disabled = false;
    $("log").scrollTop = $("log").scrollHeight;
    input.focus();
  }
}

// ------------------------------------------------------------------------ start

let view = null;
const EXAGGERATIONS = [1, 20, 60];
let exagIndex = 0;

(async function start() {
  try {
    view = new Cell3D($("view"));
    cell = await fetch("/api/cell").then(r => r.json());
    view.build(cell);
    view.frameCell();
  } catch (err) {
    setConn(false, "3D view failed to start");
  }

  $("v-cell").onclick = () => { view?.frameCell(); setActive("v-cell"); };
  $("v-seam").onclick = () => { view?.frameSeam(); setActive("v-seam"); };
  $("v-spin").onclick = () => {
    if (!view) return;
    view.spinning = !view.spinning;
    $("v-spin").classList.toggle("on", view.spinning);
  };
  $("v-exag").onclick = () => {
    exagIndex = (exagIndex + 1) % EXAGGERATIONS.length;
    const k = EXAGGERATIONS[exagIndex];
    view?.setExaggeration(k);
    // Always say the factor. A bowed seam a millimetre wide is invisible at the
    // scale of a workcell, so the shape has to be multiplied to be read, and a
    // picture that is exaggerated without saying so is not evidence.
    $("v-exag").textContent = `Exaggerate ×${k}`;
    $("v-exag").classList.toggle("on", k > 1);
  };
  function setActive(id) {
    for (const b of ["v-cell", "v-seam"]) $(b).classList.toggle("on", b === id);
  }

  $("part-pick").onchange = (e) => { chosenPart = Number(e.target.value); render(); };
  $("send").onclick = send;
  $("input").addEventListener("keydown", e => { if (e.key === "Enter") send(); });

  await refresh();
  setInterval(refresh, 5000);
})();
