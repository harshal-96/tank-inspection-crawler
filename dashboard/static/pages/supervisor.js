/* The supervisor: the same rows, asked in words, answered through tools. */

import * as V from "../viz.js";
import { refreshLive, state } from "../store.js";

export const meta = {
  id: "supervisor",
  title: "Cell supervisor",
  crumb: "Live",
  group: "The cell, live",
  icon: "M12 3a6 6 0 016 6v2a6 6 0 01-12 0V9a6 6 0 016-6ZM5 20h14",
};

const PROMPTS = [
  "Is the press drifting, or is it the fixture?",
  "Which parts refused a point, and how open was the joint there?",
  "Give me a shift report.",
  "How far off would a taught path have been on the worst part?",
  "Has the inspection missed anything I should be worried about?",
];

export function render() {
  return `
    <p class="lede">
      The supervisor has no private copy of anything. It answers by querying the
      same tables this dashboard draws from, and it can search published welding standards
      when a question needs a figure it should not invent. Every answer it gives can be
      checked against a page you can open yourself.
    </p>

    <div class="grid">
      <div class="card w8">
        <header><h2>Conversation</h2></header>
        <div id="log">
          <div class="msg assistant">Supervisor online. Ask about drift, a specific part, or request a shift report.</div>
        </div>
        <div class="composer">
          <input id="input" placeholder="Is the press drifting, or is it the fixture?" autocomplete="off">
          <button id="send">Send</button>
        </div>
        <div class="suggest" id="suggest"></div>
      </div>

      <div class="card w4">
        <header><h2>What it can reach</h2></header>
        <div id="sv-tools"></div>
        <footer>
          Nothing in an answer comes from the model's memory of welding. A number it quotes is
          either read out of the tables on this dashboard or fetched from a published source and
          attributed.
        </footer>
      </div>

      <div class="card w4">
        <header><h2>Where the run is heading</h2></header>
        <div class="hint">The corner's position across the recent window, against the baseline it
          started from. This is the evidence the shift-level question is asked about.</div>
        <div id="sv-drift"></div>
        <footer id="sv-drift-meta"></footer>
      </div>

      <div class="card w8">
        <header><h2>What the supervisor sees when it looks</h2></header>
        <div id="sv-state"></div>
      </div>
    </div>`;
}

export async function mount() {
  await refreshLive();
  V.$("send").onclick = send;
  V.$("input").onkeydown = (e) => { if (e.key === "Enter") send(); };
  V.$("suggest").innerHTML = PROMPTS.map((p) =>
    `<button data-p="${V.esc(p)}">${V.esc(p)}</button>`).join("");
  V.$("suggest").querySelectorAll("button").forEach((b) => {
    b.onclick = () => { V.$("input").value = b.dataset.p; send(); };
  });
  tools();
  update();
}

export function update() { drift(); runState(); }

function tools() {
  const rows = [
    ["recent_parts", "the last n parts with their measured corner and status"],
    ["part_detail", "everything recorded against one serial"],
    ["drift_check", "the corner's trend across a window, against the baseline"],
    ["inspection_summary", "verdicts, and how they scored against what was injected"],
    ["look_up_standard", "published welding guidance, fetched and attributed"],
    ["write_report", "a shift report, stored back into the same database"],
  ];
  V.$("sv-tools").innerHTML = `<table class="plain"><tbody>${rows.map(([n, d]) =>
    `<tr><td class="mono" style="width:150px;color:var(--series-1)">${V.esc(n)}</td>
      <td class="muted">${V.esc(d)}</td></tr>`).join("")}</tbody></table>`;
}

function drift() {
  const d = state.live?.stats?.drift;
  const rows = (state.live?.measurements || []).filter((m) => m.detected_y != null)
    .slice().sort((a, b) => a.part_id - b.part_id);
  const host = V.$("sv-drift");
  if (!host || !rows.length) return;

  const w = 400, h = 190, padL = 46, padR = 12, padT = 18, padB = 28;
  const ys = rows.map((m) => m.detected_y * 1000);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const pad = Math.max(1, (hi - lo) * 0.15);
  const yOf = (v) => h - padB - ((v - (lo - pad)) / ((hi + pad) - (lo - pad))) * (h - padB - padT);
  const xOf = (i) => padL + (i / Math.max(1, rows.length - 1)) * (w - padL - padR);

  let out = "";
  [lo, (lo + hi) / 2, hi].forEach((v) => {
    out += `<line class="gridline" x1="${padL}" y1="${yOf(v)}" x2="${w - padR}" y2="${yOf(v)}"/>
      <text class="axis-label" x="${padL - 7}" y="${yOf(v) + 3.5}" text-anchor="end">${v.toFixed(0)}</text>`;
  });
  if (d?.baseline_mean_y_mm != null) {
    out += `<line x1="${padL}" y1="${yOf(d.baseline_mean_y_mm)}" x2="${w - padR}" y2="${yOf(d.baseline_mean_y_mm)}"
      stroke="var(--before)" stroke-width="2" stroke-dasharray="4 3"/>
      <text class="axis-label" x="${w - padR}" y="${yOf(d.baseline_mean_y_mm) - 6}" text-anchor="end"
        fill="var(--before)">baseline</text>`;
  }
  out += V.line(rows.map((m, i) => `${xOf(i)},${yOf(m.detected_y * 1000)}`).join(" "),
    "var(--series-1)", 1.8);
  rows.forEach((m, i) => {
    out += V.dot(xOf(i), yOf(m.detected_y * 1000), 2.4, "var(--series-1)",
      `data-tip="<b>part #${m.part_id}</b><br><span class='k'>corner y</span> <b>${(m.detected_y * 1000).toFixed(2)} mm</b><br>
        <span class='k'>error against truth</span> <b>${V.fmt(m.error_mm, 2)} mm</b>"`);
  });
  out += V.axisTitle(padL, h - 5, "parts, in build order");
  out += V.axisTitle(padL - 40, padT - 4, "mm");
  V.paint(host, V.svg(w, h, out));

  if (d) {
    V.$("sv-drift-meta").innerHTML =
      `Over the last <b>${d.window}</b> parts the corner sits <b>${V.fmt(d.shift_mm, 2)} mm</b> from the
       baseline, against a part-to-part scatter of <b>${V.fmt(d.spread_mm, 1)} mm</b>. That is
       <b>${V.fmt(d.sigma, 2)}σ</b>, so the run reads as
       <b>${d.trending ? "trending" : "not trending"}</b>: the press is handing over parts that vary,
       not parts that are walking.`;
  }
}

function runState() {
  const live = state.live;
  const host = V.$("sv-state");
  if (!host || !live) return;
  const s = live.stats;
  const refused = live.schedules.reduce((a, x) => a + (x.refused_stations || 0), 0);
  const looks = live.decisions.reduce((a, x) => a + (x.after_looks || 0), 0);

  const cards = [
    ["Parts through the cell", s.parts_total, `${live.seams.length} with a tracked seam`],
    ["Correction", `${V.fmt(s.correction.mean_mm, 2)} mm`,
      `median ${V.fmt(s.correction.median_mm, 2)}, p95 ${V.fmt(s.correction.p95_mm, 2)}, none over 3 mm`],
    ["Points refused", refused, `out of ${live.schedules.length * 20} scheduled`],
    ["Looks spent", looks, `${(looks / Math.max(1, live.decisions.length)).toFixed(2)} per part`],
    ["Inspection", V.pct(s.inspection.accuracy),
      `${s.inspection.true_pass} clean, ${s.inspection.true_defect} caught, ${s.inspection.false_defect} false alarms`],
    ["Drift", `${V.fmt(s.drift.sigma, 2)}σ`, s.drift.trending ? "trending" : "inside normal scatter"],
  ];
  host.innerHTML = `<div class="tiles">${cards.map(([l, v, sub]) => `<div class="tile">
    <div class="label">${V.esc(l)}</div><div class="value" style="font-size:24px">${v}</div>
    <div class="sub">${V.esc(sub)}</div></div>`).join("")}</div>`;
}

/* ------------------------------------------------------------------- chatting */

let busy = false;

async function send() {
  if (busy) return;
  const input = V.$("input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  bubble("user", text);
  busy = true;
  V.$("send").disabled = true;
  const thinking = bubble("assistant", "…");
  try {
    const res = await fetch("/api/agent/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    thinking.remove();
    (data.tool_calls || []).forEach((t) => {
      const args = typeof t.arguments === "string" ? t.arguments : JSON.stringify(t.arguments ?? {});
      bubble("tool", `${t.name}(${args})`);
    });
    bubble("assistant", data.reply || "(no reply)");
  } catch (err) {
    thinking.remove();
    bubble("assistant", `Could not reach the supervisor: ${err.message}`);
  } finally {
    busy = false;
    V.$("send").disabled = false;
    refreshLive();
  }
}

function bubble(kind, text) {
  const log = V.$("log");
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}
