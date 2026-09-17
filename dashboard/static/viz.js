/* Chart primitives.
 *
 * Every figure on this dashboard is built from these, so a bar on one page means
 * the same thing as a bar on another. The rules they enforce: one axis per chart,
 * recessive grid, thin marks, colour assigned by identity rather than by rank, and
 * a hover reading on every mark, because a number nobody can read off the picture
 * is a number nobody checks.
 */

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)",
                       "var(--series-4)", "var(--series-5)"];

// Colour follows the model, not its position in whatever list is on screen.
export const ARM_COLOUR = {
  nano: "var(--series-1)",
  lightning: "var(--series-2)",
  super: "var(--series-3)",
  ultra: "var(--series-4)",
  routed: "var(--series-5)",
};
export const ARM_CLASS = { nano: "s1", lightning: "s2", super: "s3", ultra: "s4", routed: "s5" };

export const BEFORE = "var(--before)";
export const AFTER = "var(--after)";
export const GOOD = "var(--good)";
export const WARN = "var(--warning)";
export const BAD = "var(--critical)";

export const $ = (id) => document.getElementById(id);
export const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? "–" : Number(v).toFixed(d);

export const pct = (v, d = 0) => (v === null || v === undefined) ? "–"
  : (v * 100).toFixed(d) + "%";

export function nice(max, ticks = 4) {
  if (!(max > 0)) return [1, [0, 0.25, 0.5, 0.75, 1]];
  const raw = max / ticks;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].find((m) => m * mag >= raw) * mag;
  const top = Math.ceil(max / step) * step;
  const out = [];
  for (let v = 0; v <= top + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return [top, out];
}

/* ------------------------------------------------------------------- tooltip */

let tip;
export function tipInit() {
  if (tip) return;
  tip = document.createElement("div");
  tip.id = "tip";
  document.body.appendChild(tip);
  document.addEventListener("mousemove", (e) => {
    if (!tip.classList.contains("on")) return;
    const pad = 16, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > innerWidth - 8) x = e.clientX - w - pad;
    if (y + h > innerHeight - 8) y = e.clientY - h - pad;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  });
}

/** Attach a hover reading to every [data-tip] inside a container. */
export function bindTips(root) {
  tipInit();
  root.querySelectorAll("[data-tip]").forEach((el) => {
    el.addEventListener("mouseenter", () => {
      tip.innerHTML = el.getAttribute("data-tip");
      tip.classList.add("on");
    });
    el.addEventListener("mouseleave", () => tip.classList.remove("on"));
  });
}

/** Render markup into a host and wire its tooltips in one step. */
export function paint(host, markup) {
  if (!host) return;
  host.innerHTML = markup;
  bindTips(host);
}

/* --------------------------------------------------------------- scaffolding */

export function svg(w, h, inner, cls = "chart") {
  return `<svg class="${cls}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet"
    style="height:${h}px;max-height:${h}px">${inner}</svg>`;
}

/** Horizontal gridlines plus their labels, drawn behind everything else. */
export function gridY(x0, x1, yOf, ticks, unit = "", fmtTick = (v) => v) {
  return ticks.map((v) => `
    <line class="gridline" x1="${x0}" y1="${yOf(v)}" x2="${x1}" y2="${yOf(v)}"/>
    <text class="axis-label" x="${x0 - 7}" y="${yOf(v) + 3.5}" text-anchor="end">${fmtTick(v)}${unit}</text>
  `).join("");
}

export function gridX(y0, y1, xOf, ticks, unit = "", fmtTick = (v) => v) {
  return ticks.map((v) => `
    <line class="gridline" x1="${xOf(v)}" y1="${y0}" x2="${xOf(v)}" y2="${y1}"/>
    <text class="axis-label" x="${xOf(v)}" y="${y1 + 13}" text-anchor="middle">${fmtTick(v)}${unit}</text>
  `).join("");
}

export function axisTitle(x, y, text, anchor = "start") {
  return `<text class="axis-label" x="${x}" y="${y}" text-anchor="${anchor}">${esc(text)}</text>`;
}

/* -------------------------------------------------------------------- marks */

/** A bar with a rounded data-end, anchored to the baseline. */
export function bar(x, y, w, h, colour, extra = "") {
  const r = Math.min(4, w / 2, Math.max(0, h));
  if (h <= 0.5) return `<rect x="${x}" y="${y}" width="${w}" height="1" fill="${colour}" ${extra}/>`;
  return `<path d="M${x},${y + h} L${x},${y + r} Q${x},${y} ${x + r},${y}
    L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r} L${x + w},${y + h} Z"
    fill="${colour}" ${extra}/>`;
}

/** A bar growing to the right, rounded at the data-end. */
export function barH(x, y, w, h, colour, extra = "") {
  const r = Math.min(4, h / 2, Math.max(0, w));
  if (w <= 0.5) return `<rect x="${x}" y="${y}" width="1" height="${h}" fill="${colour}" ${extra}/>`;
  return `<path d="M${x},${y} L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r}
    L${x + w},${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} L${x},${y + h} Z"
    fill="${colour}" ${extra}/>`;
}

export function line(points, colour, width = 2, extra = "") {
  return `<polyline points="${points}" fill="none" stroke="${colour}"
    stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round" ${extra}/>`;
}

/** A dot with a surface ring, so overlapping marks stay countable. */
export function dot(x, y, r, colour, extra = "") {
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="${colour}"
    stroke="var(--bg)" stroke-width="1.4" ${extra}/>`;
}

/* ------------------------------------------------------------- compositions */

/** A sparkline for a stat tile: no axes, no labels, just the shape of the run. */
export function spark(values, colour, w = 220, h = 26, fill = false) {
  if (!values || values.length < 2) return "";
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = Math.max(1e-9, hi - lo);
  const xOf = (i) => (i / (values.length - 1)) * w;
  const yOf = (v) => h - 3 - ((v - lo) / span) * (h - 8);
  const pts = values.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const area = fill
    ? `<polygon points="0,${h} ${pts} ${w},${h}" fill="${colour}" opacity=".13"/>` : "";
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    ${area}${line(pts, colour, 1.3)}</svg>`;
}

/**
 * Two states of the same entity, one row each, joined by a rule.
 * The form the before/after claim deserves: the pair is the unit, not two bars.
 */
export function dumbbell(rows, opts = {}) {
  const w = opts.width ?? 700, rowH = opts.rowH ?? 13, padL = opts.padL ?? 46,
        padR = 90, padT = 26, padB = 26;
  const h = padT + rows.length * rowH + padB;
  const all = rows.flatMap((r) => [r.a, r.b]).filter((v) => v > 0);
  const lo = Math.max(0.05, Math.min(...all) * 0.7), hi = Math.max(...all) * 1.25;
  const xOf = (v) => padL + (Math.log10(Math.max(v, lo)) - Math.log10(lo)) /
    (Math.log10(hi) - Math.log10(lo)) * (w - padL - padR);

  const decades = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= lo && v <= hi) decades.push(v);
    }
  }
  let out = decades.map((v) => `
    <line class="gridline" x1="${xOf(v)}" y1="${padT - 8}" x2="${xOf(v)}" y2="${h - padB + 2}"/>
    <text class="axis-label" x="${xOf(v)}" y="${h - padB + 15}" text-anchor="middle">${v >= 1 ? v : v.toFixed(1)}</text>`).join("");
  out += axisTitle(padL, padT - 14, opts.xLabel ?? "millimetres, log scale");

  rows.forEach((r, i) => {
    const y = padT + i * rowH + rowH / 2;
    const xa = xOf(r.a), xb = xOf(r.b);
    out += `<g data-tip="${esc(r.tip ?? "")}">
      <rect x="${padL - 6}" y="${y - rowH / 2}" width="${w - padL - padR + 12}" height="${rowH}" fill="transparent"/>
      <line x1="${xa}" y1="${y}" x2="${xb}" y2="${y}" stroke="${opts.ruleColour ?? "#212a35"}" stroke-width="1"/>
      ${dot(xa, y, 2.6, opts.aColour ?? BEFORE)}
      ${dot(xb, y, 2.6, opts.bColour ?? AFTER)}
    </g>`;
  });
  return svg(w, h, out);
}

/**
 * A matrix of counts. One hue, light to dark, because the value is a magnitude.
 * Every cell carries its number, so the colour is a summary and not the only read.
 */
export function heat(rowLabels, colLabels, matrix, opts = {}) {
  const cell = opts.cell ?? 46, padL = opts.padL ?? 92, padT = opts.padT ?? 44;
  const w = padL + colLabels.length * cell + 12;
  const h = padT + rowLabels.length * cell + 28;
  const hi = Math.max(1, ...matrix.flat());
  let out = "";
  colLabels.forEach((label, j) => {
    out += `<text class="axis-label" x="${padL + j * cell + cell / 2}" y="${padT - 10}"
      text-anchor="middle" font-size="9.5">${esc(label)}</text>`;
  });
  rowLabels.forEach((r, i) => {
    out += `<text class="axis-label" x="${padL - 10}" y="${padT + i * cell + cell / 2 + 3.5}"
      text-anchor="end" font-size="9.5">${esc(r)}</text>`;
    matrix[i].forEach((v, j) => {
      const t = v / hi;
      const x = padL + j * cell, y = padT + i * cell;
      const diag = i === j;
      const colour = v === 0 ? "var(--surface-2)"
        : diag ? `color-mix(in oklab, var(--good) ${18 + t * 74}%, var(--surface-1))`
               : `color-mix(in oklab, var(--critical) ${16 + t * 70}%, var(--surface-1))`;
      out += `<rect x="${x + 1}" y="${y + 1}" width="${cell - 2}" height="${cell - 2}" rx="7"
        fill="${colour}" stroke="var(--bg)" stroke-width="2"
        data-tip="${esc(opts.tip ? opts.tip(i, j, v) : `${r} → ${colLabels[j]}: <b>${v}</b>`)}"/>`;
      if (v > 0) out += `<text x="${x + cell / 2}" y="${y + cell / 2 + 4}" text-anchor="middle"
        font-size="12" fill="${t > 0.45 ? "#ffffff" : "var(--text-secondary)"}" font-weight="600">${v}</text>`;
    });
  });
  out += axisTitle(padL, h - 8, opts.xTitle ?? "");
  return svg(w, h, out);
}

/** Proportions of a whole, one row, segments separated by a surface gap. */
export function stackedRow(segments, opts = {}) {
  const w = opts.width ?? 320, h = opts.height ?? 16;
  const total = segments.reduce((a, s) => a + s.value, 0) || 1;
  let x = 0, out = "";
  segments.forEach((s) => {
    const sw = (s.value / total) * w;
    if (sw > 0) {
      out += `<rect x="${x + 1}" y="0" width="${Math.max(0, sw - 2)}" height="${h}" rx="4"
        fill="${s.colour}" data-tip="${esc(s.tip ?? `${s.label}: <b>${s.value}</b>`)}"/>`;
    }
    x += sw;
  });
  return svg(w, h, out);
}

/** A distribution, binned, with the mean marked. Answers "is this typical?". */
export function histogram(values, opts = {}) {
  const w = opts.width ?? 320, h = opts.height ?? 150;
  const padL = 34, padB = 26, padT = 16, padR = 8;
  if (!values.length) return svg(w, h, "");
  const lo = opts.min ?? 0, hi = opts.max ?? Math.max(...values) * 1.05;
  const bins = opts.bins ?? 16;
  const counts = new Array(bins).fill(0);
  values.forEach((v) => {
    const k = Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)));
    counts[k]++;
  });
  const [top, ticks] = nice(Math.max(...counts), 3);
  const yOf = (v) => h - padB - (v / top) * (h - padB - padT);
  const bw = (w - padL - padR) / bins;
  let out = gridY(padL, w - padR, yOf, ticks, "", (v) => v);
  counts.forEach((n, i) => {
    const x = padL + i * bw;
    const a = lo + (i / bins) * (hi - lo), b = lo + ((i + 1) / bins) * (hi - lo);
    out += bar(x + 1, yOf(n), Math.max(1, bw - 2), h - padB - yOf(n), opts.colour ?? SERIES[0],
      `opacity=".9" data-tip="${esc(`${a.toFixed(2)}–${b.toFixed(2)} ${opts.unit ?? ""}: <b>${n}</b> ${opts.noun ?? "parts"}`)}"`);
  });
  const xOf = (v) => padL + ((v - lo) / (hi - lo)) * (w - padL - padR);
  [lo, (lo + hi) / 2, hi].forEach((v) => {
    out += `<text class="axis-label" x="${xOf(v)}" y="${h - padB + 14}" text-anchor="middle">${v.toFixed(v < 10 ? 1 : 0)}</text>`;
  });
  if (opts.mark !== undefined && opts.mark >= lo && opts.mark <= hi) {
    out += `<line x1="${xOf(opts.mark)}" y1="${padT - 6}" x2="${xOf(opts.mark)}" y2="${h - padB}"
      stroke="${opts.markColour ?? "var(--after)"}" stroke-width="2" stroke-dasharray="3 3"/>
      <text class="axis-label" x="${xOf(opts.mark)}" y="${padT - 10}" text-anchor="middle"
        fill="${opts.markColour ?? "var(--after)"}">${esc(opts.markLabel ?? "")}</text>`;
  }
  out += axisTitle(padL, h - 3, opts.xTitle ?? "");
  return svg(w, h, out);
}

/** Ranked horizontal bars with the value written at the data-end. */
export function ranked(rows, opts = {}) {
  const w = opts.width ?? 340, rowH = opts.rowH ?? 30, padL = opts.padL ?? 118, padR = 52;
  const h = rows.length * rowH + 14;
  const hi = opts.max ?? Math.max(...rows.map((r) => r.value), 1e-9);
  const xOf = (v) => (v / hi) * (w - padL - padR);
  let out = "";
  rows.forEach((r, i) => {
    const y = i * rowH + 7, bh = rowH - 14;
    out += `<text x="${padL - 10}" y="${y + bh / 2 + 4}" text-anchor="end" font-size="11"
      fill="var(--text-secondary)" font-family="var(--sans)">${esc(r.label)}</text>`;
    out += barH(padL, y, Math.max(1, xOf(r.value)), bh, r.colour ?? SERIES[0],
      `data-tip="${esc(r.tip ?? `${r.label}: <b>${r.display ?? r.value}</b>`)}"`);
    out += `<text x="${padL + xOf(r.value) + 8}" y="${y + bh / 2 + 4}" font-size="11.5"
      fill="var(--text-primary)">${esc(r.display ?? r.value)}</text>`;
  });
  return svg(w, h, out);
}
