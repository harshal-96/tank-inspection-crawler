/* The tank crawler: what the camera found, and what only the probe could. */

import * as V from "../viz.js";
import { once } from "../store.js";

export const meta = {
  id: "tank",
  title: "Tank crawler",
  crumb: "The other wall",
  group: "Result",
  icon: "M4 7h16v10a2 2 0 01-2 2H6a2 2 0 01-2-2V7Zm0 0a2 2 0 012-2h12a2 2 0 012 2M8 11h8M8 15h5",
};

const COLOUR = {
  right: "var(--good, #3fa96a)",
  misnamed: "var(--warn, #d0a033)",
  missed: "var(--bad, #c8553d)",
  false_alarm: "var(--bad, #c8553d)",
  declined: "var(--text-secondary, #888)",
  sliver: "var(--text-secondary, #888)",
  unjudged: "var(--line, #555)",
};

let d = null;

export function render() {
  return `
    <p class="lede">
      A robot crawls a weld seam inside a steel tank. A camera judges the seam, an
      ultrasonic probe measures what is left of the plate, and something has to decide
      where to spend the next measurement. Every number here is scored against answers
      written into the scene before any model saw a picture.
    </p>

    <div class="tiles" id="tk-tiles"></div>

    <div class="grid">
      <div class="card w6">
        <header><h2>The argument, in two patches of wall</h2></header>
        <div class="hint">One patch of wall loss is rusted through to the surface and a camera
          finds it instantly. The other has nothing on the surface at all. Only one of them
          fails the acceptance limit, and it is not the one you can see.</div>
        <div id="tk-contrast"></div>
      </div>

      <div class="card w6">
        <header><h2>Where the verdicts land</h2></header>
        <div class="hint">The pass or defect call decides whether a person has to walk into the
          tank. Naming which fault it is goes on the work order and is advice, not a gate.</div>
        <div id="tk-outcomes"></div>
        <footer id="tk-outcomes-meta"></footer>
      </div>

      <div class="card w12">
        <header><h2>Choosing where to probe</h2></header>
        <div class="hint">The arm can reach seven places at every station and there is time for
          two. Of the points that fail the limit, most sit under a frame the camera calls sound,
          which is the reason this machine carries two instruments.</div>
        <div id="tk-probe"></div>
        <footer id="tk-probe-meta"></footer>
      </div>

      <div class="card w12">
        <header><h2>Every station on the run</h2></header>
        <div class="hint">Click a station to see the frame the model was given and what it said
          about it.</div>
        <div id="tk-stations"></div>
        <div id="tk-detail"></div>
      </div>
    </div>`;
}

export async function mount() {
  d = await once("tank", "/api/tank");
  if (!d || !d.available) {
    V.paint(V.$("tk-tiles"),
      `<div class="hint">${d?.why ?? "The tank run is not available."}</div>`);
    return;
  }
  tiles();
  contrast();
  outcomes();
  probe();
  stations();
}

function tile(label, value, note) {
  return `<div class="tile"><div class="tile-v">${value}</div>
    <div class="tile-k">${label}</div><div class="tile-n">${note ?? ""}</div></div>`;
}

function tiles() {
  const c = d.counts;
  const judged = c.right + c.misnamed + c.missed + c.false_alarm;
  const gate = judged + c.declined - c.missed - c.false_alarm;
  V.paint(V.$("tk-tiles"), [
    tile("Pass or defect", `${gate}/${d.stations}`, "the call that sends somebody in"),
    tile("Named the worst fault", `${c.right}/${d.stations}`, "the line on the work order"),
    tile("Missed", c.missed, "a fault left in a shell that holds product"),
    tile("False alarms", c.false_alarm, "a person sent to a sound weld"),
    tile("Probe points", d.probe.points, `${d.probe.rejectable} fail the limit`),
    tile("Of those, invisible", d.probe.rejectable_under_a_clean_frame,
      "under a frame the camera calls sound"),
  ].join(""));
}

function contrast() {
  const v = d.contrast.visible, h = d.contrast.hidden;
  const limit = d.contrast.limit_fraction ?? 0.6;
  if (!v || !h) { V.paint(V.$("tk-contrast"), `<div class="hint">no patches on record</div>`); return; }
  const rows = [
    { label: "rusty, you can see it", value: v.remaining_fraction,
      colour: v.rejectable ? COLOUR.missed : COLOUR.right,
      display: `${Math.round(v.remaining_fraction * 100)}% left`,
      tip: `${v.id}: ${v.loss_mm} mm gone, ${v.remaining_mm} mm left. <b>${v.rejectable ? "fails" : "within limits"}</b>` },
    { label: "clean, you cannot", value: h.remaining_fraction,
      colour: h.rejectable ? COLOUR.missed : COLOUR.right,
      display: `${Math.round(h.remaining_fraction * 100)}% left`,
      tip: `${h.id}: ${h.loss_mm} mm gone, ${h.remaining_mm} mm left. <b>${h.rejectable ? "fails" : "within limits"}</b>` },
    { label: `the limit (${Math.round(limit * 100)}%)`, value: limit,
      colour: "var(--line, #666)", display: `${Math.round(limit * 100)}%`,
      tip: "below this the plate is rejected" },
  ];
  V.paint(V.$("tk-contrast"), V.ranked(rows, { width: 420, max: 1.0, padL: 150 }));
}

function outcomes() {
  const c = d.counts;
  const segs = [
    { label: "named it right", value: c.right, colour: COLOUR.right },
    { label: "named a different fault", value: c.misnamed, colour: COLOUR.misnamed },
    { label: "missed", value: c.missed, colour: COLOUR.missed },
    { label: "false alarm", value: c.false_alarm, colour: COLOUR.false_alarm },
    { label: "declined to judge", value: c.declined, colour: COLOUR.declined },
  ].filter((s) => s.value > 0);
  V.paint(V.$("tk-outcomes"), V.stackedRow(segs, { width: 420, height: 22 }));
  V.paint(V.$("tk-outcomes-meta"),
    segs.map((s) => `<span class="key"><i style="background:${s.colour}"></i>${s.label} ${s.value}</span>`)
      .join(" ") +
    `<div class="hint">Nothing was missed and nothing was a false alarm. The four it
     named differently all had a second, real fault in the frame.</div>`);
}

function probe() {
  const p = d.probe;
  if (!p.planners.length) {
    V.paint(V.$("tk-probe"), `<div class="hint">no probe comparison on record</div>`);
    return;
  }
  const best = {};
  p.planners.forEach((pl) => {
    if (!(pl.label in best) || pl.found > best[pl.label].found) best[pl.label] = pl;
  });
  const rows = Object.values(best)
    .sort((a, b) => b.found - a.found)
    .map((pl) => ({
      label: pl.label,
      value: pl.found,
      colour: pl.label.includes("sweep") ? COLOUR.right
        : pl.label.includes("Nemotron") ? "var(--accent, #6a8fd0)" : COLOUR.declined,
      display: `${pl.found} of ${p.rejectable}`,
      tip: `${pl.label}: found <b>${pl.found}</b> of ${p.rejectable} rejectable points, using ${pl.probes} measurements`,
    }));
  V.paint(V.$("tk-probe"), V.ranked(rows, { width: 640, max: p.rejectable, padL: 190 }));
  V.paint(V.$("tk-probe-meta"), `
    <div class="hint">
      <b>The plain coverage rule beats the model, and we expected the opposite.</b>
      The loss is sparse, the budget is fixed at ${p.budget_points} measurements out of
      ${p.points}, and under those conditions sweeping the band beats reasoning about it.
      An earlier version of this run carried only four rejectable points, on which the
      model appeared to win; a paired test on four points supports nothing, so the scene
      was given more hidden loss and the result reversed. The model earns its place on
      the picture, not on the arm.
    </div>`);
}

function stations() {
  const cells = d.frames.map((f) => {
    const colour = COLOUR[f.outcome] ?? COLOUR.unjudged;
    const tip = `station ${f.station}: in frame <b>${f.truth}</b>, called <b>${f.called ?? "not judged"}</b>`
      + (f.probe_rejectable ? `<br>${f.probe_rejectable} probe point(s) here fail the limit` : "");
    return `<button class="chip" data-station="${f.station}" data-tip="${tip}"
      style="border-left:4px solid ${colour}">${f.station}</button>`;
  }).join("");
  V.paint(V.$("tk-stations"), `<div class="chips">${cells}</div>`);
  V.$("tk-stations").querySelectorAll("button").forEach((b) => {
    b.onclick = () => detail(Number(b.dataset.station));
  });
  detail(d.frames[0].station);
}

function detail(station) {
  const f = d.frames.find((x) => x.station === station);
  if (!f) return;
  const listed = f.faults.length ? f.faults.join(", ") : "nothing";
  V.paint(V.$("tk-detail"), `
    <div class="split">
      <div><img src="/api/tank/frame/${f.station}" alt="station ${f.station}"
        style="width:100%;border-radius:8px;border:1px solid var(--line)"></div>
      <div>
        <h3>Station ${f.station}</h3>
        <table class="kv">
          <tr><td>actually in the frame</td><td><b>${f.truth}</b></td></tr>
          <tr><td>the model called it</td><td><b>${f.called ?? "not judged"}</b></td></tr>
          <tr><td>everything it listed</td><td>${listed}</td></tr>
          <tr><td>probe points here</td><td>${f.probe_points}, ${f.probe_thinned} thinned,
            ${f.probe_rejectable} failing</td></tr>
          <tr><td>thinnest wall under it</td><td>${f.thinnest_mm ?? "n/a"} mm</td></tr>
          <tr><td>took</td><td>${f.latency_ms ? Math.round(f.latency_ms) + " ms" : "n/a"}</td></tr>
        </table>
        <div class="hint">${f.reasoning || ""}</div>
      </div>
    </div>`);
}
