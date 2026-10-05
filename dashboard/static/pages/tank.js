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

const WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight",
               "nine", "ten"];
const word = (n) => WORDS[n] ?? String(n);
const plural = (n, one, many) => (n === 1 ? one : many);
// One decimal, dropped when it is zero: the worst hidden patch has 34.5% of its
// wall left, and rounding it to 35% would put a number on this page that no
// document says.
const pc = (f) => `${(f * 100).toFixed(1).replace(/\.0$/, "")}%`;

let d = null;

export function render() {
  return `
    <p class="lede">
      A robot crawls a weld seam, the line where two steel plates are welded together,
      inside a steel tank. A camera judges the seam, an
      ultrasonic probe measures what is left of the plate, and something has to decide
      where to spend the next measurement. Every number here is scored against answers
      worked out from the scene file. These frames are the development set: the scene was
      corrected after reading the model's first answers, and the hidden patches were placed
      after the first probe result.
    </p>

    <div class="tiles" id="tk-tiles"></div>

    <div class="grid">
      <div class="card w6">
        <header><h2>The argument: what you can see is not what fails</h2></header>
        <div class="hint" id="tk-contrast-hint"></div>
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
        <div class="hint" id="tk-probe-hint"></div>
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
      `<div class="hint">${V.esc(d?.why ?? "The tank run is not available.")}</div>`);
    return;
  }
  tiles();
  contrast();
  outcomes();
  probe();
  stations();
}

function tile(label, value, note) {
  return `<div class="tile"><div class="label">${V.esc(label)}</div>
    <div class="value">${value}</div><div class="sub">${V.esc(note ?? "")}</div></div>`;
}

function tiles() {
  const c = d.counts, g = d.gate, p = d.probe, f = d.faults_on_the_run;
  // The gate is over the frames it judged. A frame it declined is a frame a
  // person has to open, which is neither right nor wrong, so it is shown beside
  // the score and never counted into it.
  const aside = [];
  if (g.declined) aside.push(`${g.declined} declined`);
  if (g.slivers) aside.push(`${g.slivers} called on a sliver`);
  const out = [
    tile("Pass or defect", `${g.right}/${g.judged}`,
      "the call that sends somebody in" + (aside.length ? `; ${aside.join(", ")}` : "")),
  ];
  if (g.strict_right !== undefined) {
    out.push(tile("Defect call for a fault in view", `${g.strict_right}/${g.judged}`,
      "the call, and what it named was really there"));
  }
  out.push(
    tile("Named the worst fault", `${c.right}/${g.judged}`, "the line on the work order"),
    tile("Faults on the run named", `${f.named}/${f.total}`, "in at least one frame that held them"),
    tile("Missed", c.missed, "a fault left in a shell that holds product"),
    tile("False alarms", c.false_alarm, "a person sent to a sound weld"),
    tile("Probe points", p.points, `${p.rejectable} fail the limit`),
    tile("Failing, under a pass", p.rejectable_under_a_pass,
      "under a frame the inspector passed"),
    tile("Failing, nothing visible", p.rejectable_under_a_clean_frame,
      "under a frame with no visible fault at all"),
  );
  V.paint(V.$("tk-tiles"), out.join(""));
}

function contrast() {
  const k = d.contrast, v = k.visible, h = k.hidden;
  const limit = k.limit_fraction ?? 0.6;
  if (!v || !h) {
    V.paint(V.$("tk-contrast-hint"), "");
    V.paint(V.$("tk-contrast"), `<div class="hint">no patches on record</div>`);
    return;
  }
  // Every clause is read off the scene, and the closing one is only said when
  // it holds: a scene where a visible patch failed would not get that line.
  const verdict = (p) => (p.rejectable ? "fails" : "is within limits");
  const seen = k.visible_count === 1
    ? `One patch of wall loss on this run shows as rust on the surface, and a camera
       finds it at once. It has ${pc(v.remaining_fraction)} of its wall left and ${verdict(v)}.`
    : `${word(k.visible_count)} patches of wall loss on this run show as rust on the
       surface. The thinnest has ${pc(v.remaining_fraction)} of its wall left and ${verdict(v)}.`;
  const unseen = `The other ${word(k.hidden_count)} ${plural(k.hidden_count, "leaves", "leave")}
    no mark on the surface. The worst has ${pc(h.remaining_fraction)} left, and
    ${k.hidden_failing === k.hidden_count ? (k.hidden_count === 1 ? "it fails" : `all ${word(k.hidden_count)} fail`)
      : `${word(k.hidden_failing)} of them ${plural(k.hidden_failing, "fails", "fail")}`}
    the ${pc(limit)} screening limit set for this run.`;
  // The scene was built to hold this premise, so it is said to be one: what is
  // measured is how each planner copes with it, not whether it is true.
  const close = k.visible_failing === 0 && k.hidden_failing > 0
    ? " Every patch that fails is one a camera cannot see. The scene is built that way on"
      + " purpose, to encode the premise; what is measured is how each planner copes with it."
    : "";
  V.paint(V.$("tk-contrast-hint"), `${seen} ${unseen}${close}`);

  const rows = [
    { label: "rusty, you can see it", value: v.remaining_fraction,
      colour: v.rejectable ? COLOUR.missed : COLOUR.right,
      display: `${pc(v.remaining_fraction)} left`,
      tip: `${V.esc(v.id)}: ${v.loss_mm} mm gone, ${v.remaining_mm} mm left. <b>${v.rejectable ? "fails" : "within limits"}</b>` },
    { label: "clean, you cannot (worst)", value: h.remaining_fraction,
      colour: h.rejectable ? COLOUR.missed : COLOUR.right,
      display: `${pc(h.remaining_fraction)} left`,
      tip: `${V.esc(h.id)}: ${h.loss_mm} mm gone, ${h.remaining_mm} mm left. <b>${h.rejectable ? "fails" : "within limits"}</b>` },
    { label: `screening limit (${pc(limit)})`, value: limit,
      colour: "var(--line, #666)", display: pc(limit),
      tip: "below this the plate is rejected on this run" },
  ];
  V.paint(V.$("tk-contrast"), V.ranked(rows, { width: 420, max: 1.0, padL: 160 }));
}

/** The fault in a frame, and the other frames on the run that named it. */
function namedElsewhere(station) {
  const hit = (d.faults_on_the_run.faults || []).find(
    (x) => x.frames.includes(station) && x.named_in.some((s) => s !== station));
  return hit ? hit.named_in.filter((s) => s !== station) : [];
}

const stationList = (xs) => (xs.length === 1 ? `station ${xs[0]}`
  : `stations ${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

function outcomes() {
  const c = d.counts, g = d.gate;
  const segs = [
    { label: "named it right", value: c.right, colour: COLOUR.right },
    { label: "named a different fault", value: c.misnamed, colour: COLOUR.misnamed },
    { label: "missed", value: c.missed, colour: COLOUR.missed },
    { label: "false alarm", value: c.false_alarm, colour: COLOUR.false_alarm },
    { label: "called on a sliver", value: c.sliver, colour: COLOUR.sliver },
    { label: "declined to judge", value: c.declined, colour: COLOUR.declined },
    { label: "not judged", value: c.unjudged, colour: COLOUR.unjudged },
  ].filter((s) => s.value > 0);
  V.paint(V.$("tk-outcomes"), V.stackedRow(segs, { width: 420, height: 22 }));

  // Built from the counts, and a sentence is only said when it is true.
  const said = [];
  if (!c.missed && !c.false_alarm) {
    said.push("Nothing was missed and nothing was a false alarm.");
  } else {
    d.frames.filter((f) => f.outcome === "missed").forEach((f) => {
      const other = namedElsewhere(f.station);
      said.push(`It passed station ${f.station}, which holds a ${V.esc(f.truth)}`
        + (other.length ? `, the same one it had named at ${stationList(other)}.` : "."));
    });
    if (c.false_alarm) {
      said.push(`It called ${c.false_alarm} sound ${plural(c.false_alarm, "frame", "frames")} a defect.`);
    }
  }
  if (c.misnamed) {
    const bits = [];
    if (g.named_a_lesser_fault_there) {
      bits.push(`${g.named_a_lesser_fault_there} named a lesser fault that really was in the frame`);
    }
    if (g.named_only_a_sliver) {
      bits.push(`${g.named_only_a_sliver} named only a sliver at the edge of the frame and not
        the fault in front of the lens`);
    }
    if (bits.length) {
      said.push(`Of the ${c.misnamed} it named differently, ${bits.join(", and ")}.`);
    }
  }
  (d.faults_on_the_run.faults || []).filter((x) => !x.named_in.length).forEach((x) => {
    said.push(`The ${V.esc(x.kind)} at ${stationList(x.frames)} was never named in any frame
      that held it.`);
  });
  if (g.strict_right !== undefined && g.strict_right !== g.right) {
    said.push(`Pass or defect credits a defect call whenever a fault is in view; the strict
      count, ${g.strict_right} of ${g.judged}, also asks that what it named was one of them.`);
  }
  if (c.declined) said.push(`It declined ${c.declined}, each one a frame a person has to open.`);

  V.paint(V.$("tk-outcomes-meta"),
    segs.map((s) => `<span class="key"><i style="background:${s.colour}"></i>${s.label} ${s.value}</span>`)
      .join(" ") + (said.length ? `<div class="hint">${said.join(" ")}</div>` : ""));
}

function probe() {
  const p = d.probe;
  const perStation = d.stations ? Math.round(p.points / d.stations) : null;
  const intro = [];
  if (perStation) {
    intro.push(`The arm can reach ${word(perStation)} places at every station and there is
      time for ${word(p.budget)}.`);
  }
  intro.push(`${p.rejectable_under_a_pass} of the ${p.rejectable} points that fail the limit sit
    under a frame the inspector passed, and ${p.rejectable_under_a_clean_frame} under a frame with
    no visible fault at all.`);
  if (p.rejectable_on_a_visible_patch === 0 && p.rejectable) {
    intro.push("None is on the patch the camera can see.");
  }
  if (p.rejectable_under_a_pass * 2 > p.rejectable) {
    intro.push("That is the reason this machine carries two instruments.");
  }
  V.paint(V.$("tk-probe-hint"), intro.join(" "));

  if (!p.planners.length) {
    V.paint(V.$("tk-probe"), `<div class="hint">no probe comparison on record</div>`);
    return;
  }
  // The coverage rule the comparison is about. Other arithmetic arms are drawn
  // too, as context, and are not it.
  const rule = p.planners.find((pl) => pl.label === "sweep (no model)")
    || p.planners.find((pl) => !pl.uses_a_model && pl.label.startsWith("sweep"));
  const range = (pl) => (pl.low === pl.high ? `${pl.low}` : `${pl.low} to ${pl.high}`);
  const runs = (pl) => (pl.runs > 1 ? `, ${pl.runs} runs` : "");
  // Drawn at the worst attempt, with the range beside it: a model that came back
  // with different picks on different runs is shown by all of them, not its best.
  const rows = [...p.planners]
    .sort((a, b) => b.low - a.low)
    .map((pl) => ({
      label: pl.label,
      value: pl.low,
      colour: pl === rule ? COLOUR.right
        : pl.uses_a_model ? "var(--accent, #6a8fd0)" : COLOUR.declined,
      display: `${range(pl)} of ${p.rejectable}${runs(pl)}`,
      tip: `${V.esc(pl.label)}: found <b>${range(pl)}</b> of ${p.rejectable} failing points`
        + ` over ${pl.runs} ${plural(pl.runs, "run", "runs")}, reaching `
        + `${pl.patches_low === pl.patches_high ? pl.patches_low : `${pl.patches_low} to ${pl.patches_high}`}`
        + ` of ${p.patches_with_failing_points} failing patches, using ${pl.probes} measurements`,
    }));
  if (p.best_possible != null) {
    rows.unshift({
      label: "best possible", value: p.best_possible, colour: "var(--line, #666)",
      display: `${p.best_possible} of ${p.rejectable}`,
      tip: `${p.budget} probes a station, choosing with the answers known: `
        + `<b>${p.best_possible}</b> of ${p.rejectable}. A ceiling, not a planner`,
    });
  }
  if (p.chance) {
    rows.push({
      label: "random picks", value: p.chance.mean, colour: "var(--line, #666)",
      display: `${p.chance.mean} expected, ${p.chance.low} to ${p.chance.high}`,
      tip: `picking ${p.budget} of the reachable places at random at every station: `
        + `<b>${p.chance.mean}</b> expected, ${p.chance.low} to ${p.chance.high} in the `
        + `${V.esc(p.chance.band)}`,
    });
  }
  V.paint(V.$("tk-probe"), V.ranked(rows, { width: 640, max: p.rejectable, padL: 190 }));

  // The verdict on the planner, said only as far as the runs on record carry it.
  const models = p.planners.filter((pl) => pl.uses_a_model);
  const said = [];
  let ruleWon = false;
  if (rule && models.length) {
    const top = Math.max(...models.map((m) => m.high));
    const low = Math.min(...models.map((m) => m.low));
    const runsAll = models.reduce((a, m) => a + m.runs, 0);
    ruleWon = rule.low > top;
    const modelRange = low === top ? `${low}` : `${low} to ${top}`;
    if (ruleWon) {
      // Said as what the runs show, not as the rule beating the model: the
      // rule's count is one tie-break order, and no run differs significantly.
      said.push(`<b>The model was never ahead of the plain coverage rule in any of the
        ${runsAll} model runs on record, and we expected the opposite.</b>`);
    } else if (rule.high < low) {
      said.push(`<b>The model found more than the coverage rule on every run on record.</b>`);
    } else {
      said.push(`<b>The coverage rule and the model overlap across the runs on record.</b>`);
    }
    said.push(`At its shipped tie-break order the rule found ${rule.low} and the model
      ${modelRange}, with the budget fixed at ${p.budget_points} measurements out of
      ${p.points}.`);
    if (p.chance && low >= p.chance.low && top <= p.chance.high) {
      said.push(top === p.chance.high
        ? `Every model run sits inside the band random picks land in
          (${p.chance.low} to ${p.chance.high}), the best of them at its top edge, so these runs
          cannot tell the model's picks from chance with any confidence.`
        : `Every model run sits inside the band random picks land in
          (${p.chance.low} to ${p.chance.high}), so these runs cannot tell the model's picks
          from chance.`);
    }
    if (p.chance && rule.low > p.chance.high) {
      said.push(`The rule's ${rule.low} is above that band.`);
    }
    // How typical the rule's count is. It is one order of its tie-break, so the
    // study over every order says what the rule usually finds, with its full
    // history and when shown only what the model's prompt carries. An earlier
    // version of this paragraph read the second arm at its shipped order alone
    // and put the gap down to the history the rule reads; over all orders,
    // withholding that history does not hurt the rule.
    const t = p.tiebreaks;
    const matched = p.planners.find((pl) => pl.label === "sweep, model's view");
    if (t && t.shipped_found === rule.low) {
      said.push(`That ${rule.low} is one tie-break order out of ${t.orders}. Over all of them
        the same rule finds ${t.min} to ${t.max}, median ${t.median}, and
        ${pc(t.share_below_shipped)} of the orders find fewer than the shipped one.`);
      const mv = t.models_view;
      if (matched && mv) {
        said.push(`Shown only the state the model's prompt carries, it finds
          ${range(matched)} at the shipped order and ${mv.min} to ${mv.max}, median
          ${mv.median}, over all orders${mv.median >= t.median
            ? ", so withholding the state it keeps does not hurt it" : ""}.`);
        const typical = [t.median, mv.median].sort((a, b) => a - b);
        said.push(`So the typical rule finds ${typical[0] === typical[1] ? typical[0]
          : `${typical[0]} to ${typical[1]}`} against the model's ${modelRange}.`);
      }
    } else {
      said.push(`That ${rule.low} is one order of the rule's tie-break: tank/score_probes.py
        --tiebreaks scores every order.`);
    }
    // The paired test, read per run from the API (the same exact McNemar as
    // tank/significance.py) and said only as far as it holds: a run that was
    // not tested leaves the sentence out rather than letting it stand.
    const ps = models.map((m) => m.paired_p);
    if (ps.length && ps.every(Boolean)) {
      const pmin = Math.min(...ps.map((r) => r[0]));
      const pmax = Math.max(...ps.map((r) => r[1]));
      const fmt = (x) => x.toFixed(3);
      const spread = pmin === pmax ? fmt(pmin) : `${fmt(pmin)} to ${fmt(pmax)}`;
      const below = models.flatMap((m) => m.attempts)
        .filter((a) => a.paired_p !== null && a.paired_p < 0.05).length;
      said.push(below === 0
        ? `No run on record differs from the rule by a paired exact test at 0.05 (p
          ${spread}, one per run, tank/significance.py).`
        : `${below} of the ${runsAll} runs ${below === 1 ? "differs" : "differ"} from the rule by a paired exact test at
          0.05 (p ${spread}, one per run, tank/significance.py).`);
    }
    said.push(`The hidden patches were also placed by hand, in a layout that favours
      coverage; tank/layout_sweep.py scores the rule on other layouts.`);
  }
  said.push(`An earlier version of this run carried only four failing points, on which the
    model appeared to win. A paired test on four points supports nothing, so the scene was
    given more hidden loss, placed by hand after that result, and on the new scene the model
    was not ahead in any run.`);
  // Which model does which job, named from the records: the inspector from the
  // verdict rows, the planner from the probe runs. They are different models,
  // and the planner is not the one that looks at the picture.
  const inspector = (d.inspector || []).map((m) => m.replace(/-/g, " ")).join(" and ");
  const planner = [...new Set(models.map((m) => m.label.replace(/\s*\(.*\)$/, "")))]
    .join(" and ");
  const jobs = [];
  if (inspector) jobs.push(`The picture is judged by ${V.esc(inspector)}.`);
  if (planner) {
    jobs.push(ruleWon
      ? `${V.esc(planner)} was measured as the probe planner and is reported here as a
        negative result. Arithmetic drives the arm because on this evidence it is no worse,
        and it is deterministic, free and instant.`
      : `${V.esc(planner)} was measured as the probe planner, beside the coverage rule.`);
  }
  said.push(...jobs);
  V.paint(V.$("tk-probe-meta"), `<div class="hint">${said.join(" ")}</div>`);
}

function stations() {
  const cells = d.frames.map((f) => {
    const colour = COLOUR[f.outcome] ?? COLOUR.unjudged;
    const tip = `station ${f.station}: in frame <b>${V.esc(f.truth)}</b>, called `
      + `<b>${V.esc(f.called ?? "not judged")}</b>`
      + (f.probe_rejectable ? `<br>${f.probe_rejectable} probe point(s) here fail the limit` : "");
    return `<button class="chip" data-station="${f.station}" data-tip="${V.esc(tip)}"
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
  // Everything the model wrote is shown as text, never as markup.
  const listed = f.faults.length ? f.faults.map(V.esc).join(", ") : "nothing";
  V.paint(V.$("tk-detail"), `
    <div class="split">
      <div><img src="/api/tank/frame/${f.station}" alt="station ${f.station}"
        style="width:100%;border-radius:8px;border:1px solid var(--line)">
        <div class="hint">Weld bead exaggerated for visibility, about 2.8 times the API 650
          reinforcement limit.</div></div>
      <div>
        <h3>Station ${f.station}</h3>
        <table class="kv">
          <tr><td>actually in the frame</td><td><b>${V.esc(f.truth)}</b></td></tr>
          <tr><td>the model called it</td><td><b>${V.esc(f.called ?? "not judged")}</b></td></tr>
          <tr><td>everything it listed</td><td>${listed}</td></tr>
          <tr><td>probe points here</td><td>${f.probe_points}, ${f.probe_thinned} thinned,
            ${f.probe_rejectable} failing</td></tr>
          <tr><td>thinnest wall under it</td><td>${f.thinnest_mm ?? "n/a"} mm</td></tr>
          <tr><td>took</td><td>${f.latency_ms ? Math.round(f.latency_ms) + " ms" : "n/a"}</td></tr>
        </table>
        <div class="hint">${V.esc(f.reasoning || "")}</div>
      </div>
    </div>`);
}
