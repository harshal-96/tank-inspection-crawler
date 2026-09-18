/* What to do about the faults the cell found, with the sources it read.
 *
 * The cell can say the joint opened to 5.00 mm at one station on 2 mm sheet. It
 * cannot say what to do about that, because the answer is not in the measurement:
 * it is in a welding code or a machine handbook, it changes, and a model's
 * recollection of it is the least reliable thing in the building. So the fault
 * raises a search, the documents are fetched and read, and the instruction is
 * written from their text with a citation on every claim.
 */

import * as V from "../viz.js";
import { refreshLive, state } from "../store.js";

export const meta = {
  id: "briefs",
  title: "Corrective action",
  crumb: "Live",
  group: "Cell",
  icon: "M9 5h6a2 2 0 012 2v12H7V7a2 2 0 012-2ZM9 5V4a1 1 0 011-1h4a1 1 0 011 1v1M10 11h5M10 15h3",
};

const TRIGGER_LABEL = {
  gap: "raised by a refused station",
  drift: "raised by the shift diagnosis",
  defect: "raised by the post-weld eye",
  supervisor: "asked for by an operator",
};

export function render() {
  return `
    <p class="lede">
      A measurement nobody can act on gets ignored. So when the cell refuses a joint,
      names a drifting machine, or keeps reporting the same defect, it writes the question
      out of its own numbers, searches published welding guidance for the answer, and turns
      that into steps somebody can follow at the machine. Every statement carries a link to
      where it came from, because an operator should not be asked to follow an instruction
      they cannot check.
    </p>

    <div class="tiles" id="br-tiles"></div>

    <div class="grid">
      <div class="card w12">
        <header><h2>How a brief is built</h2></header>
        <div class="hint">Three steps, each one a different job. The measurement already says what is
          wrong, so no model is asked to diagnose. The model's job is to work out what to
          do about it.</div>
        <div id="br-how"></div>
      </div>

      <div class="card w12">
        <header><h2>Raised by this run</h2></header>
        <div id="br-list"></div>
      </div>
    </div>`;
}

const STEPS = [
  { n: 1, name: "The cell writes the question",
    what: "Built out of numbers it measured: a 5.00 mm root opening on 2 mm sheet, a tilt trending 0.05 mm a part.",
    why: "No model is asked to work out what is wrong. The measurement already settled that, and a question built from it cannot drift off the fault.",
    out: "a technical search, unpadded" },
  { n: 2, name: "Tavily finds it and fetches it",
    what: "Search, then extract: the text of the pages, not the snippets the search chose to show.",
    why: "A snippet is enough to decide a page is worth reading and not enough to quote a clause out of it. The cell is about to tell somebody to stop a machine.",
    out: "the documents themselves" },
  { n: 3, name: "Nemotron writes the instruction",
    what: "Steps a person can take at a machine, with [n] against every claim and an explicit refusal where the sources do not answer.",
    why: "A number from the cell is labelled measured; a number from a document is labelled with the document. An operator who cannot tell those apart cannot act on either.",
    out: "a brief, stored with its sources" },
];

export async function mount() {
  await refreshLive();
  V.$("br-how").innerHTML = `<div class="steps">${STEPS.map((s) => `
    <div class="step">
      <div class="n">${String(s.n).padStart(2, "0")}</div>
      <div class="t">${V.esc(s.name)}</div>
      <div class="d">${V.esc(s.what)}</div>
      <div class="w">${V.esc(s.why)}</div>
      <div class="o">→ ${V.esc(s.out)}</div>
    </div>`).join("")}</div>`;
  update();
}

export function update() {
  const briefs = state.live?.briefs || [];
  tiles(briefs);
  list(briefs);
}

function tiles(briefs) {
  const good = briefs.filter((b) => !b.error);
  const docs = good.reduce((a, b) => a + (b.documents_read || 0), 0);
  const sources = good.reduce((a, b) => a + (b.sources?.length || 0), 0);
  const steps = good.reduce((a, b) => a + (b.do_now?.length || 0), 0);
  const seconds = good.reduce((a, b) => a + (b.total_ms || 0), 0) / 1000;

  V.$("br-tiles").innerHTML = [
    tile("Briefs raised", String(good.length), "after", "each one by a measurement, not by a person asking"),
    tile("Documents read", String(docs), "", `from ${sources} sources the search returned`),
    tile("Actions proposed", String(steps), "", "each one a thing to do at a machine"),
    tile("Time to a cited answer", good.length ? `${(seconds / good.length).toFixed(0)}` : "–",
      "", "seconds, search and read and write", "s"),
  ].join("");
}

const tile = (label, value, tone, sub, unit = "") => `<div class="tile">
  <div class="label">${V.esc(label)}</div>
  <div class="value ${tone}">${value}${unit ? `<span class="unit">${unit}</span>` : ""}</div>
  <div class="sub">${V.esc(sub)}</div></div>`;

function list(briefs) {
  const host = V.$("br-list");
  if (!host) return;
  if (!briefs.length) {
    host.innerHTML = `<div class="empty">No briefs yet. They are raised by
      <span class="mono">scripts/raise_briefs.py</span>, or by asking the supervisor
      how to fix something.</div>`;
    return;
  }

  host.innerHTML = briefs.map((b) => {
    if (b.error) {
      return `<div class="card" style="border-top-color:var(--line)">
        <header><h2>${V.esc(b.topic)}</h2></header>
        <div class="empty">This one could not be written: ${V.esc(b.error)}</div></div>`;
    }
    const confidence = (b.confidence || "low").toLowerCase();
    const tone = confidence === "high" ? "good" : confidence === "medium" ? "warn" : "bad";
    return `<div class="card">
      <header>
        <h2>${V.esc(b.headline || b.topic)}</h2>
        <span class="grow" style="flex:1"></span>
        <span class="badge chip ${tone}">confidence ${V.esc(confidence)}</span>
      </header>
      <div class="hint" style="margin-bottom:14px">
        ${V.esc(TRIGGER_LABEL[b.trigger] || b.trigger || "raised by the cell")}
        &middot; read ${b.documents_read || 0} of ${b.sources?.length || 0} sources
        &middot; ${V.fmt((b.total_ms || 0) / 1000, 1)} s
        &middot; written by ${V.esc((b.model || "").split("/").pop() || "–")}
      </div>

      <div class="grid" style="gap:22px 26px">
        <div class="w4" style="min-width:0">
          <div class="eyebrow">What the cell measured</div>
          <div class="d" style="font-size:12.5px;color:var(--text-secondary);line-height:1.7">
            ${V.esc(b.what_the_cell_measured)}</div>
        </div>
        <div class="w4" style="min-width:0">
          <div class="eyebrow">What the sources say</div>
          <div class="d" style="font-size:12.5px;color:var(--text-secondary);line-height:1.7">
            ${cited(b.what_the_sources_say, b.sources)}</div>
        </div>
        <div class="w4" style="min-width:0">
          <div class="eyebrow">Do now</div>
          <ol style="margin:0;padding-left:18px;font-size:12.5px;color:var(--text-primary);
                     line-height:1.7">
            ${(b.do_now || []).map((s) => `<li style="margin-bottom:5px">${cited(s, b.sources)}</li>`).join("")}
          </ol>
        </div>
      </div>

      ${b.caveat ? `<div class="hint" style="margin:16px 0 0">
        <b style="color:var(--warning)">Not settled by the sources.</b> ${V.esc(b.caveat)}</div>` : ""}

      <div style="margin-top:16px;padding-top:13px;border-top:1px solid var(--hair)">
        <div class="eyebrow">Sources</div>
        ${(b.sources || []).map((s) => `<div style="font-size:11.5px;margin-bottom:4px">
          <span class="mono muted">[${V.esc(s.n)}]</span>
          <a href="${V.esc(s.url)}" target="_blank" rel="noopener"
             style="color:var(--series-1);text-decoration:none">${V.esc(s.title || s.url)}</a>
        </div>`).join("")}
      </div>
    </div>`;
  }).join("");
}

/** Turn the [n] markers into something that points at the source list. */
function cited(text, sources) {
  const n = (sources || []).length;
  return V.esc(text || "").replace(/\[(\d+)\]/g, (whole, digits) => {
    const index = Number(digits);
    if (!index || index > n) return whole;
    return `<span class="mono" style="color:var(--series-1)">[${index}]</span>`;
  });
}
