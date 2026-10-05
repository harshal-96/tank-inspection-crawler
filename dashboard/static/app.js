/* The shell: a rail, a top bar, and one page at a time.
 *
 * Each page owns its own markup and its own drawing. The shell only decides which
 * one is on screen, keeps the live rows fresh, and tells whichever page is mounted
 * that they changed. Deep links work, so a finding can be sent to somebody.
 */

import * as V from "./viz.js";
import { state, refreshLive, subscribe } from "./store.js";

import * as overview from "./pages/overview.js";
import * as tank from "./pages/tank.js";
import * as seam from "./pages/seam.js";
import * as routing from "./pages/routing.js";
import * as diagnosis from "./pages/diagnosis.js";
import * as inspection from "./pages/inspection.js";
import * as limits from "./pages/limits.js";
import * as compensation from "./pages/compensation.js";
import * as budget from "./pages/budget.js";
import * as cell from "./pages/cell.js";
import * as parts from "./pages/parts.js";
import * as supervisor from "./pages/supervisor.js";
import * as briefs from "./pages/briefs.js";
import * as method from "./pages/method.js";
import * as platform from "./pages/platform.js";

// The tank first, because it is what the project leads with; then the cell in
// pipeline order. A reader following the rest of the rail top to bottom follows
// a part through the cell, which the old "Study 1..7" numbering did not: thermal
// compensation was Study 6 and happens between the routing call and the weld.
const PAGES = [tank, overview, seam, routing, compensation, inspection, diagnosis,
               briefs, limits, budget, cell, parts, supervisor, method, platform];
// Where the dashboard opens, and where a link to a page that does not exist lands.
const HOME = tank;
const BY_ID = new Map(PAGES.map((p) => [p.meta.id, p]));
const GROUPS = ["Result", "What happens to a part", "What it leaves out",
                "The cell, live", "Reference"];

/** The six stages, in order, for the previous and next links on each one. */
const PIPELINE = ["seam", "routing", "compensation", "inspection", "diagnosis", "briefs"];

let current = null;

// Mounting is asynchronous: a page fetches, then draws into elements its own
// render() put on screen. Navigate away while that is in flight and the elements
// are gone, so the draw throws on a null. Every navigation takes a ticket, and a
// mount that comes back holding a stale one stops rather than writing into a page
// that is no longer there.
let visit = 0;

function rail() {
  const groups = GROUPS.map((g) => {
    const items = PAGES.filter((p) => p.meta.group === g);
    if (!items.length) return "";
    return `<div class="group">${g}</div>` + items.map((p) => `
      <a class="nav" href="#/${p.meta.id}" data-id="${p.meta.id}">
        <svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="${p.meta.icon}"/></svg>
        <span>${V.esc(p.meta.title)}</span>
        ${p.meta.tag ? `<span class="tag">${V.esc(p.meta.tag)}</span>` : ""}
      </a>`).join("");
  }).join("");

  return `
    <div class="brand">
      <div class="mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="#0a0d12" stroke-width="2.4"
             stroke-linecap="round" stroke-linejoin="round"><path d="M4 18 L10 8 L14 14 L20 5"/></svg>
      </div>
      <div>
        <div class="name">Tank Inspection<br>Crawler</div>
        <div class="sub">tank inspection, and the cell it was proved on</div>
      </div>
    </div>
    ${groups}
    <div class="spacer"></div>
    <div class="foot">
      A tank survey of 24 stations, and fifty parts through the cell.<br>
      Every figure is drawn from the run that produced it.
    </div>`;
}

/** Previous and next stage, so a reader can walk the pipeline without the rail. */
function stageNav(page) {
  const i = PIPELINE.indexOf(page.meta.id);
  if (i === -1) return "";
  const link = (id, side) => {
    const p = BY_ID.get(id);
    if (!p) return `<span></span>`;
    return `<a class="stage-link ${side}" href="#/${id}">
      <span class="k">${side === "prev" ? "previous stage" : "next stage"}</span>
      <span class="t">${V.esc(p.meta.title)}</span></a>`;
  };
  return `<nav class="stage-nav">
    ${i > 0 ? link(PIPELINE[i - 1], "prev") : "<span></span>"}
    <span class="of">stage ${i + 1} of ${PIPELINE.length}</span>
    ${i < PIPELINE.length - 1 ? link(PIPELINE[i + 1], "next") : "<span></span>"}
  </nav>`;
}

function topbar(page) {
  return `
    <div>
      <div class="crumb">${V.esc(page.meta.crumb)}</div>
      <h1>${V.esc(page.meta.title)}</h1>
    </div>
    <span class="grow"></span>
    <span class="pill" id="tb-parts"></span>
    <span class="pill"><span class="dot" id="tb-dot"></span><span id="tb-conn">connecting</span></span>
    <button class="pill theme-toggle" id="tb-theme" type="button"
            title="Switch between the light and dark theme"
            aria-label="Switch theme">${themeIcon()}</button>`;
}

/* ------------------------------------------------------------------- theme */

/** What the page is showing. Light unless dark was explicitly chosen. */
function activeTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function themeIcon() {
  // A sun for the light theme it would switch TO, a moon for the dark one.
  return activeTheme() === "dark"
    ? `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" aria-hidden="true">
         <circle cx="12" cy="12" r="4.2"/>
         <path d="M12 2.6v2.2M12 19.2v2.2M2.6 12h2.2M19.2 12h2.2
                  M5.4 5.4l1.6 1.6M17 17l1.6 1.6M18.6 5.4L17 7M7 17l-1.6 1.6"/>
       </svg>`
    : `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
         <path d="M20.5 14.3A8.6 8.6 0 1 1 9.7 3.5a6.9 6.9 0 0 0 10.8 10.8Z"/>
       </svg>`;
}

function flipTheme() {
  const next = activeTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("cell-theme", next);
  } catch (e) { /* storage can be blocked; the choice just will not persist */ }
  const button = V.$("tb-theme");
  if (button) button.innerHTML = themeIcon();
  // The 3D view builds its materials from the tokens, so it has to be told.
  current?.retheme?.();
}

async function route() {
  const id = (location.hash.replace(/^#\/?/, "") || HOME.meta.id).split("?")[0];
  const page = BY_ID.get(id) || HOME;
  if (current === page) return;

  const ticket = ++visit;
  current?.unmount?.();
  current = page;

  document.querySelectorAll(".rail a.nav").forEach((a) =>
    a.classList.toggle("active", a.dataset.id === page.meta.id));
  document.title = `${page.meta.title} · Tank Inspection Crawler`;

  V.$("topbar").innerHTML = topbar(page);
  V.$("tb-theme")?.addEventListener("click", flipTheme);
  const body = V.$("body");
  body.innerHTML = page.render() + stageNav(page);
  body.scrollIntoView({ block: "start" });
  window.scrollTo(0, 0);
  status();

  try {
    await page.mount?.();
  } catch (err) {
    if (ticket !== visit) return;        // navigated away; not this page's problem
    body.insertAdjacentHTML("afterbegin",
      `<div class="card w12"><div class="empty">This page could not be drawn: ${V.esc(err.message)}</div></div>`);
    console.error(err);
  }
}

function status() {
  const dot = V.$("tb-dot"), conn = V.$("tb-conn"), parts = V.$("tb-parts");
  if (!dot) return;
  dot.className = "dot " + (state.connected ? "live" : "down");
  conn.textContent = state.connected ? "live" : (state.error ? "service unreachable" : "connecting");
  if (state.live && parts) {
    parts.textContent = `${state.live.stats.parts_total} parts · ${state.live.seams.length} seams tracked`;
  }
}

(async function start() {
  V.tipInit();
  V.$("rail").innerHTML = rail();
  window.addEventListener("hashchange", route);

  subscribe(() => {
    status();
    const ticket = visit;
    try {
      current?.update?.();
    } catch (err) {
      if (ticket === visit) console.error(err);
    }
  });

  await refreshLive();
  await route();

  setInterval(refreshLive, 8000);
})();
