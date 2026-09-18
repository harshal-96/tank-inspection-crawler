/* The shell: a rail, a top bar, and one page at a time.
 *
 * Each page owns its own markup and its own drawing. The shell only decides which
 * one is on screen, keeps the live rows fresh, and tells whichever page is mounted
 * that they changed. Deep links work, so a finding can be sent to somebody.
 */

import * as V from "./viz.js";
import { state, refreshLive, subscribe } from "./store.js";

import * as overview from "./pages/overview.js";
import * as seam from "./pages/seam.js";
import * as routing from "./pages/routing.js";
import * as diagnosis from "./pages/diagnosis.js";
import * as inspection from "./pages/inspection.js";
import * as limits from "./pages/limits.js";
import * as compensation from "./pages/compensation.js";
import * as cell from "./pages/cell.js";
import * as parts from "./pages/parts.js";
import * as supervisor from "./pages/supervisor.js";
import * as briefs from "./pages/briefs.js";
import * as method from "./pages/method.js";
import * as platform from "./pages/platform.js";

const PAGES = [overview, seam, routing, diagnosis, inspection, limits, compensation,
               cell, parts, briefs, supervisor, method, platform];
const BY_ID = new Map(PAGES.map((p) => [p.meta.id, p]));
const GROUPS = ["Result", "Studies", "Cell", "Reference"];

let current = null;

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
        <div class="name">Self-Correcting<br>Production Cell</div>
        <div class="sub">robotic seam welding</div>
      </div>
    </div>
    ${groups}
    <div class="spacer"></div>
    <div class="foot">
      Fifty parts, one thousand measured stations.<br>
      Every figure is drawn from the run that produced it.
    </div>`;
}

function topbar(page) {
  return `
    <div>
      <div class="crumb">${V.esc(page.meta.crumb)}</div>
      <h1>${V.esc(page.meta.title)}</h1>
    </div>
    <span class="grow"></span>
    <span class="pill" id="tb-parts"></span>
    <span class="pill"><span class="dot" id="tb-dot"></span><span id="tb-conn">connecting</span></span>`;
}

async function route() {
  const id = (location.hash.replace(/^#\/?/, "") || "overview").split("?")[0];
  const page = BY_ID.get(id) || overview;
  if (current === page) return;

  current?.unmount?.();
  current = page;

  document.querySelectorAll(".rail a.nav").forEach((a) =>
    a.classList.toggle("active", a.dataset.id === page.meta.id));
  document.title = `${page.meta.title} · Self-Correcting Production Cell`;

  V.$("topbar").innerHTML = topbar(page);
  const body = V.$("body");
  body.innerHTML = page.render();
  body.scrollIntoView({ block: "start" });
  window.scrollTo(0, 0);
  status();

  try {
    await page.mount?.();
  } catch (err) {
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
    current?.update?.();
  });

  await refreshLive();
  await route();

  setInterval(refreshLive, 8000);
})();
