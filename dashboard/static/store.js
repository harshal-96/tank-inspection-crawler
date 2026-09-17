/* One fetch layer for the whole app.
 *
 * Pages ask for what they need and get it from here, so switching sections does
 * not re-download the studies and two pages can never be drawn from two different
 * snapshots. Live rows are refreshed on a timer; the studies are read once,
 * because a completed campaign does not change.
 */

const cache = new Map();
const listeners = new Set();

export const state = {
  live: null,        // stats, parts, seams, schedules, decisions, inspections
  evidence: null,    // the five completed studies
  cell: null,        // the cell's own dimensions
  chosenPart: null,  // which part the detail views are showing
  connected: false,
  error: null,
};

async function json(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
}

/** Fetched once and kept; used for anything that cannot change under us. */
export async function once(key, url) {
  if (!cache.has(key)) cache.set(key, json(url));
  return cache.get(key);
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
function announce() { listeners.forEach((fn) => fn(state)); }

export async function loadEvidence() {
  if (state.evidence) return state.evidence;
  state.evidence = await once("evidence", "/api/evidence");
  return state.evidence;
}

export async function loadCell() {
  if (state.cell) return state.cell;
  state.cell = await once("cell", "/api/cell");
  return state.cell;
}

export async function refreshLive() {
  try {
    const [stats, parts, seams, schedules, decisions, inspections, measurements] =
      await Promise.all([
        json("/api/stats"),
        json("/api/parts?limit=200"),
        json("/api/seam-tracks?limit=200"),
        json("/api/weld-schedules?limit=200"),
        json("/api/view-decisions?limit=200"),
        json("/api/inspections?limit=200"),
        json("/api/measurements?limit=200"),
      ]);
    // Everything indexes by part_id, so the detail views line up by construction.
    const byPart = new Map();
    const put = (rows, key) => rows.forEach((r) => {
      const slot = byPart.get(r.part_id) || { part_id: r.part_id };
      if (!(key in slot)) slot[key] = r;   // newest wins, the API sorts descending
      byPart.set(r.part_id, slot);
    });
    put(seams, "seam");
    put(schedules, "schedule");
    put(decisions, "decision");
    put(inspections, "inspection");
    put(measurements, "measurement");

    state.live = {
      stats, parts, seams, schedules, decisions, inspections, measurements,
      byPart,
      ids: [...byPart.keys()].sort((a, b) => b - a),
    };
    state.connected = true;
    state.error = null;
    if (state.chosenPart === null || !byPart.has(state.chosenPart)) {
      state.chosenPart = state.live.ids[0] ?? null;
    }
  } catch (err) {
    state.connected = false;
    state.error = String(err.message || err);
  }
  announce();
  return state.live;
}

export function chosen() {
  if (!state.live || state.chosenPart === null) return null;
  return state.live.byPart.get(state.chosenPart) ?? null;
}

export function choose(partId) {
  state.chosenPart = Number(partId);
  announce();
}
