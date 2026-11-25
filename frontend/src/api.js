// Small fetch helpers for the FastAPI backend (proxied at /api by vite.config.js).
async function handle(res) {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const getJSON = (path) => fetch(`/api${path}`).then(handle);

export const postJSON = (path, body) =>
  fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(handle);

export const postForm = (path, form) => fetch(`/api${path}`, { method: "POST", body: form }).then(handle);

export const simulate = (scenario, controller, extra = {}) =>
  postJSON("/network/simulate", { scenario, controller, seed: 0, ...extra });

export const CONTROLLER_LABEL = {
  fixed: "Fixed timer (today)",
  webster: "Webster, one junction",
  max_pressure: "Max-Pressure network",
  rl: "AI agent (RL)",
};

export const fmtS = (s) => (s == null ? "–" : s >= 120 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`);
