// Thin fetch wrapper for the FastAPI backend (same origin in production, proxied in dev).
export async function api(path, params = {}) {
  const qs = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "")
  ).toString();
  const res = await fetch(`/api${path}${qs ? `?${qs}` : ""}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
export const pct = (x, d = 1) => (x === null || x === undefined ? "—" : `${(100 * x).toFixed(d)}%`);
export const modelLabel = (key) =>
  key
    .replace("retain", "RETAIN")
    .replace("transformer", "Transformer")
    .replace("_withlabs", " · with labs")
    .replace("_nolabs", " · no labs");
