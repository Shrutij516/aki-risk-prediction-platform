import { useEffect, useState } from "react";
import { api, fmt } from "../api.js";

const TYPES = ["diagnosis", "procedure", "medication", "lab"];
const PAGE = 50;

export default function Timeline({ pid }) {
  const [types, setTypes] = useState(TYPES);
  const [obsOnly, setObsOnly] = useState(false);
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (types.length === 0) {
      setData({ total: 0, items: [] });
      return;
    }
    api(`/patients/${encodeURIComponent(pid)}/timeline`, {
      types: types.join(","),
      observation_only: obsOnly,
      limit: PAGE,
      offset,
    })
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => setError(e.message));
  }, [pid, types, obsOnly, offset]);

  const toggle = (t) => {
    setOffset(0);
    setTypes(types.includes(t) ? types.filter((x) => x !== t) : [...types, t]);
  };

  return (
    <div className="card">
      <h3>Event timeline</h3>
      <div className="controls">
        {TYPES.map((t) => (
          <button key={t} className={`chip ${t} ${types.includes(t) ? "on" : ""}`} onClick={() => toggle(t)} aria-pressed={types.includes(t)}>
            {t}
          </button>
        ))}
        <label>
          <input
            type="checkbox"
            checked={obsOnly}
            onChange={(e) => {
              setObsOnly(e.target.checked);
              setOffset(0);
            }}
          />{" "}
          Only model-input window (before blanking)
        </label>
        {data && <span className="muted">{data.total.toLocaleString()} events</span>}
      </div>
      {error && <div className="error">{error}</div>}
      {data && (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Type</th>
                  <th>Code system</th>
                  <th>Code</th>
                  <th>Value</th>
                  <th>Window</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((e, i) => (
                  <tr key={`${offset}-${i}`} className={e.in_observation_window ? "" : "outside"}>
                    <td>{e.date}</td>
                    <td>
                      <span className={`dot ${e.type}`} /> {e.type}
                    </td>
                    <td>{e.code_system || "—"}</td>
                    <td>
                      <code>{e.code}</code>
                    </td>
                    <td>{e.value !== null ? `${fmt(e.value, 2)} ${e.units ?? ""}` : e.units ?? ""}</td>
                    <td className="muted">{e.in_observation_window ? "model input" : "after blanking start"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pager">
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
              ← Newer
            </button>
            <span className="muted">
              {data.total ? `${offset + 1}–${Math.min(offset + PAGE, data.total)} of ${data.total}` : ""}
            </span>
            <button disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>
              Older →
            </button>
          </div>
        </>
      )}
    </div>
  );
}
