import { useEffect, useState } from "react";
import { api, fmt, modelLabel } from "../api.js";
import RiskBar from "./RiskBar.jsx";

const PAGE = 25;

export default function PatientList({ models }) {
  const [q, setQ] = useState("");
  const [model, setModel] = useState("");
  const [sort, setSort] = useState("id");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState({ total: 0, items: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!model && models.length) setModel(models[0].model_key);
  }, [models, model]);

  useEffect(() => {
    if (!model) return;
    setLoading(true);
    const t = setTimeout(() => {
      api("/patients", { q, model, sort, limit: PAGE, offset })
        .then((d) => {
          setData(d);
          setError(null);
        })
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    }, 200);
    return () => clearTimeout(t);
  }, [q, model, sort, offset]);

  const threshold = models.find((m) => m.model_key === model)?.threshold;

  return (
    <section>
      <div className="controls">
        <input
          type="search"
          placeholder="Search patient ID…"
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setOffset(0);
          }}
          aria-label="Search patient ID"
        />
        <label>
          Risk model{" "}
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m.model_key} value={m.model_key}>
                {modelLabel(m.model_key)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Sort{" "}
          <select
            value={sort}
            onChange={(e) => {
              setSort(e.target.value);
              setOffset(0);
            }}
          >
            <option value="id">Patient ID</option>
            <option value="risk">Highest risk first</option>
          </select>
        </label>
        <span className="muted">{loading ? "Loading…" : `${data.total.toLocaleString()} cohort patients`}</span>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Patient</th>
              <th>Sex</th>
              <th className="num">Age at AKI</th>
              <th>AKI index date</th>
              <th>Risk ({threshold !== undefined ? `threshold ${fmt(threshold, 2)}` : "—"})</th>
              <th>Observed outcome</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((p) => (
              <tr key={p.patient_id} onClick={() => (window.location.hash = `#/patients/${encodeURIComponent(p.patient_id)}`)}>
                <td>
                  <a href={`#/patients/${encodeURIComponent(p.patient_id)}`}>{p.patient_id}</a>
                </td>
                <td>{p.sex || "—"}</td>
                <td className="num">{p.age_at_index ?? "—"}</td>
                <td>{p.aki_index_date ?? "—"}</td>
                <td>
                  {p.probability === null ? "—" : <RiskBar value={p.probability} threshold={threshold} compact />}
                </td>
                <td>{p.outcome === null ? "—" : p.outcome ? "ESRD/dialysis ≤2y" : "No"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
          ← Previous
        </button>
        <span className="muted">
          {data.total ? `${offset + 1}–${Math.min(offset + PAGE, data.total)} of ${data.total}` : ""}
        </span>
        <button disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>
          Next →
        </button>
      </div>
    </section>
  );
}
