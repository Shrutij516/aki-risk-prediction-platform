import { useEffect, useState } from "react";
import { api, fmt, modelLabel } from "../api.js";
import RiskBar from "./RiskBar.jsx";
import Timeline from "./Timeline.jsx";

export default function PatientDetail({ pid, models }) {
  const [p, setP] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [risk, setRisk] = useState(null);
  const [riskError, setRiskError] = useState(null);

  useEffect(() => {
    setP(null);
    setRisk(null);
    api(`/patients/${encodeURIComponent(pid)}`)
      .then((d) => {
        setP(d);
        setError(null);
        const retain = d.predictions.find((x) => x.model_key.startsWith("retain")) ?? d.predictions[0];
        setSelected(retain?.model_key ?? null);
      })
      .catch((e) => setError(e.message));
  }, [pid]);

  useEffect(() => {
    if (!selected) return;
    setRisk(null);
    api(`/patients/${encodeURIComponent(pid)}/risk`, { model: selected })
      .then((r) => {
        setRisk(r);
        setRiskError(null);
      })
      .catch((e) => setRiskError(e.message));
  }, [pid, selected]);

  if (error) return <div className="error">{error}</div>;
  if (!p) return <p className="muted">Loading…</p>;

  return (
    <section className="detail">
      <a href="#/patients" className="back">
        ← All patients
      </a>
      <div className="cards">
        <div className="card">
          <h2>{p.patient_id}</h2>
          <dl>
            <dt>Sex</dt>
            <dd>{p.sex || "—"}</dd>
            <dt>Age at AKI</dt>
            <dd>{p.age_at_index ?? "—"}</dd>
            <dt>Race / ethnicity</dt>
            <dd>
              {p.race} / {p.ethnicity}
            </dd>
            <dt>AKI index date</dt>
            <dd>{p.aki_index_date ?? "not in cohort"}</dd>
            <dt>Model inputs before</dt>
            <dd>{p.blank_start_date ?? "—"} (90-day blanking)</dd>
            <dt>Observed outcome</dt>
            <dd>
              {p.in_cohort
                ? p.y_esrd_or_dialysis_2yr
                  ? `ESRD/dialysis within 2 years (${[p.y_esrd_2yr && "ESRD", p.y_dialysis_2yr && "dialysis"].filter(Boolean).join(", ")})`
                  : "No ESRD/dialysis within 2 years"
                : "—"}
            </dd>
            <dt>Events</dt>
            <dd>
              {Object.entries(p.event_counts)
                .map(([k, v]) => `${v} ${k}`)
                .join(" · ")}
            </dd>
          </dl>
        </div>
        <div className="card">
          <h3>Predicted 2-year ESRD/dialysis risk</h3>
          {p.predictions.length === 0 && <p className="muted">No predictions (patient not in the modelled cohort).</p>}
          <table className="risk-table">
            <tbody>
              {p.predictions.map((x) => (
                <tr
                  key={x.model_key}
                  className={x.model_key === selected ? "selected" : ""}
                  onClick={() => setSelected(x.model_key)}
                >
                  <td>{modelLabel(x.model_key)}</td>
                  <td>
                    <RiskBar value={x.probability} threshold={x.threshold} />
                  </td>
                  <td className="muted">{x.above_threshold ? "above threshold" : "below threshold"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="fine">
            Threshold = F1-optimal on the validation set. Click a model to see what it used. {p.disclaimer}
          </p>
        </div>
      </div>

      {selected && (
        <div className="card">
          <h3>{modelLabel(selected)} — live scoring</h3>
          {riskError && <div className="error">{riskError}</div>}
          {!risk && !riskError && <p className="muted">Scoring…</p>}
          {risk && (
            <>
              <p>
                Probability <strong>{fmt(risk.probability)}</strong> (stored batch score {fmt(risk.stored_probability)}) ·
                model saw {risk.n_used} of {risk.n_events} events · {risk.n_oov} unseen codes · source:{" "}
                {risk.model_source}
              </p>
              {risk.attention ? (
                <>
                  <h4>Events with the highest RETAIN attention</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Event</th>
                        <th className="num">Attention α</th>
                      </tr>
                    </thead>
                    <tbody>
                      {risk.attention.map((a) => (
                        <tr key={a.position}>
                          <td>{a.date}</td>
                          <td>
                            <code>{a.token}</code>
                          </td>
                          <td className="num">
                            <span className="alpha" style={{ width: `${Math.min(100, a.alpha * 400)}px` }} />
                            {fmt(a.alpha, 4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : null}
              <p className="fine">{risk.note} Attention shows where the model looked; it is not a causal explanation.</p>
            </>
          )}
        </div>
      )}

      <Timeline pid={pid} />
    </section>
  );
}
