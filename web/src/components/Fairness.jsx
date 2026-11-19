import { useEffect, useState } from "react";
import { api, fmt, modelLabel, pct } from "../api.js";

const ci = (v) => (v ? `[${fmt(v[0])}, ${fmt(v[1])}]` : "");

export default function Fairness() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [model, setModel] = useState(null);

  useEffect(() => {
    api("/fairness")
      .then((r) => {
        setReport(r);
        setModel(Object.keys(r.models)[0]);
      })
      .catch((e) => setError(e.status === 404 ? "No fairness report has been generated on this machine." : e.message));
  }, []);

  if (error) return <div className="card muted">{error}</div>;
  if (!report) return <p className="muted">Loading…</p>;
  const m = report.models[model];

  return (
    <section>
      <div className="card">
        <h2>Subgroup evaluation</h2>
        <p>
          <strong>{report.data_label}</strong> Aggregate metrics only. Groups with fewer than {report.suppression.min_n}{" "}
          patients or fewer than {report.suppression.min_cell} positives or negatives are suppressed. Threshold per model is the
          validation-selected one; 95% CIs are percentile bootstrap within each group.
        </p>
        <label>
          Model{" "}
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {Object.keys(report.models).map((k) => (
              <option key={k} value={k}>
                {modelLabel(k)}
              </option>
            ))}
          </select>
        </label>
      </div>
      {Object.entries(m.attributes).map(([attr, groups]) => (
        <div className="card" key={attr}>
          <h3>{attr.replace("_", " ")}</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Group</th>
                  <th className="num">n</th>
                  <th className="num">Prevalence</th>
                  <th className="num">ROC-AUC [95% CI]</th>
                  <th className="num">PR-AUC [95% CI]</th>
                  <th className="num">TPR [95% CI]</th>
                  <th className="num">FPR [95% CI]</th>
                  <th className="num">Mean predicted / observed</th>
                </tr>
              </thead>
              <tbody>
                {groups.map((g) =>
                  g.suppressed ? (
                    <tr key={g.group} className="outside">
                      <td>{g.group}</td>
                      <td colSpan={7} className="muted">
                        suppressed ({g.reason})
                      </td>
                    </tr>
                  ) : (
                    <tr key={g.group}>
                      <td>{g.group}</td>
                      <td className="num">{g.n.toLocaleString()}</td>
                      <td className="num">{pct(g.prevalence)}</td>
                      <td className="num">
                        {fmt(g.roc_auc)} {ci(g.ci95.roc_auc)}
                      </td>
                      <td className="num">
                        {fmt(g.pr_auc)} {ci(g.ci95.pr_auc)}
                      </td>
                      <td className="num">
                        {fmt(g.tpr)} {ci(g.ci95.tpr)}
                      </td>
                      <td className="num">
                        {fmt(g.fpr)} {ci(g.ci95.fpr)}
                      </td>
                      <td className="num">
                        {fmt(g.mean_predicted)} / {fmt(g.prevalence)}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        </div>
      ))}
      <div className="card">
        <h3>Limitations</h3>
        <ul>
          {report.limitations.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
