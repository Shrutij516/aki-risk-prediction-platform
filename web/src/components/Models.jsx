import { fmt, modelLabel, pct } from "../api.js";

// Dot + 95% CI chart of test ROC-AUC (reconstructed) with the original published point.
function AucChart({ models }) {
  const rows = models.filter((m) => m.test?.roc_auc !== undefined);
  const lo = 0.55,
    hi = 0.72,
    W = 560,
    left = 190,
    rowH = 34,
    H = rows.length * rowH + 40;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - 20);
  const ticks = [0.55, 0.6, 0.65, 0.7];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Test ROC-AUC with 95% confidence intervals">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={10} y2={H - 25} className="grid" />
          <text x={x(t)} y={H - 8} textAnchor="middle" className="axis">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {rows.map((m, i) => {
        const y = 24 + i * rowH;
        const ci = m.test.ci95?.roc_auc;
        return (
          <g key={m.model_key}>
            <text x={left - 10} y={y + 4} textAnchor="end" className="label">
              {modelLabel(m.model_key)}
            </text>
            {ci && <line x1={x(ci[0])} x2={x(ci[1])} y1={y} y2={y} className="ci" />}
            <circle cx={x(m.test.roc_auc)} cy={y} r={5} className="point" />
            {m.original_published && (
              <rect x={x(m.original_published.roc_auc) - 4} y={y - 4} width={8} height={8} className="orig" />
            )}
          </g>
        );
      })}
    </svg>
  );
}

export default function Models({ models }) {
  if (!models.length) return <p className="muted">Loading…</p>;
  const real = models.some((m) => m.source === "real-trained");
  return (
    <section>
      <div className="card">
        <h2>Model performance</h2>
        <p>
          {real ? (
            <>
              <strong>Real-data held-out test set</strong> results of the reconstructed PyTorch models (patient-level 64/16/20
              split, training-only vocabulary, hyperparameters and threshold chosen on validation only, test scored once).
              These are <strong>reconstructed results, not exact reproductions</strong> of the original Keras notebooks.
            </>
          ) : (
            <>These models were trained on synthetic data only; their metrics say nothing about real-world performance.</>
          )}
        </p>
        <AucChart models={models} />
        <p className="fine">
          ● reconstructed test ROC-AUC with 95% bootstrap CI · ■ original published value (Keras; threshold and
          hyperparameters were tuned on the test set).
        </p>
      </div>
      <div className="card">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Selected α / dropout</th>
                <th className="num">Threshold</th>
                <th className="num">Val ROC-AUC</th>
                <th className="num">Test ROC-AUC [95% CI]</th>
                <th className="num">Test PR-AUC [95% CI]</th>
                <th className="num">F1</th>
                <th className="num">Recall</th>
                <th className="num">Precision</th>
                <th>TN / FP / FN / TP</th>
                <th className="num">Test prevalence</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const t = m.test || {};
                const ci = t.ci95 || {};
                return (
                  <tr key={m.model_key}>
                    <td>{modelLabel(m.model_key)}</td>
                    <td>
                      {m.focal_alpha} / {m.dropout}
                    </td>
                    <td className="num">{fmt(m.threshold, 2)}</td>
                    <td className="num">{fmt(m.validation?.roc_auc)}</td>
                    <td className="num">
                      {fmt(t.roc_auc)} {ci.roc_auc && `[${fmt(ci.roc_auc[0])}, ${fmt(ci.roc_auc[1])}]`}
                    </td>
                    <td className="num">
                      {fmt(t.pr_auc)} {ci.pr_auc && `[${fmt(ci.pr_auc[0])}, ${fmt(ci.pr_auc[1])}]`}
                    </td>
                    <td className="num">{fmt(t.f1)}</td>
                    <td className="num">{fmt(t.sensitivity)}</td>
                    <td className="num">{fmt(t.precision)}</td>
                    <td>{t.tn !== undefined ? `${t.tn} / ${t.fp} / ${t.fn} / ${t.tp}` : "—"}</td>
                    <td className="num">{t.n ? `${pct(t.positives / t.n)} (${t.positives}/${t.n})` : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      <div className="card">
        <h3>Original published results (Keras, README)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">ROC-AUC</th>
                <th className="num">PR-AUC</th>
                <th className="num">F1</th>
                <th className="num">Recall</th>
                <th className="num">Precision</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              {models
                .filter((m) => m.original_published)
                .map((m) => {
                  const o = m.original_published;
                  return (
                    <tr key={m.model_key}>
                      <td>{modelLabel(m.model_key)}</td>
                      <td className="num">{fmt(o.roc_auc)}</td>
                      <td className="num">{fmt(o.pr_auc)}</td>
                      <td className="num">{fmt(o.f1)}</td>
                      <td className="num">{fmt(o.sensitivity)}</td>
                      <td className="num">{fmt(o.precision)}</td>
                      <td className="fine">{o.note}</td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
