import { fmt } from "../api.js";

// Horizontal 0–1 probability bar with a threshold tick.
export default function RiskBar({ value, threshold, compact = false }) {
  const above = threshold !== undefined && value >= threshold;
  return (
    <div className={`riskbar ${compact ? "compact" : ""}`} title={`p = ${fmt(value)}${threshold !== undefined ? `, threshold ${fmt(threshold, 2)}` : ""}`}>
      <div className="track">
        <div className={`fill ${above ? "above" : ""}`} style={{ width: `${Math.max(1, value * 100)}%` }} />
        {threshold !== undefined && <div className="tick" style={{ left: `${threshold * 100}%` }} />}
      </div>
      <span className={`val ${above ? "above" : ""}`}>{fmt(value)}</span>
    </div>
  );
}
