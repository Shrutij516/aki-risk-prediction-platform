import { useEffect, useState } from "react";
import { api } from "./api.js";
import PatientList from "./components/PatientList.jsx";
import PatientDetail from "./components/PatientDetail.jsx";
import Models from "./components/Models.jsx";
import Fairness from "./components/Fairness.jsx";

// Minimal hash router: #/patients, #/patients/<id>, #/models, #/fairness
function useRoute() {
  const read = () => (window.location.hash.replace(/^#/, "") || "/patients").split("/").filter(Boolean);
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export default function App() {
  const route = useRoute();
  const [dataset, setDataset] = useState(null);
  const [models, setModels] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api("/dataset"), api("/models")])
      .then(([d, m]) => {
        setDataset(d);
        setModels(m.models);
      })
      .catch((e) => setError(e.message));
  }, []);

  const page = route[0];
  const tab = (name, label) => (
    <a href={`#/${name}`} className={page === name ? "tab active" : "tab"}>
      {label}
    </a>
  );

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <strong>AKI → dialysis risk</strong>
          <span className="sub">Reconstructed research dashboard</span>
        </div>
        <nav>
          {tab("patients", "Patients")}
          {tab("models", "Model results")}
          {tab("fairness", "Fairness")}
        </nav>
      </header>
      {dataset && (
        <div className={`banner ${dataset.kind}`} role="status">
          <strong>{dataset.kind === "synthetic" ? "SYNTHETIC DEMO DATA" : "REAL DATA — LOCAL ONLY"}</strong>
          <span>{dataset.disclaimer}</span>
        </div>
      )}
      {error && <div className="error">Could not reach the API: {error}</div>}
      <main>
        {page === "patients" && !route[1] && <PatientList models={models} />}
        {page === "patients" && route[1] && (
          <PatientDetail pid={decodeURIComponent(route[1])} models={models} datasetKind={dataset?.kind} />
        )}
        {page === "models" && <Models models={models} />}
        {page === "fairness" && <Fairness />}
      </main>
      {dataset && (
        <footer>
          {dataset.counts.patients.toLocaleString()} patients · {dataset.counts.cohort.toLocaleString()} in AKI cohort ·{" "}
          {(
            dataset.counts.diagnosis_events +
            dataset.counts.procedure_events +
            dataset.counts.medication_events +
            dataset.counts.lab_events
          ).toLocaleString()}{" "}
          EHR events in this database
        </footer>
      )}
    </div>
  );
}
