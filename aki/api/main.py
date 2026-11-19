"""FastAPI backend for the AKI dashboard.

    uvicorn aki.api.main:app --host 127.0.0.1 --port 8000

Configuration (environment):
  AKI_APP_DSN      database to serve (default: local server, database 'aki_demo')
  AKI_ALLOW_REAL   must be '1' to serve a database marked REAL in app.dataset_info
  AKI_MODELS_DIR   model bundles for live scoring (default: chosen from the registered models' source)
  AKI_FAIRNESS     aggregate fairness report JSON (default reports/fairness/fairness_report.json)

Every response carries the dataset kind (header X-AKI-Dataset) so synthetic demo
predictions can never be mistaken for the real-data evaluation.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from psycopg.rows import dict_row

from aki.eval.original_results import ORIGINAL
from aki.serve.registry import load_all

ROOT = Path(__file__).resolve().parents[2]
EVENT_TYPES = ("diagnosis", "procedure", "medication", "lab")
DISCLAIMER = {
    "synthetic": "Synthetic demo patients. Scores show the pipeline working and have no clinical meaning.",
    "real": "Real patient data (local only). Research model; not validated for clinical use.",
}


def _default_dsn() -> str:
    from aki.db.connection import dsn_for_database, get_dsn

    return dsn_for_database(get_dsn(), "aki_demo")


def create_app(dsn: str | None = None, models_dir: Path | None = None, fairness_path: Path | None = None,
               allow_real: bool | None = None, web_dist: Path | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["dsn"] = dsn or os.environ.get("AKI_APP_DSN") or _default_dsn()
        with psycopg.connect(state["dsn"], row_factory=dict_row) as c:
            info = c.execute("SELECT kind, description FROM app.dataset_info").fetchone()
            sources = {r["source"] for r in c.execute("SELECT DISTINCT source FROM app.model")}
        if info is None:
            raise RuntimeError("app.dataset_info is empty: build the database with `python -m aki.demo.build` first")
        allowed = allow_real if allow_real is not None else os.environ.get("AKI_ALLOW_REAL") == "1"
        if info["kind"] == "real" and not allowed:
            raise RuntimeError("database is marked REAL; set AKI_ALLOW_REAL=1 to serve it (local use only)")
        state["dataset"] = info
        mdir = models_dir or (Path(os.environ["AKI_MODELS_DIR"]) if os.environ.get("AKI_MODELS_DIR") else None)
        if mdir is None:
            mdir = ROOT / "outputs" / ("models" if sources == {"real-trained"} else "models_synthetic")
        state["models"] = load_all(mdir)
        state["fairness_path"] = fairness_path or Path(os.environ.get(
            "AKI_FAIRNESS", ROOT / "reports" / "fairness" / "fairness_report.json"))
        yield

    app = FastAPI(title="AKI dialysis-risk dashboard API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def dataset_header(request: Request, call_next):
        response = await call_next(request)
        if "dataset" in state:
            response.headers["X-AKI-Dataset"] = state["dataset"]["kind"]
        return response

    def db():
        with psycopg.connect(state["dsn"], row_factory=dict_row) as conn:
            yield conn

    def _patient_or_404(conn, pid: str) -> dict:
        row = conn.execute("""
            SELECT p.patient_id, p.sex, p.race, p.ethnicity, p.patient_regional_location AS region,
                   CASE WHEN p.year_of_birth IS NULL OR c.aki_index_date IS NULL THEN NULL
                        ELSE extract(year FROM c.aki_index_date)::int - p.year_of_birth END AS age_at_index,
                   c.aki_index_date, c.blank_start_date, c.prediction_end_date,
                   c.y_esrd_2yr, c.y_dialysis_2yr, c.y_esrd_or_dialysis_2yr,
                   (c.patient_id IS NOT NULL) AS in_cohort
            FROM ehr.patient p LEFT JOIN cohort.aki_cohort c USING (patient_id)
            WHERE p.patient_id = %s""", (pid,)).fetchone()
        if row is None:
            raise HTTPException(404, "patient not found")
        return row

    # ------------------------------------------------------------------ endpoints
    @app.get("/api/health")
    def health(conn=Depends(db)):
        conn.execute("SELECT 1")
        return {"status": "ok", "dataset_kind": state["dataset"]["kind"], "models_loaded": sorted(state["models"])}

    @app.get("/api/dataset")
    def dataset(conn=Depends(db)):
        counts = conn.execute("""
            SELECT (SELECT count(*) FROM ehr.patient) AS patients,
                   (SELECT count(*) FROM cohort.aki_cohort) AS cohort,
                   (SELECT count(*) FILTER (WHERE y_esrd_or_dialysis_2yr) FROM cohort.aki_cohort) AS cohort_positive,
                   (SELECT count(*) FROM ehr.diagnosis) AS diagnosis_events,
                   (SELECT count(*) FROM ehr.procedure) AS procedure_events,
                   (SELECT count(*) FROM ehr.medication_drug) AS medication_events,
                   (SELECT count(*) FROM ehr.lab_result) AS lab_events""").fetchone()
        return {"kind": state["dataset"]["kind"], "description": state["dataset"]["description"],
                "disclaimer": DISCLAIMER[state["dataset"]["kind"]], "counts": counts}

    @app.get("/api/models")
    def models(conn=Depends(db)):
        rows = conn.execute("SELECT * FROM app.model ORDER BY model, variant").fetchall()
        out = []
        for r in rows:
            original = ORIGINAL.get((r["model"], r["variant"]))
            out.append({
                "model_key": r["model_key"], "model": r["model"], "variant": r["variant"], "source": r["source"],
                "mlflow_run_id": r["mlflow_run_id"], "threshold": r["threshold"], "focal_alpha": r["focal_alpha"],
                "dropout": r["dropout"], "maxlen": r["maxlen"], "truncate": r["truncate"], "split": r["split"],
                "validation": r["val_metrics"], "test": r["test_metrics"],
                "evaluation_label": ("Real-data held-out test set (reconstructed PyTorch model, validation-only selection)"
                                     if r["source"] == "real-trained" else "Synthetic data only"),
                "original_published": original, "loaded_for_live_scoring": r["model_key"] in state["models"]})
        return {"dataset_kind": state["dataset"]["kind"], "models": out}

    @app.get("/api/patients")
    def patients(q: str = "", in_cohort: bool = True, model: str | None = None,
                 sort: str = Query("id", pattern="^(id|risk)$"), limit: int = Query(25, ge=1, le=200),
                 offset: int = Query(0, ge=0), conn=Depends(db)):
        if sort == "risk" and not model:
            raise HTTPException(422, "sort=risk needs model=<model_key>")
        where = ["p.patient_id ILIKE %(q)s"]
        if in_cohort:
            where.append("c.patient_id IS NOT NULL")
        order = "pr.probability DESC NULLS LAST, p.patient_id" if sort == "risk" else "p.patient_id"
        params = {"q": f"%{q}%", "model": model, "limit": limit, "offset": offset}
        rows = conn.execute(f"""
            SELECT p.patient_id, p.sex,
                   CASE WHEN p.year_of_birth IS NULL OR c.aki_index_date IS NULL THEN NULL
                        ELSE extract(year FROM c.aki_index_date)::int - p.year_of_birth END AS age_at_index,
                   c.aki_index_date, c.y_esrd_or_dialysis_2yr AS outcome,
                   pr.probability, pr.above_threshold, count(*) OVER () AS total
            FROM ehr.patient p LEFT JOIN cohort.aki_cohort c USING (patient_id)
            LEFT JOIN app.prediction pr ON pr.patient_id = p.patient_id AND pr.model_key = %(model)s
            WHERE {' AND '.join(where)} ORDER BY {order} LIMIT %(limit)s OFFSET %(offset)s""", params).fetchall()
        total = rows[0]["total"] if rows else 0
        for r in rows:
            r.pop("total")
        return {"dataset_kind": state["dataset"]["kind"], "total": total, "items": rows}

    @app.get("/api/patients/{pid}")
    def patient(pid: str, conn=Depends(db)):
        p = _patient_or_404(conn, pid)
        p["event_counts"] = conn.execute("""
            SELECT (SELECT count(*) FROM ehr.diagnosis WHERE patient_id = %(p)s) AS diagnosis,
                   (SELECT count(*) FROM ehr.procedure WHERE patient_id = %(p)s) AS procedure,
                   (SELECT count(*) FROM ehr.medication_drug WHERE patient_id = %(p)s) AS medication,
                   (SELECT count(*) FROM ehr.lab_result WHERE patient_id = %(p)s) AS lab""", {"p": pid}).fetchone()
        p["predictions"] = conn.execute("""
            SELECT model_key, probability, threshold, above_threshold, n_events, n_used, n_oov, scored_at
            FROM app.prediction WHERE patient_id = %s ORDER BY model_key""", (pid,)).fetchall()
        return {"dataset_kind": state["dataset"]["kind"], "disclaimer": DISCLAIMER[state["dataset"]["kind"]], **p}

    @app.get("/api/patients/{pid}/timeline")
    def timeline(pid: str, types: str = ",".join(EVENT_TYPES), observation_only: bool = False,
                 limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0), conn=Depends(db)):
        p = _patient_or_404(conn, pid)
        wanted = [t for t in types.split(",") if t in EVENT_TYPES]
        parts = {
            "diagnosis": "SELECT date, 'diagnosis' AS type, code_system, code, NULL::float8 AS value, NULL AS units, encounter_id FROM ehr.diagnosis WHERE patient_id = %(p)s",
            "procedure": "SELECT date, 'procedure', code_system, code, NULL, NULL, encounter_id FROM ehr.procedure WHERE patient_id = %(p)s",
            "medication": "SELECT start_date, 'medication', code_system, code, NULL, brand, encounter_id FROM ehr.medication_drug WHERE patient_id = %(p)s",
            "lab": "SELECT date, 'lab', code_system, code, lab_result_num_val, units_of_measure, encounter_id FROM ehr.lab_result WHERE patient_id = %(p)s",
        }
        if not wanted:
            raise HTTPException(422, f"types must be a subset of {EVENT_TYPES}")
        union = " UNION ALL ".join(parts[t] for t in wanted)
        cond = "WHERE e.date < %(blank)s" if observation_only and p["blank_start_date"] else ""
        rows = conn.execute(f"""
            SELECT e.date, e.type, e.code_system, e.code, e.value, e.units, count(*) OVER () AS total,
                   (%(blank)s::date IS NOT NULL AND e.date < %(blank)s::date) AS in_observation_window
            FROM ({union}) AS e(date, type, code_system, code, value, units, encounter_id) {cond}
            ORDER BY e.date DESC NULLS LAST, e.type, e.code LIMIT %(limit)s OFFSET %(offset)s""",
            {"p": pid, "blank": p["blank_start_date"], "limit": limit, "offset": offset}).fetchall()
        total = rows[0]["total"] if rows else 0
        for r in rows:
            r.pop("total")
        return {"dataset_kind": state["dataset"]["kind"], "patient_id": pid, "aki_index_date": p["aki_index_date"],
                "blank_start_date": p["blank_start_date"], "total": total, "items": rows}

    @app.get("/api/patients/{pid}/risk")
    def risk(pid: str, model: str, top_k: int = Query(10, ge=1, le=50), conn=Depends(db)):
        _patient_or_404(conn, pid)
        m = state["models"].get(model)
        if m is None:
            raise HTTPException(404, f"model {model!r} not loaded; available: {sorted(state['models'])}")
        variant = "withlabs" if m.meta["labs"] else "nolabs"
        seq = conn.execute("SELECT tokens, dates FROM app.sequence WHERE patient_id = %s AND variant = %s",
                           (pid, variant)).fetchone()
        if seq is None:
            raise HTTPException(404, "no model-input sequence for this patient (not in cohort, or < 3 events)")
        s = m.score(seq["tokens"], top_k=top_k)
        if s["attention"]:
            for a in s["attention"]:
                a["date"] = seq["dates"][a["position"]]
        stored = conn.execute("SELECT probability FROM app.prediction WHERE patient_id = %s AND model_key = %s",
                              (pid, model)).fetchone()
        return {"dataset_kind": state["dataset"]["kind"], "disclaimer": DISCLAIMER[state["dataset"]["kind"]],
                "patient_id": pid, "model_key": model, "model": m.meta["model"], "variant": variant,
                "model_source": m.meta["source"], **s,
                "stored_probability": stored["probability"] if stored else None,
                "note": ("RETAIN attention (alpha) over the events the model saw; with keep_first truncation only "
                         f"the first {m.meta['maxlen']} events are used.") if s["attention"] else
                        "The Transformer has no per-event attribution in this version."}

    @app.get("/api/fairness")
    def fairness():
        path = state["fairness_path"]
        if not path.exists():
            raise HTTPException(404, "no fairness report found; run `python -m aki.eval.fairness`")
        return json.loads(path.read_text())

    # ------------------------------------------------------------------ static frontend
    dist = web_dist or ROOT / "web" / "dist"
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
