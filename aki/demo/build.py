"""Build the synthetic demo database end to end, reusing the real pipeline code.

    python -m aki.demo.build --reset

Steps: synthetic TriNetX CSVs -> separate database (default aki_demo) -> migrations ->
ingestion + SQL cohort -> sequences (same builder as training) -> app.sequence ->
register model bundles -> batch scoring -> app.prediction.

Model source (--model-source):
  real       bundles exported from the real-data MLflow runs (outputs/models/, local only).
             Their *scores on synthetic patients have no clinical meaning*; the dashboard says so.
  synthetic  small models trained here on the synthetic data only (outputs/models_synthetic/),
             so the demo runs on a machine without the real data or weights.
  auto       real if outputs/models/ has bundles, else synthetic.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psycopg

from aki.db.connection import ensure_database, get_dsn
from aki.db.migrate import migrate
from aki.demo.synthetic_ehr import generate
from aki.ingest.trinetx import IngestConfig, ingest
from aki.model.sequences import build_sequences, cohort_from_postgres
from aki.serve.registry import load_all
from aki.serve.scoring import dataset_kind, load_sequences, register_models, score_all, set_dataset_info

ROOT = Path(__file__).resolve().parents[2]
REAL_MODELS = ROOT / "outputs" / "models"
SYN_MODELS = ROOT / "outputs" / "models_synthetic"


def _train_synthetic_models(seq_dir: Path) -> None:
    """Train one small config per model/variant on the SYNTHETIC sequences and export bundles."""
    env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1")
    for model in ("retain", "transformer"):
        for variant in ("nolabs", "withlabs"):
            subprocess.run([sys.executable, "-m", "aki.model.experiment", "--data", str(seq_dir / f"sequences_{variant}.parquet"),
                            "--model", model, "--alphas", "0.5", "--dropouts", "0.1", "--epochs", "3", "--device", "cpu",
                            "--n-boot", "200", "--experiment", "aki-demo-synthetic",
                            "--results-out", str(ROOT / "reports" / "experiments" / f"demo_synthetic_{model}_{variant}.json")],
                           check=True, env=env)
    subprocess.run([sys.executable, "-m", "aki.serve.export", "--experiment", "aki-demo-synthetic", "--source",
                    "synthetic-trained", "--sequences-dir", str(seq_dir), "--no-predictions", "--models-dir", str(SYN_MODELS)],
                   check=True, env=env)


def build(db: str, n: int, seed: int, reset: bool, model_source: str) -> dict:
    t0 = time.time()
    data_dir = ROOT / "outputs" / "demo_data"
    seq_dir = ROOT / "outputs" / "demo_sequences"
    summary: dict = {"database": db, "n_patients": n, "seed": seed}
    summary["generated"] = generate(data_dir, n, seed)

    dsn = ensure_database(get_dsn(), db, reset=reset)
    migrate(dsn)
    with psycopg.connect(dsn) as conn:
        kind = dataset_kind(conn)
    if kind == "real":
        raise SystemExit(f"database {db} is marked REAL; refusing to overwrite it with demo data")
    s = ingest(dsn, IngestConfig(data_dir=data_dir, select="all", replace=True, build_cohort=True))
    summary["cohort_rows"] = s["cohort_rows"]
    summary["ingested_rows"] = {f["target_table"]: f["rows_loaded"] for f in s["files"]}

    cohort = cohort_from_postgres(dsn)
    seq_dir.mkdir(parents=True, exist_ok=True)
    seqs = {}
    for variant in ("nolabs", "withlabs"):
        df, stats = build_sequences(data_dir, cohort, include_labs=variant == "withlabs")
        df.to_parquet(seq_dir / f"sequences_{variant}.parquet", index=False)
        seqs[variant] = df
        summary[f"sequences_{variant}"] = {k: stats[k] for k in ("sequence_patients", "sequence_positives")}

    if model_source == "auto":
        model_source = "real" if load_all(REAL_MODELS) else "synthetic"
    if model_source == "synthetic" and not load_all(SYN_MODELS):
        _train_synthetic_models(seq_dir)
    models = load_all(REAL_MODELS if model_source == "real" else SYN_MODELS)
    if not models:
        raise SystemExit(f"no model bundles found for source {model_source!r}")
    summary["model_source"] = model_source

    with psycopg.connect(dsn) as conn:
        set_dataset_info(conn, "synthetic", f"Synthetic demo data from aki.demo.synthetic_ehr (n={n}, seed={seed}). "
                                            "Not real patients; predictions have no clinical meaning.")
        for variant, df in seqs.items():
            load_sequences(conn, df, variant)
        conn.execute("DELETE FROM app.prediction")
        conn.execute("DELETE FROM app.model")
        register_models(conn, models)
        summary["predictions"] = score_all(conn, models)
    summary["seconds"] = round(time.time() - t0, 1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="aki_demo")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reset", action="store_true", help="drop and recreate the demo database")
    ap.add_argument("--model-source", choices=["auto", "real", "synthetic"], default="auto")
    a = ap.parse_args()
    if a.db in ("postgres", "aki_full_cohort"):
        raise SystemExit("refusing to build demo data into a database that holds real data")
    print(json.dumps(build(a.db, a.n, a.seed, a.reset, a.model_source), indent=2))


if __name__ == "__main__":
    main()
