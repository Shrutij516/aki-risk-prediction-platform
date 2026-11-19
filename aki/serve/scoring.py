"""Load sequences into app.sequence, register model bundles in app.model, batch-score into app.prediction."""
from __future__ import annotations

import json

import pandas as pd
import psycopg

from aki.serve.registry import ServingModel


def dataset_kind(conn: psycopg.Connection) -> str | None:
    row = conn.execute("SELECT kind FROM app.dataset_info").fetchone()
    return row[0] if row else None


def set_dataset_info(conn: psycopg.Connection, kind: str, description: str) -> None:
    conn.execute("""INSERT INTO app.dataset_info (kind, description) VALUES (%s, %s)
                    ON CONFLICT (singleton) DO UPDATE SET kind = EXCLUDED.kind, description = EXCLUDED.description,
                                                          created_at = now()""", (kind, description))


def load_sequences(conn: psycopg.Connection, seqs: pd.DataFrame, variant: str) -> int:
    conn.execute("DELETE FROM app.sequence WHERE variant = %s", (variant,))
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO app.sequence (patient_id, variant, tokens, dates) VALUES (%s, %s, %s, %s)",
                        [(r.patient_id, variant, [str(t) for t in r.tokens],
                          [pd.Timestamp(d).date() for d in r.dates]) for r in seqs.itertuples()])
    return len(seqs)


def register_models(conn: psycopg.Connection, models: dict[str, ServingModel]) -> None:
    for key, m in models.items():
        meta = m.meta
        conn.execute("""
            INSERT INTO app.model (model_key, model, variant, source, mlflow_run_id, threshold, maxlen, truncate,
                                   focal_alpha, dropout, val_metrics, test_metrics, split)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_key) DO UPDATE SET model = EXCLUDED.model, variant = EXCLUDED.variant,
                source = EXCLUDED.source, mlflow_run_id = EXCLUDED.mlflow_run_id, threshold = EXCLUDED.threshold,
                maxlen = EXCLUDED.maxlen, truncate = EXCLUDED.truncate, focal_alpha = EXCLUDED.focal_alpha,
                dropout = EXCLUDED.dropout, val_metrics = EXCLUDED.val_metrics, test_metrics = EXCLUDED.test_metrics,
                split = EXCLUDED.split, registered_at = now()""",
            (key, meta["model"], "withlabs" if meta["labs"] else "nolabs", meta["source"], meta.get("mlflow_run_id"),
             meta["threshold"], meta["maxlen"], meta["truncate"], meta.get("focal_alpha"), meta.get("dropout"),
             json.dumps(meta.get("val_metrics")), json.dumps(meta.get("test_metrics")),
             json.dumps({"sizes": meta.get("split_sizes"), "positives": meta.get("split_positives")})))


def score_all(conn: psycopg.Connection, models: dict[str, ServingModel]) -> dict[str, int]:
    """Score every patient that has a sequence of the model's variant. Returns rows written per model."""
    written = {}
    for key, m in models.items():
        variant = "withlabs" if m.meta["labs"] else "nolabs"
        rows = conn.execute("SELECT patient_id, tokens FROM app.sequence WHERE variant = %s", (variant,)).fetchall()
        conn.execute("DELETE FROM app.prediction WHERE model_key = %s", (key,))
        out = []
        for pid, tokens in rows:
            s = m.score(tokens)
            out.append((pid, key, s["probability"], s["threshold"], s["above_threshold"], s["n_events"], s["n_used"],
                        s["n_oov"], json.dumps(s["attention"])))
        with conn.cursor() as cur:
            cur.executemany("""INSERT INTO app.prediction (patient_id, model_key, probability, threshold, above_threshold,
                                   n_events, n_used, n_oov, attention) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""", out)
        written[key] = len(out)
    return written
