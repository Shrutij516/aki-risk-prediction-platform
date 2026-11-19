"""Export trained models from MLflow into self-contained local serving bundles, and verify them.

For each selected run, this:
  1. reads the run's summary.json (selected config, threshold, maxlen, truncate) from MLflow;
  2. rebuilds the training-only vocabulary exactly as aki.model.experiment did
     (same parquet row order, split_indices(labels, seed), Vocab.build(train sequences));
  3. loads model_state_dict.pt and runs CPU inference on the held-out TEST split;
  4. verifies the CPU predictions against the test metrics logged in MLflow
     (training ran on MPS, so tiny float differences are expected; tolerances below);
  5. writes outputs/models/<key>/{weights.pt, vocab.json, meta.json} and, for the
     fairness step, outputs/predictions/<key>_test.parquet (patient-level: local only).

No training happens here. Nothing in outputs/ is committed.

    python -m aki.serve.export --experiment aki-real-baseline
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from aki.model.data import Vocab, pad_truncate, split_indices
from aki.model.experiment import DEFAULT_TRACKING
from aki.model.metrics import compute_metrics
from aki.model.models import build_model
from aki.model.train import predict_proba

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "outputs" / "models"
PRED_DIR = ROOT / "outputs" / "predictions"
AUC_TOL = 0.002          # |ROC-AUC, PR-AUC difference| allowed between MPS-trained/logged and CPU re-scored
COUNT_TOL_FRAC = 0.002   # confusion-matrix cells may move by <= 0.2% of the test set (threshold-edge cases)


def _load_sequences(path: Path):
    df = pd.read_parquet(path)
    seqs = [list(t) for t in df["tokens"]]
    return df["patient_id"].tolist(), df["label"].to_numpy().astype(int), seqs


def export_run(run_id: str, key: str, sequences_path: Path, source: str, tracking_uri: str = DEFAULT_TRACKING,
               models_dir: Path = MODELS_DIR, pred_dir: Path | None = PRED_DIR) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    tmp = Path(mlflow.artifacts.download_artifacts(run_id=run_id))
    summary = json.loads((tmp / "summary.json").read_text())
    base = summary["base_config"]
    sel = summary["selected"]

    pids, labels, seqs = _load_sequences(sequences_path)
    from aki.model.data import fingerprint
    if fingerprint(pids, labels) != summary["data_fingerprint"]:
        raise RuntimeError(f"{key}: sequences file does not match the run's data fingerprint")
    tr, va, te = split_indices(labels, seed=base["seed"])
    vocab = Vocab.build([seqs[i] for i in tr])
    if len(vocab) - 2 != summary["vocab_size_train"]:
        raise RuntimeError(f"{key}: rebuilt vocab size {len(vocab) - 2} != logged {summary['vocab_size_train']}")

    model = build_model(base["model"], len(vocab), base["maxlen"], sel["dropout"], base["mask_padding"])
    model.load_state_dict(torch.load(tmp / "model_state_dict.pt", map_location="cpu"))
    device = torch.device("cpu")
    X_te = pad_truncate([vocab.encode(seqs[i]) for i in te], base["maxlen"], summary["truncate"])
    p_te = predict_proba(model.to(device), X_te, device)
    thr = sel["threshold_val_f1"]
    m = compute_metrics(labels[te], p_te, thr)
    logged = summary["test"]
    tol_n = max(1, int(COUNT_TOL_FRAC * len(te)))
    checks = {
        "roc_auc": (m["roc_auc"], logged["roc_auc"], abs(m["roc_auc"] - logged["roc_auc"]) <= AUC_TOL),
        "pr_auc": (m["pr_auc"], logged["pr_auc"], abs(m["pr_auc"] - logged["pr_auc"]) <= AUC_TOL),
        **{k: (m[k], logged[k], abs(m[k] - logged[k]) <= tol_n) for k in ("tn", "fp", "fn", "tp")},
    }
    ok = all(c[2] for c in checks.values())
    verification = {"passed": ok, "device": "cpu", "n_test": len(te),
                    "auc_tolerance": AUC_TOL, "count_tolerance": tol_n,
                    "checks": {k: {"cpu": c[0], "logged": c[1], "ok": c[2]} for k, c in checks.items()}}
    if not ok:
        raise RuntimeError(f"{key}: CPU re-scoring does not match MLflow: {json.dumps(verification['checks'])}")

    out = models_dir / key
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(tmp / "model_state_dict.pt", out / "weights.pt")
    (out / "vocab.json").write_text(json.dumps(vocab.token2id))
    meta = {"key": key, "model": base["model"], "labs": key.endswith("withlabs"),
            "source": source, "mlflow_run_id": run_id, "maxlen": base["maxlen"], "truncate": summary["truncate"],
            "mask_padding": base["mask_padding"], "dropout": sel["dropout"], "focal_alpha": sel["focal_alpha"],
            "threshold": thr, "vocab_size": len(vocab), "val_metrics": sel.get("val"),
            "test_metrics": {k: v for k, v in logged.items()}, "split_sizes": summary["split_sizes"],
            "split_positives": summary["split_positives"], "verification": verification}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"patient_id": [pids[i] for i in te], "label": labels[te], "prob": p_te}).to_parquet(
            pred_dir / f"{key}_test.parquet", index=False)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="aki-real-baseline")
    ap.add_argument("--sequences-dir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--source", default="real-trained", choices=["real-trained", "synthetic-trained"])
    ap.add_argument("--tracking-uri", default=DEFAULT_TRACKING)
    ap.add_argument("--no-predictions", action="store_true")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    a = ap.parse_args()
    mlflow.set_tracking_uri(a.tracking_uri)
    exp = mlflow.get_experiment_by_name(a.experiment)
    if exp is None:
        raise SystemExit(f"MLflow experiment {a.experiment!r} not found")
    parents = [r for r in mlflow.tracking.MlflowClient().search_runs([exp.experiment_id])
               if "mlflow.parentRunId" not in r.data.tags and r.info.status == "FINISHED"]
    results = {}
    for r in sorted(parents, key=lambda r: r.info.start_time):
        dataset = r.data.params["dataset"]                 # e.g. sequences_withlabs
        key = f"{r.data.params['model']}_{dataset.replace('sequences_', '')}"
        if key in results:
            raise SystemExit(f"duplicate finished runs for {key}; refusing to guess which one to serve")
        meta = export_run(r.info.run_id, key, a.sequences_dir / f"{dataset}.parquet", a.source,
                          a.tracking_uri, models_dir=a.models_dir,
                          pred_dir=None if a.no_predictions else PRED_DIR)
        results[key] = meta["verification"]
        c = meta["verification"]["checks"]
        print(f"{key:24s} OK  ROC-AUC cpu={c['roc_auc']['cpu']:.4f} logged={c['roc_auc']['logged']:.4f}  "
              f"PR-AUC cpu={c['pr_auc']['cpu']:.4f} logged={c['pr_auc']['logged']:.4f}  "
              f"TN/FP/FN/TP cpu={[c[k]['cpu'] for k in ('tn','fp','fn','tp')]} logged={[c[k]['logged'] for k in ('tn','fp','fn','tp')]}")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"export_verification_{a.experiment}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
