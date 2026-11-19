"""Hyperparameter search + final evaluation with MLflow tracking.

Protocol (differs from the notebooks on purpose, see docs/reconstruction/pytorch_vs_keras.md):
  1. Stratified patient-level split: train 64% / val 16% / test 20%.
  2. Vocabulary from TRAIN sequences only.
  3. For each grid config (focal alpha x dropout, the notebooks' final grid):
     train with early stopping on val ROC-AUC, choose the F1-optimal threshold on VAL.
  4. Select the config with the highest val F1 (the notebooks' criterion), ties -> val ROC-AUC.
  5. Only then score the TEST set, once, with the selected model and val-chosen threshold;
     also report test TPR at thresholds chosen on val for FPR 5% and 1%, and bootstrap 95% CIs.

MLflow: one parent run per experiment, one nested run per grid config. Only
aggregate values are logged (no patient IDs). The tracking store and artifacts
(including model weights) stay local and gitignored (mlflow.db, mlartifacts/).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from aki.model.data import Vocab, fingerprint, pad_truncate, split_indices, synthetic_sequences
from aki.model.metrics import bootstrap_ci, compute_metrics, select_threshold, threshold_at_fpr
from aki.model.train import TrainConfig, fit, predict_proba, resolve_device

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING = f"sqlite:///{ROOT / 'mlflow.db'}"
DEFAULT_ARTIFACTS = (ROOT / "mlartifacts").as_uri()


def _git_version() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def _loggable(m: dict) -> dict:
    """Scalar metrics for MLflow: rates, confusion counts (tn/fp/fn/tp), n, prevalence, recall alias."""
    out = {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out["recall"] = m["sensitivity"]
    out["prevalence"] = m["positives"] / m["n"] if m["n"] else float("nan")
    return out


def _ensure_experiment(name: str, artifact_location: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name, artifact_location=artifact_location)
    return exp.experiment_id


def run_experiment(patient_ids, labels, sequences, *, base: TrainConfig, alphas=(0.25, 0.5), dropouts=(0.1, 0.2),
                   truncate: str = "keep_first", dataset_name: str = "unnamed", tracking_uri: str = DEFAULT_TRACKING,
                   experiment_name: str = "aki-reconstruction", artifact_location: str = DEFAULT_ARTIFACTS,
                   n_boot: int = 1000, log_model: bool = True) -> dict:
    t_start = time.time()
    labels = np.asarray(labels).astype(int)
    tr, va, te = split_indices(labels, seed=base.seed)
    vocab = Vocab.build([sequences[i] for i in tr])
    enc = lambda idx: pad_truncate([vocab.encode(sequences[i]) for i in idx], base.maxlen, truncate)  # noqa: E731
    X_tr, X_va, X_te = enc(tr), enc(va), enc(te)
    y_tr, y_va, y_te = labels[tr], labels[va], labels[te]

    mlflow.set_tracking_uri(tracking_uri)
    exp_id = _ensure_experiment(experiment_name, artifact_location)
    summary = {
        "dataset": dataset_name, "data_fingerprint": fingerprint(patient_ids, labels), "model": base.model,
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "split_positives": {"train": int(y_tr.sum()), "val": int(y_va.sum()), "test": int(y_te.sum())},
        "vocab_size_train": len(vocab) - 2, "oov_rate_test": float(np.mean(X_te[X_te > 0] == 1)) if (X_te > 0).any() else 0.0,
        "truncate": truncate, "grid": {"focal_alpha": list(alphas), "dropout": list(dropouts)},
        "base_config": base.to_dict(), "code_version": _git_version(), "configs": [],
    }
    with mlflow.start_run(experiment_id=exp_id, run_name=f"{base.model}-{dataset_name}") as parent:
        mlflow.set_tags({"reconstruction": "true", "framework": "pytorch",
                         "provenance": "new PyTorch implementation of the original Keras model",
                         "selection": "val only; test scored once after selection"})
        mlflow.log_params({"model": base.model, "dataset": dataset_name, "data_fingerprint": summary["data_fingerprint"],
                           "truncate": truncate, "maxlen": base.maxlen, "epochs": base.epochs,
                           "patience": base.patience, "batch_size": base.batch_size, "lr": base.lr,
                           "mask_padding": base.mask_padding, "seed": base.seed,
                           "grid_alpha": str(list(alphas)), "grid_dropout": str(list(dropouts)),
                           "vocab_size_train": summary["vocab_size_train"], **{f"n_{k}": v for k, v in summary["split_sizes"].items()},
                           **{f"pos_{k}": v for k, v in summary["split_positives"].items()},
                           **{f"prevalence_{k}": round(summary["split_positives"][k] / summary["split_sizes"][k], 5)
                              for k in summary["split_sizes"]}})
        best = None
        for alpha, dropout in product(alphas, dropouts):
            cfg = replace(base, focal_alpha=alpha, dropout=dropout)
            with mlflow.start_run(experiment_id=exp_id, run_name=f"a{alpha}-d{dropout}", nested=True):
                mlflow.log_params({"focal_alpha": alpha, "dropout": dropout})
                t_cfg = time.time()
                model, info = fit(cfg, len(vocab), X_tr, y_tr, X_va, y_va,
                                  on_epoch=lambda e, r: mlflow.log_metrics(
                                      {"train_loss": r["train_loss"], "val_roc_auc": r["val_roc_auc"]}, step=e))
                device = resolve_device(cfg.device)
                p_va = predict_proba(model, X_va, device)
                thr, _ = select_threshold(y_va, p_va)
                val = compute_metrics(y_va, p_va, thr)
                runtime = round(time.time() - t_cfg, 1)
                mlflow.log_metrics({f"val_{k}": float(v) for k, v in _loggable(val).items()})
                mlflow.log_metrics({"best_epoch": info["best_epoch"], "epochs_run": len(info["history"]),
                                    "runtime_seconds": runtime})
                rec = {"focal_alpha": alpha, "dropout": dropout, "best_epoch": info["best_epoch"],
                       "epochs_run": len(info["history"]), "runtime_seconds": runtime, "val": val}
                summary["configs"].append(rec)
                key = (val["f1"], val["roc_auc"])
                if best is None or key > best[0]:
                    best = (key, cfg, model, p_va, thr)

        _, cfg, model, p_va, thr = best
        summary["selected"] = {"focal_alpha": cfg.focal_alpha, "dropout": cfg.dropout, "threshold_val_f1": thr}
        # ---- test set: touched exactly once, after selection ----
        device = resolve_device(cfg.device)
        summary["device"] = str(device)  # MPS kernels are not bitwise deterministic; CPU is
        mlflow.set_tag("device", str(device))
        p_te = predict_proba(model, X_te, device)
        test = compute_metrics(y_te, p_te, thr)
        test["ci95"] = bootstrap_ci(y_te, p_te, thr, n_boot=n_boot, seed=base.seed)
        at_fpr = {}
        for fpr in (0.05, 0.01):
            t = threshold_at_fpr(y_va, p_va, fpr)
            m = compute_metrics(y_te, p_te, t)
            at_fpr[f"fpr{int(fpr * 100)}"] = {"threshold_from_val": t, "test_tpr": m["sensitivity"],
                                              "test_fpr": 1 - m["specificity"], "test_precision": m["precision"]}
        test["at_val_fpr"] = at_fpr
        summary["test"] = test
        mlflow.log_params({"selected_focal_alpha": cfg.focal_alpha, "selected_dropout": cfg.dropout,
                           "selected_threshold": thr})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in _loggable(test).items()})
        for k, (lo, hi) in test["ci95"].items():
            mlflow.log_metrics({f"test_{k}_ci95_low": lo, f"test_{k}_ci95_high": hi})
        for name, m in at_fpr.items():
            mlflow.log_metric(f"test_tpr_at_val_{name}", m["test_tpr"])
        summary["selected"]["val"] = next(c["val"] for c in summary["configs"]
                                          if (c["focal_alpha"], c["dropout"]) == (cfg.focal_alpha, cfg.dropout))
        summary["runtime_seconds_total"] = round(time.time() - t_start, 1)
        mlflow.log_metric("runtime_seconds_total", summary["runtime_seconds_total"])
        mlflow.log_metrics({"selected_val_roc_auc": summary["selected"]["val"]["roc_auc"],
                            "selected_val_pr_auc": summary["selected"]["val"]["pr_auc"]})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "summary.json"
            p.write_text(json.dumps(summary, indent=2))
            mlflow.log_artifact(str(p))
            if log_model:
                w = Path(d) / "model_state_dict.pt"
                torch.save(model.state_dict(), w)
                mlflow.log_artifact(str(w))
        summary["mlflow_run_id"] = parent.info.run_id
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="parquet from aki.model.sequences (patient-level, in outputs/)")
    src.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic patients instead")
    ap.add_argument("--model", choices=["retain", "transformer"], required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--dropouts", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--maxlen", type=int, default=400)
    ap.add_argument("--truncate", choices=["keep_first", "keep_last"], default="keep_first")
    ap.add_argument("--mask-padding", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tracking-uri", default=DEFAULT_TRACKING)
    ap.add_argument("--experiment", default="aki-reconstruction")
    ap.add_argument("--results-out", type=Path, help="aggregate JSON (default reports/experiments/<run>.json)")
    a = ap.parse_args()

    if a.synthetic:
        pids, labels, seqs = synthetic_sequences(a.synthetic, seed=a.seed)
        name = f"synthetic{a.synthetic}"
    else:
        df = pd.read_parquet(a.data)
        pids, labels, seqs = df["patient_id"].tolist(), df["label"].to_numpy(), [list(t) for t in df["tokens"]]
        name = a.data.stem
    base = TrainConfig(model=a.model, epochs=a.epochs, patience=a.patience, maxlen=a.maxlen,
                       mask_padding=a.mask_padding, device=a.device, seed=a.seed)
    s = run_experiment(pids, labels, seqs, base=base, alphas=a.alphas, dropouts=a.dropouts, truncate=a.truncate,
                       dataset_name=name, tracking_uri=a.tracking_uri, experiment_name=a.experiment, n_boot=a.n_boot)
    out = a.results_out or ROOT / "reports" / "experiments" / f"{a.model}_{name}_{a.truncate}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2))
    t = s["test"]
    print(json.dumps({"selected": s["selected"], "test_roc_auc": t["roc_auc"], "test_pr_auc": t["pr_auc"],
                      "test_f1": t["f1"], "ci95": t["ci95"], "results": str(out), "mlflow_run_id": s["mlflow_run_id"]}, indent=2))


if __name__ == "__main__":
    main()
