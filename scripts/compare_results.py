"""Reconstructed (PyTorch) results next to the original published (Keras) results.

Reads reports/experiments/real_<model>_<labs>.json and writes
reports/experiments/comparison.md and comparison.json (aggregate numbers only).

Original numbers are copied from AKI_Dialysis_Prediction-main/README.md (4-way table);
the source of each column is noted because the README and saved notebook outputs
do not always agree (see PROVENANCE.md).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "experiments"

import sys

sys.path.insert(0, str(ROOT))
from aki.eval.original_results import ORIGINAL  # noqa: E402

VARIANTS = [("retain", "withlabs"), ("retain", "nolabs"), ("transformer", "withlabs"), ("transformer", "nolabs")]


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> None:
    rows, missing = [], []
    for model, labs in VARIANTS:
        p = EXP / f"real_{model}_{labs}.json"
        if not p.exists():
            missing.append(f"{model}_{labs}")
            continue
        s = json.loads(p.read_text())
        t, sel = s["test"], s["selected"]
        rows.append(dict(
            model=model, labs=labs, alpha=sel["focal_alpha"], dropout=sel["dropout"], threshold=sel["threshold_val_f1"],
            val_roc_auc=sel["val"]["roc_auc"], val_pr_auc=sel["val"]["pr_auc"], val_f1=sel["val"]["f1"],
            prevalence_test=t["positives"] / t["n"], n_test=t["n"], pos_test=t["positives"],
            prevalence_train=s["split_positives"]["train"] / s["split_sizes"]["train"],
            test=t, ci=t["ci95"], runtime_min=s.get("runtime_seconds_total", float("nan")) / 60,
            device=s.get("device"), mlflow_run_id=s["mlflow_run_id"], vocab=s["vocab_size_train"],
            oov_test=s["oov_rate_test"], configs=s["configs"], original=ORIGINAL[(model, labs)]))
    best = max(rows, key=lambda r: r["val_roc_auc"]) if rows else None

    L = ["# Reconstructed results (PyTorch) vs original published results (Keras)", "",
         "**Reconstructed results, not exact reproductions.** Different framework and random streams; validation-only "
         "selection (originals tuned on test); train-only vocabulary; 64/16/20 split vs 80/20. Baseline "
         "notebook-compatible settings: oldest-400 truncation, no Transformer padding mask.", ""]
    if missing:
        L += [f"Missing (failed or not run): {', '.join(missing)}", ""]
    L += ["## Reconstructed — selected config per variant (test set scored once)", "",
          "| Variant | α | dropout | threshold (val) | val ROC-AUC | test ROC-AUC [95% CI] | test PR-AUC [95% CI] | F1 | recall | precision | specificity | TN/FP/FN/TP | test prevalence | runtime |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t, c = r["test"], r["ci"]
        L.append(f"| {r['model']} {r['labs']}{' ★' if r is best else ''} | {r['alpha']} | {r['dropout']} | {r['threshold']:.2f} | "
                 f"{fmt(r['val_roc_auc'])} | {fmt(t['roc_auc'])} [{fmt(c['roc_auc'][0])}, {fmt(c['roc_auc'][1])}] | "
                 f"{fmt(t['pr_auc'])} [{fmt(c['pr_auc'][0])}, {fmt(c['pr_auc'][1])}] | {fmt(t['f1'])} | {fmt(t['sensitivity'])} | "
                 f"{fmt(t['precision'])} | {fmt(t['specificity'])} | {t['tn']}/{t['fp']}/{t['fn']}/{t['tp']} | "
                 f"{r['prevalence_test']:.1%} ({r['pos_test']}/{r['n_test']}) | {r['runtime_min']:.0f} min |")
    if best:
        L += ["", f"★ Best by validation ROC-AUC: **{best['model']} {best['labs']}** (val ROC-AUC {best['val_roc_auc']:.3f})."]
    L += ["", "## Original published (Keras, README table; threshold and hyperparameters tuned on test)", "",
          "| Variant | ROC-AUC | PR-AUC | F1 | recall | precision | specificity | threshold | note |", "|---|---|---|---|---|---|---|---|---|"]
    for model, labs in VARIANTS:
        o = ORIGINAL[(model, labs)]
        L.append(f"| {model} {labs} | {fmt(o['roc_auc'])} | {fmt(o['pr_auc'])} | {fmt(o['f1'])} | {fmt(o['sensitivity'])} | "
                 f"{fmt(o['precision'])} | {fmt(o['specificity'])} | {fmt(o['threshold'], 2)} | {o['note']} |")
    L += ["", "## All grid configs (validation only)", "", "| Variant | α | dropout | val ROC-AUC | val PR-AUC | val F1 | threshold | best epoch | epochs run | runtime |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for c in r["configs"]:
            v = c["val"]
            L.append(f"| {r['model']} {r['labs']} | {c['focal_alpha']} | {c['dropout']} | {fmt(v['roc_auc'])} | {fmt(v['pr_auc'])} | "
                     f"{fmt(v['f1'])} | {v['threshold']:.2f} | {c['best_epoch']} | {c['epochs_run']} | {c['runtime_seconds'] / 60:.0f} min |")
    (EXP / "comparison.md").write_text("\n".join(L) + "\n")
    (EXP / "comparison.json").write_text(json.dumps({"rows": rows, "best_by_val_roc_auc": best and f"{best['model']}_{best['labs']}",
                                                     "missing": missing}, indent=2, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
