"""Subgroup (fairness) evaluation of the real-data held-out test predictions.

Inputs (local only):
  outputs/predictions/<model_key>_test.parquet   test-split predictions written by aki.serve.export
                                                 (identical to the MLflow-logged evaluation; verified there)
  database with ehr.patient + cohort.aki_cohort  demographics as recorded in the EHR extract

Attributes: sex, race, ethnicity (as recorded), age band at AKI index (from year of birth).
Per group: n, prevalence, ROC-AUC, PR-AUC, TPR and FPR at the model's validation-selected
threshold, precision, mean predicted risk vs observed rate; 95% percentile-bootstrap CIs
(1,000 resamples within the group), and the difference vs the largest group with a 95% CI
from independent bootstrap resamples.

Suppression: a group is not reported (and its size is not shown) if n < 50 or it has fewer
than 11 positives or 11 negatives.

Output is aggregate only: reports/fairness/fairness_report.{json,md}, plus an MLflow run per
model in experiment 'aki-fairness'. Existing training runs are not modified.

    python -m aki.eval.fairness --dsn <database with the full cohort>
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "outputs" / "predictions"
OUT_DIR = ROOT / "reports" / "fairness"
MIN_N, MIN_CELL = 50, 11
ATTRIBUTES = ("sex", "race", "ethnicity", "age_band")

LIMITATIONS = [
    "Race, ethnicity and sex are as recorded in the EHR extract; 'Unknown' categories are large and recording quality is not known.",
    "The lab tokens include eGFR codes from race-based equations (MDRD 48642-3 'non-Black' / 48643-1 'Black'), so race can enter the model through its inputs; race-free CKD-EPI 2021 (88293-6) is a small share of the data.",
    "Positive counts per group are small, so intervals are wide; overlapping intervals mean no difference was detected, not that groups are treated equally.",
    "One held-out split (seed 42); no cross-validation or external validation.",
    "A single threshold per model (chosen on the overall validation set) is used for all groups; TPR/FPR gaps depend on that choice.",
    "No intersectional groups (e.g. race x sex) are reported: most would be suppressed at these sample sizes.",
    "Age is approximated from year of birth; patients with a protected birth year fall into 'Unknown'.",
    "Many comparisons are made (up to 15 groups x 3 metrics x 4 models), with no multiple-comparison correction; some intervals that exclude zero are expected by chance.",
    "Focal-loss training is not calibrated: mean predicted risk is compared with the observed rate per group, but absolute probabilities are not reliable.",
    "This is a descriptive audit of a research model. It does not establish that the model is fair or suitable for clinical use.",
]


def age_band(age) -> str:
    if age is None or pd.isna(age):
        return "Unknown"
    age = int(age)
    return "<45" if age < 45 else "45-64" if age < 65 else "65-74" if age < 75 else "75+"


def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    pos, neg = y == 1, y == 0
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    return {
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "tpr": tp / pos.sum(), "fpr": fp / neg.sum(), "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "mean_predicted": float(p.mean()), "prevalence": float(y.mean()),
    }


def _boot(y, p, thr, n_boot, rng) -> dict[str, np.ndarray]:
    keys = ("roc_auc", "pr_auc", "tpr", "fpr", "precision", "mean_predicted")
    out = {k: np.full(n_boot, np.nan) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        m = _metrics(yb, p[idx], thr)
        for k in keys:
            out[k][b] = m[k]
    return out


def _ci(a: np.ndarray) -> list[float]:
    a = a[~np.isnan(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [float("nan")] * 2


def evaluate_model(df: pd.DataFrame, threshold: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """df: label, prob, sex, race, ethnicity, age_band. Returns aggregate-only results."""
    rng = np.random.default_rng(seed)
    y_all, p_all = df["label"].to_numpy().astype(int), df["prob"].to_numpy()
    overall = _metrics(y_all, p_all, threshold)
    overall["n"] = len(df)
    overall["ci95"] = {k: _ci(v) for k, v in _boot(y_all, p_all, threshold, n_boot, rng).items()}
    result = {"threshold": threshold, "overall": overall, "attributes": {}}
    for attr in ATTRIBUTES:
        groups, boots = [], {}
        col = df[attr].fillna("Unknown").replace("", "Unknown")
        for g in sorted(col.unique(), key=lambda g: -(col == g).sum()):
            sub = df[col == g]
            y, p = sub["label"].to_numpy().astype(int), sub["prob"].to_numpy()
            npos, nneg = int(y.sum()), int(len(y) - y.sum())
            if len(y) < MIN_N or npos < MIN_CELL or nneg < MIN_CELL:
                reason = f"n < {MIN_N}" if len(y) < MIN_N else f"fewer than {MIN_CELL} positives or negatives"
                groups.append({"group": str(g), "suppressed": True, "reason": reason})
                continue
            m = _metrics(y, p, threshold)
            b = _boot(y, p, threshold, n_boot, rng)
            boots[g] = (m, b)
            groups.append({"group": str(g), "suppressed": False, "n": int(len(y)), "positives": npos, **m,
                           "ci95": {k: _ci(v) for k, v in b.items()}})
        reported = [g for g in groups if not g["suppressed"]]
        if reported:
            ref = reported[0]["group"]   # largest reported group
            for g in reported:
                g["reference_group"] = ref
                if g["group"] == ref:
                    continue
                mg, bg = boots[g["group"]]
                mr, br = boots[ref]
                g["diff_vs_reference"] = {k: {"diff": mg[k] - mr[k], "ci95": _ci(bg[k] - br[k])}
                                          for k in ("roc_auc", "tpr", "fpr")}
        result["attributes"][attr] = groups
    return result


def load_frame(pred_path: Path, dsn: str) -> pd.DataFrame:
    import psycopg

    preds = pd.read_parquet(pred_path)
    with psycopg.connect(dsn) as c:
        demo = pd.DataFrame(c.execute("""
            SELECT p.patient_id, p.sex, p.race, p.ethnicity,
                   extract(year FROM a.aki_index_date)::int - p.year_of_birth AS age
            FROM ehr.patient p JOIN cohort.aki_cohort a USING (patient_id)""").fetchall(),
            columns=["patient_id", "sex", "race", "ethnicity", "age"])
    df = preds.merge(demo, on="patient_id", how="left", validate="one_to_one")
    if df["race"].isna().any():
        raise RuntimeError("some test patients have no demographics in this database (wrong --dsn?)")
    df["age_band"] = df["age"].map(age_band)
    return df


def to_markdown(report: dict) -> str:
    f = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"  # noqa: E731
    L = ["# Subgroup evaluation — real-data held-out test set (aggregate only)", "",
         report["data_label"], "",
         f"Suppression: groups with n < {report['suppression']['min_n']} or fewer than "
         f"{report['suppression']['min_cell']} positives/negatives are not shown.", ""]
    for key, m in report["models"].items():
        o = m["overall"]
        L += [f"## {key} (threshold {m['threshold']:.2f})", "",
              f"Overall: n={o['n']}, prevalence {o['prevalence']:.1%}, ROC-AUC {f(o['roc_auc'])} "
              f"[{f(o['ci95']['roc_auc'][0])}, {f(o['ci95']['roc_auc'][1])}], TPR {f(o['tpr'])}, FPR {f(o['fpr'])}", "",
              "| Attribute | Group | n | Prevalence | ROC-AUC [95% CI] | TPR [95% CI] | FPR [95% CI] | Mean predicted | ΔROC-AUC vs ref [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|"]
        for attr, groups in m["attributes"].items():
            for g in groups:
                if g["suppressed"]:
                    L.append(f"| {attr} | {g['group']} | suppressed ({g['reason']}) | | | | | | |")
                    continue
                c = g["ci95"]
                d = g.get("diff_vs_reference", {}).get("roc_auc")
                L.append(f"| {attr} | {g['group']} | {g['n']} | {g['prevalence']:.1%} | {f(g['roc_auc'])} [{f(c['roc_auc'][0])}, {f(c['roc_auc'][1])}] | "
                         f"{f(g['tpr'])} [{f(c['tpr'][0])}, {f(c['tpr'][1])}] | {f(g['fpr'])} [{f(c['fpr'][0])}, {f(c['fpr'][1])}] | "
                         f"{f(g['mean_predicted'])} | " + (f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d else "reference") + " |")
        L.append("")
    L += ["## Limitations", ""] + [f"- {x}" for x in report["limitations"]]
    return "\n".join(L) + "\n"


def main() -> None:
    from aki.db.connection import dsn_for_database, get_dsn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", help="database with ehr.patient + cohort.aki_cohort for the full cohort "
                                  "(default: local server, database aki_full_cohort)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    dsn = a.dsn or dsn_for_database(get_dsn(), "aki_full_cohort")
    report = {"data_label": "Real-data held-out test set (reconstructed PyTorch models; validation-selected thresholds). "
                            "Aggregate metrics only.",
              "suppression": {"min_n": MIN_N, "min_cell": MIN_CELL}, "n_boot": a.n_boot,
              "attributes": list(ATTRIBUTES), "limitations": LIMITATIONS, "models": {}}
    for path in sorted(PRED_DIR.glob("*_test.parquet")):
        key = path.stem.replace("_test", "")
        meta = json.loads((ROOT / "outputs" / "models" / key / "meta.json").read_text())
        df = load_frame(path, dsn)
        res = evaluate_model(df, meta["threshold"], n_boot=a.n_boot)
        # sanity: overall must equal the logged held-out evaluation
        logged = meta["test_metrics"]
        if abs(res["overall"]["roc_auc"] - logged["roc_auc"]) > 1e-6:
            raise RuntimeError(f"{key}: overall ROC-AUC {res['overall']['roc_auc']} != logged {logged['roc_auc']}")
        res["mlflow_run_id"] = meta["mlflow_run_id"]
        report["models"][key] = res
        o = res["overall"]
        print(f"{key:22s} n={o['n']} overall ROC-AUC {o['roc_auc']:.4f} (matches logged)  "
              f"reported groups: {sum(not g['suppressed'] for gs in res['attributes'].values() for g in gs)}, "
              f"suppressed: {sum(g['suppressed'] for gs in res['attributes'].values() for g in gs)}")
    if not report["models"]:
        raise SystemExit("no test predictions found; run `python -m aki.serve.export` first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fairness_report.json").write_text(json.dumps(report, indent=2))
    (OUT_DIR / "fairness_report.md").write_text(to_markdown(report))
    if not a.no_mlflow:
        import mlflow

        from aki.model.experiment import DEFAULT_ARTIFACTS, DEFAULT_TRACKING, _ensure_experiment
        mlflow.set_tracking_uri(DEFAULT_TRACKING)
        exp = _ensure_experiment("aki-fairness", DEFAULT_ARTIFACTS)
        for key, res in report["models"].items():
            with mlflow.start_run(experiment_id=exp, run_name=f"fairness-{key}"):
                mlflow.set_tags({"source_run_id": res["mlflow_run_id"], "data": "real held-out test set",
                                 "aggregate_only": "true"})
                mlflow.log_params({"threshold": res["threshold"], "min_n": MIN_N, "min_cell": MIN_CELL,
                                   "n_boot": a.n_boot})
                for attr, groups in res["attributes"].items():
                    for g in groups:
                        if not g["suppressed"]:
                            name = g["group"].replace("+", "plus").replace("<", "lt").replace(">", "gt")
                            tag = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{attr}.{name}")[:200]
                            mlflow.log_metrics({f"{tag}.roc_auc": g["roc_auc"], f"{tag}.tpr": g["tpr"],
                                                f"{tag}.fpr": g["fpr"], f"{tag}.n": g["n"]})
                with tempfile.TemporaryDirectory() as d:
                    p = Path(d) / "fairness_report.md"
                    p.write_text(to_markdown({**report, "models": {key: res}}))
                    mlflow.log_artifact(str(p))
    print(f"wrote {OUT_DIR / 'fairness_report.json'} and .md")


if __name__ == "__main__":
    main()
