"""Fairness metric code on synthetic predictions with a known, built-in subgroup gap."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aki.eval.fairness import MIN_CELL, age_band, evaluate_model, to_markdown


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    # Group A: informative scores. Group B: pure noise. Group C: tiny (must be suppressed).
    for group, n, informative in (("A", 2000, True), ("B", 800, False), ("C", 30, True)):
        y = (rng.random(n) < 0.15).astype(int)
        p = np.clip(0.15 + (0.25 * (y - 0.15) if informative else 0) + rng.normal(0, 0.1, n), 0, 1)
        rows.append(pd.DataFrame({"label": y, "prob": p, "race": group, "sex": rng.choice(["F", "M"], n),
                                  "ethnicity": "X", "age_band": "45-64"}))
    return pd.concat(rows, ignore_index=True)


def test_detects_built_in_gap_and_overall_matches_sklearn():
    df = _frame()
    r = evaluate_model(df, threshold=0.2, n_boot=200)
    assert abs(r["overall"]["roc_auc"] - roc_auc_score(df["label"], df["prob"])) < 1e-12
    g = {x["group"]: x for x in r["attributes"]["race"]}
    assert g["A"]["roc_auc"] > 0.8 and abs(g["B"]["roc_auc"] - 0.5) < 0.08
    d = g["B"]["diff_vs_reference"]["roc_auc"]
    assert g["B"]["reference_group"] == "A" and d["ci95"][1] < 0  # gap detected, CI excludes 0
    sex = r["attributes"]["sex"]
    assert all(not x["suppressed"] for x in sex)


def test_small_groups_are_suppressed_without_revealing_size():
    r = evaluate_model(_frame(), threshold=0.2, n_boot=50)
    c = next(x for x in r["attributes"]["race"] if x["group"] == "C")
    assert c["suppressed"] and "n" not in c and "roc_auc" not in c
    df = _frame()
    df.loc[df["race"] == "B", "label"] = 0
    df.loc[df.index[df["race"] == "B"][: MIN_CELL - 1], "label"] = 1  # 10 positives -> suppressed
    b = next(x for x in evaluate_model(df, 0.2, n_boot=20)["attributes"]["race"] if x["group"] == "B")
    assert b["suppressed"] and "positives" in b["reason"]


def test_age_bands_and_markdown():
    assert [age_band(a) for a in (30, 45, 64, 65, 74, 75, None)] == ["<45", "45-64", "45-64", "65-74", "65-74", "75+", "Unknown"]
    r = evaluate_model(_frame(), 0.2, n_boot=20)
    md = to_markdown({"data_label": "x", "suppression": {"min_n": 50, "min_cell": 11}, "limitations": ["l"],
                      "models": {"m": r}})
    assert "suppressed" in md and "| race | A |" in md
