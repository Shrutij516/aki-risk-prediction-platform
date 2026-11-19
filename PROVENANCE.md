# Provenance

This repository contains five kinds of material. Each section says what it is, when it was made and how far it can be trusted. If a file isn't listed under **1. Original research**, it is reconstruction work from November 2026.

| # | Category | When | Where its results live |
|---|---|---|---|
| 1 | Original research (TensorFlow/Keras, Colab) | Spring 2026 (Mar–May) | Original notebooks, original README and progress report (local only); public notebook copies (code and markdown only) |
| 2 | Reconstruction code | 2026-11-17 → 2026-11-19 | `aki/`, `migrations/`, `scripts/`, `tests/`, `web/`, `cpp/`, `docs/reconstruction/` |
| 3 | Synthetic demo | 2026-11-19 | Generated on demand by `scripts/demo.sh`; nothing synthetic is committed |
| 4 | Real-data evaluation (reconstructed models) | 2026-11-18 → 2026-11-19 | **Local only**: `reports/`, `mlflow.db`, `mlartifacts/`, `outputs/` (all gitignored) |
| 5 | AI-assisted implementation | 2026-11-17 → 2026-11-19 | Applies to all of category 2 (see below) |

## 1. Original research (Spring 2026)

Research project for AMS 585 (Internship in Data Science), Stony Brook University, Spring 2026, under the supervision of **Dr. Alisa Yurovsky**, as recorded in the original project README.

| Path | What it is | Notes |
|---|---|---|
| `AKI_Dialysis_Prediction-main/` | **Local only (gitignored):** the five original Colab notebooks (TensorFlow/Keras) and the original README | The notebooks' saved outputs contain patient-level rows, and the README contains figures from the extract, so neither is published. |
| `notebooks/sanitized/*.ipynb` | Public copies of the five notebooks: **code and markdown only** | Every saved output is removed. Colab identity metadata is stripped. Source text is unchanged except for logged placeholder rewrites: personal paths → `/path/to/`; the prior student's name → "a prior student researcher"; two extract-specific literals in code are withheld; count-like figures in markdown → "[figure withheld]". The list of rewritten cells is in `manifest.json`; the replacement rules themselves are kept locally. Made by `scripts/sanitize_notebooks.py`. |
| `docs/Progress report.docx` | **Local only (gitignored):** weekly progress notes, Mar–May 2026 | Includes another student's results and figures from the extract. |
| `data/*.csv` | Recovered TriNetX `eGFR_gt20_95KPatients` extract | Restricted clinical data. **Never committed or uploaded.** |

The original published results (test-set metrics from the original README) are reproduced, with notes on their sources, in `aki/eval/original_results.py`. One known discrepancy: the README's Transformer-without-labs column doesn't match that notebook's saved output; it matches progress-report screenshots.

### Relationship to a prior student researcher's work
*As stated by the project author:* earlier research by a prior student researcher on the same AKI → ESRD/dialysis question served as a **methodological reference**. That covers the outcome definition (ESRD or dialysis within 2 years), the event-sequence approach and the metrics to compare against. The author **independently implemented and extended** the pipeline **without access to the prior student's source code**.

The author's own research added:
- investigating laboratory results (curated kidney LOINC codes, clinical severity binning, and the effect of labs on each model);
- comparing predictive approaches (Transformer vs RETAIN, with and without labs, tuning focal-loss α, dropout and the threshold) against the earlier work's reported results.

The notebooks' own wording, such as "REPLICATING … PIPELINE" and "what … did", refers to reproducing that methodology and those results for comparison. It does not mean reuse of the prior student's implementation. The prior student did not contribute code to the original notebooks or to this reconstruction. Their name is replaced in the public copies; the original notebooks keep the original text locally.

**What the original work contained:**
- cohort construction (AKI N17*/584*, ESRD exclusion, 2-year ESRD-or-dialysis label, 90-day blanking);
- event tokenisation and kidney-lab severity binning;
- Keras Transformer and RETAIN models with focal loss;
- a threshold, α and dropout search;
- a single 80/20 split.

**Not in the original work** (no code, outputs or mentions found): PyTorch, C++, MLflow, fairness or subgroup analysis, FastAPI, React, PostgreSQL. Everything on that list here is category 2.

**Known properties of the original evaluation** (recorded here, not changed in the originals):
- Hyperparameters, threshold and focal α were chosen on the test set.
- There was a single 80/20 split, and TensorFlow seeds weren't set.
- Sequences kept their first (oldest) 400 events.

## 2. Reconstruction code (2026-11-17 → 2026-11-19)

| Path | What it is | Relationship to the original |
|---|---|---|
| `aki/lab_codes.py` | Kidney LOINC sets, units, plausibility ranges, severity bins | Copied verbatim from `RETAIN_WithLabs.ipynb` cell 80 |
| `migrations/001_raw_ehr.sql`, `aki/db/`, `aki/ingest/` | PostgreSQL schema, migration runner, embedded local server, streaming CSV → PostgreSQL ingestion | New |
| `migrations/002_cohort.sql` | SQL cohort and labels | Re-expressed from notebook cells 5–29; reproduces all 8 cohort counts the notebook printed |
| `aki/model/` | **PyTorch** RETAIN and Transformer, focal loss, train/val/test protocol, MLflow tracking, sequence builder | A new implementation following the Keras code. Parameter counts equal the Keras summaries. The sequence builder reproduces 17 notebook counts and the event order. Differences: `docs/reconstruction/pytorch_vs_keras.md` |
| `scripts/run_real_grid.sh`, `scripts/compare_results.py`, `aki/eval/original_results.py` | Real-data training grid; reconstructed-vs-original table | New. Original numbers copied from the README, with source notes |
| `aki/serve/` (`export.py`, `registry.py`, `scoring.py`) | Model bundles (weights plus the training-only vocabulary, rebuilt deterministically), CPU inference, batch scoring | New. No retraining; CPU re-scoring matches MLflow exactly |
| `aki/serve/export_cpp.py`, `cpp/`, `scripts/build_cpp.sh`, `scripts/cpp_parity.py` | **C++** inference: `torch.export` + AOTInductor package run by a LibTorch executable | New. One model (`transformer_withlabs`); parity with Python ≤ 9e-8 (tolerance 1e-5). `docs/reconstruction/cpp_inference.md` |
| `migrations/003_app.sql`, `aki/api/`, `web/` | App tables, FastAPI backend, React/JavaScript (Vite) dashboard | New |
| `aki/demo/`, `scripts/demo.sh`, `scripts/smoke_test.py` | Synthetic data generator, one-command demo, end-to-end check | New (see category 3) |
| `aki/eval/fairness.py` | Subgroup evaluation of the real held-out predictions | New; method and limits in `docs/reconstruction/app_and_fairness.md` |
| `scripts/profile_csvs.py`, `scripts/extra_checks.py`, `scripts/validate_cohort.py`, `scripts/sanitize_notebooks.py`, `scripts/check_commit_safety.py` | Data profiling and checks, cohort validation, notebook sanitizing, pre-commit safety check | New |
| `tests/` | 64 tests on synthetic data only | New |
| `docs/reconstruction/`, `README.md`, `pyproject.toml`, `.gitignore`, `.env.example` | Docs and project setup | New |

Changes to earlier reconstruction code are recorded in the docs. For example, on 2026-11-19 ingestion `--replace` started cascading to the app tables, and the sequence builder gained a `dates` column; tokens and order are unchanged.

## 3. Synthetic demo
- `aki/demo/synthetic_ehr.py` generates invented patients (seeded) in the TriNetX file format. Codes come from short hand-written lists of public codes. **Nothing is derived from the real data.**
- The demo runs them through the same ingestion, cohort and sequence code as the real data, into a separate database (`aki_demo`). Every API response and dashboard page labels them **synthetic**.
- **Predictions for synthetic patients have no clinical meaning,** whichever model produces them: the real-data models (local) or small models trained on synthetic data only (the fresh-clone fallback, MLflow experiment `aki-demo-synthetic`).

## 4. Real-data evaluation of the reconstructed models (local only)
- **What it is:**
  - four PyTorch models trained on the recovered TriNetX data on 2026-11-18 (MLflow experiment `aki-real-baseline`);
  - their held-out test metrics with bootstrap intervals;
  - the subgroup report (MLflow experiment `aki-fairness`).
- **Where it lives:** in `reports/`, `mlflow.db`, `mlartifacts/` and `outputs/`, which are **gitignored**. The local PostgreSQL in `~/.aki_reconstruction/pgdata` holds real patient-level rows and must stay on this machine.
- **Not a reproduction:** these are **new results**. The framework is different, selection used validation data only, the vocabulary came from training data only, and the split was 64/16/20. They don't reproduce the original Keras numbers and must never be merged into the original results table.
- **No clinical claim:** nothing here has been validated for clinical use, and the subgroup report is a descriptive audit, not proof of fairness.

## 5. AI-assisted implementation
- All category-2 code and docs were written between 2026-11-17 and 2026-11-19 by an AI coding assistant (Claude, in Claude Code), directed and reviewed by the project author.
- The original Spring 2026 research (category 1) was not AI-generated by this process.
- No design reviews took place for the reconstruction. The checks that do exist are automated tests and recorded verification runs (cohort and sequence counts, MLflow re-scoring, C++ parity).

## Rules for reconstruction work
1. Never edit files under `AKI_Dialysis_Prediction-main/` or `docs/Progress report.docx`.
2. Never call a number "original" unless it appears in a saved notebook output or the progress report, and cite where.
3. Report results from reconstructed code separately from the originals.
4. Development, tests and demos use synthetic data; real data stays in `data/` and local databases.
5. Run `scripts/check_commit_safety.py` before every commit.
