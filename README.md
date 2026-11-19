# AKI → dialysis risk: reconstruction

This repository reconstructs, in November 2026, a Spring 2026 research project that predicts ESRD or dialysis within 2 years of an acute kidney injury (AKI) from EHR event sequences. The original project was done for AMS 585 (Internship in Data Science) at Stony Brook University, supervised by Dr. Alisa Yurovsky.

| | What | When |
|---|---|---|
| **Original research** | TensorFlow/Keras Colab notebooks: cohort, event tokens, kidney-lab investigation and severity binning, Transformer and RETAIN with focal loss, compared against a prior student researcher's reported results (a methodological reference only; see PROVENANCE.md). Public copies in `notebooks/sanitized/` contain code and markdown only | Spring 2026 |
| **Reconstruction (AI-assisted)** | PostgreSQL schema and ingestion; an SQL cohort that reproduces the notebooks' cohort counts (checked locally); PyTorch re-implementations of RETAIN and the Transformer, with MLflow tracking and validation-only selection; a FastAPI + React/JavaScript dashboard; C++ (LibTorch) inference; a subgroup evaluation | 2026-11-17 → 11-19 |
| **Synthetic demo** | Invented patients that run through the real pipeline code, for demos and tests | generated on demand |
| **Real-data evaluation** | Held-out metrics and the subgroup report of the reconstructed models on the TriNetX extract | local only, not in this repo |

**[PROVENANCE.md](PROVENANCE.md)** lists every file by category. The reconstruction code was written with an AI coding assistant (Claude Code) under the author's direction. The reconstructed models' results are new results, not reproductions of the original numbers. Nothing here has been validated for clinical use.

## Quick start: synthetic demo (no real data needed)

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/), Node.js 18+ with npm.

```bash
scripts/demo.sh                 # → http://127.0.0.1:8000
```

The script:
1. installs the Python package into `.venv`;
2. starts an embedded PostgreSQL 16 (Unix socket only);
3. generates **600 synthetic patients** (`aki.demo.synthetic_ehr`) and runs them through the same ingestion, cohort and sequence code used for the real data, into a separate `aki_demo` database;
4. scores every cohort patient with the four models;
5. builds the React app and serves it from FastAPI.

The models used depend on what's available:
- **Locally**, if you have the real-data model bundles in `outputs/models/`, it uses them.
- **On a fresh clone**, it trains four small models on the synthetic data only. To force this, run `AKI_MODEL_SOURCE=synthetic scripts/demo.sh`.

Every page shows a "SYNTHETIC DEMO DATA" banner, and every API response carries `X-AKI-Dataset: synthetic`. **Scores for synthetic patients mean nothing clinically.**

Check a running demo end to end:

```bash
.venv/bin/python scripts/smoke_test.py --base http://127.0.0.1:8000
```

## Dashboard
- **Patients:** search by ID, pick a risk model, sort by risk, paginate.
- **Patient page:** demographics, the four model scores against their validation-chosen thresholds, live re-scoring, RETAIN's highest-attention events with dates, and a timeline filterable by type and observation window.
- **Model results:** held-out test metrics with 95% bootstrap CIs, confusion counts and prevalence, next to the original published (Keras) numbers.
- **Fairness:** the aggregate subgroup report, if one has been generated on this machine.

## API (FastAPI, `aki/api/main.py`)

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | DB check, dataset kind, loaded models |
| `GET /api/dataset` | Dataset kind, disclaimer, row counts |
| `GET /api/models` | Registered models: validation and test metrics, CIs, threshold, original published values |
| `GET /api/patients?q=&model=&sort=id\|risk&limit=&offset=` | Patient search / ranking |
| `GET /api/patients/{id}` | Demographics, cohort dates, observed outcome, stored predictions |
| `GET /api/patients/{id}/timeline?types=&observation_only=&limit=&offset=` | Events from `ehr.*` |
| `GET /api/patients/{id}/risk?model=` | Live CPU scoring from `app.sequence`, plus RETAIN attention |
| `GET /api/fairness` | Aggregate subgroup report (`reports/fairness/fairness_report.json`) |

The API refuses to serve a database marked `real` unless `AKI_ALLOW_REAL=1` is set, and it binds to 127.0.0.1.

## Real-data pipeline (local, authorised data only)
The real data (`data/`) never leaves this machine: databases, sequences, predictions, model weights, MLflow runs and reports are all gitignored.

```bash
python -m aki.db.local_server start
python -m aki.ingest --dsn <db> --select all --rows cohort-codes --build-cohort   # cohort (validated locally: scripts/validate_cohort.py, needs local reference counts)
python -m aki.model.sequences --dsn <db> --labs --out outputs/sequences_withlabs.parquet
python -m aki.model.experiment --data outputs/sequences_withlabs.parquet --model retain   # MLflow: sqlite:///mlflow.db
python -m aki.serve.export --experiment aki-real-baseline   # bundles + CPU re-scoring checked against MLflow
python -m aki.eval.fairness                                 # aggregate subgroup report
scripts/compare_results.py                                  # reconstructed vs original table
```

## C++ inference (LibTorch)
**The web app doesn't use C++.** FastAPI scores patients in Python (PyTorch eager, CPU, `aki/serve/registry.py`). The C++ program is a separate runtime, checked against Python, for one model.

- **Model:** the Transformer with binned labs (best by validation ROC-AUC), exported from the saved weights without retraining.
- **Export:** `torch.export` with a dynamic batch dimension, compiled ahead of time with AOTInductor into a `.pt2` package. TorchScript was avoided because it is deprecated in the PyTorch 2.15 used here.
- **Runtime:** `cpp/aki_infer.cpp` loads the package with LibTorch's `AOTIModelPackageLoader`.
- **Input and output:** it reads already-encoded, padded token-id rows (one patient per line) and prints `sigmoid(logit)` per row. Vocabulary lookup, truncation, padding, the threshold and RETAIN attention stay in Python.
- **Safety checks:** it refuses inputs whose width or batch size doesn't match what was exported (recorded in the package metadata).
- **Parity:** the largest difference from eager PyTorch is 9e-8 across synthetic, held-out and edge-case inputs at batch sizes 1 and 256 (tolerance 1e-5). `tests/test_cpp.py` repeats the check on tiny untrained Transformer and RETAIN models.
- **Limits:** one exported model; the package is tied to the PyTorch version and platform it was built on (2.15, macOS arm64, CPU); the plain-text input is slow.

```bash
python -m aki.serve.export_cpp --model transformer_withlabs   # needs the local model bundles (outputs/models/)
scripts/build_cpp.sh                                          # cmake + LibTorch from the venv's PyTorch wheel
cpp/build/aki_infer outputs/cpp/transformer_withlabs.pt2 outputs/cpp/inputs_synthetic_demo.txt
python scripts/cpp_parity.py
```

Details: [docs/reconstruction/cpp_inference.md](docs/reconstruction/cpp_inference.md).

## Tests and safety checks

```bash
.venv/bin/python -m pytest                    # 64 tests, synthetic data only; throwaway embedded PostgreSQL (C++ tests skip unless built)
(cd web && npm ci && npm run build)           # frontend build
.venv/bin/python scripts/check_commit_safety.py   # run before every commit
```

## Docs
- [docs/reconstruction/data_inventory.md](docs/reconstruction/data_inventory.md): the recovered data, and how it compares with the notebooks
- [docs/reconstruction/schema_design.md](docs/reconstruction/schema_design.md): PostgreSQL schema and ingestion
- [docs/reconstruction/pytorch_vs_keras.md](docs/reconstruction/pytorch_vs_keras.md): what matches the original models and what differs
- [docs/reconstruction/app_and_fairness.md](docs/reconstruction/app_and_fairness.md): app architecture, model integration checks, fairness method and limits
- [docs/reconstruction/cpp_inference.md](docs/reconstruction/cpp_inference.md): C++ export, build, parity checks and limits
