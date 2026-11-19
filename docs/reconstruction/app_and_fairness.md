# Application, model integration and subgroup evaluation

*Reconstruction document (2026-11-19). All of this is new work; the original project had no application, no PostgreSQL and no fairness analysis (see PROVENANCE.md).*

## Architecture

```
synthetic_ehr.py ─► TriNetX-format CSVs ─► aki.ingest ─► PostgreSQL (aki_demo)
                                                          ├─ ehr.*      raw events
                                                          ├─ cohort.*   SQL cohort + labels
                                                          └─ app.*      sequences, models, predictions, dataset kind
outputs/models/<key>/ (weights + train-only vocab + meta) ─► aki.serve.registry (CPU) ─► FastAPI ─► React (Vite build)
```

- **Same code as the research pipeline.** The demo uses the ingestion, SQL cohort and sequence builder that were validated against the notebooks. Nothing is re-implemented for the app.
- **`app.dataset_info`** marks a database as `synthetic` or `real`. The API puts that label on every response, and won't serve `real` without `AKI_ALLOW_REAL=1`.
- **Two ways to score.** `app.prediction` holds batch scores. `/risk` re-scores live from `app.sequence`, and the smoke test checks that the live and stored scores agree.

## Model integration checks

| Check | Result | Where |
|---|---|---|
| Vocabulary rebuilt from the saved sequences with the same split, matching the logged vocabulary size and data fingerprint | Passed for all 4 models | `aki/serve/export.py` |
| CPU re-scoring of the held-out test set vs the MLflow-logged metrics (training ran on MPS) | ROC-AUC and PR-AUC identical to 4 decimals; confusion counts identical for all 4 models | `reports/export_verification_aki-real-baseline.json` |
| Single-patient scoring (API path) vs batch prediction | Max difference 1.2e-7 over 200 test patients per model | checked during the build |
| Live API score vs stored batch score | Equal within 1e-5 for every model | `scripts/smoke_test.py`, `tests/test_api.py` |

**Synthetic patients and real-trained models.** When the real-data models score synthetic patients, every synthetic code is in the real vocabulary (0% unseen codes), so the scoring path really runs. But the patients are invented, so **the scores have no meaning.** The demo labels them that way everywhere.

**Synthetic-trained fallback.** On a machine without the real weights, `aki.demo.build --model-source synthetic` trains one setting per model and variant on the synthetic data (3 epochs, CPU, MLflow experiment `aki-demo-synthetic`) and serves those. Their metrics describe synthetic data only.

## Subgroup ("fairness") evaluation: method

- **Data:** the held-out test predictions of the four real-data models (`outputs/predictions/*_test.parquet`, local only). These are the same predictions as the MLflow evaluation, and each model's overall ROC-AUC is checked against the logged value before any group is analysed.
- **Attributes:** sex, race and ethnicity as recorded in the EHR extract, and age band at the AKI index date (from year of birth). These are the only demographics available.
- **Per group:**
  - n, prevalence, ROC-AUC and PR-AUC;
  - TPR, FPR and precision at the model's validation-selected threshold;
  - mean predicted risk vs the observed rate;
  - 95% percentile bootstrap CIs (1,000 resamples within each group);
  - the difference from the largest group, with a 95% CI from independent bootstrap resamples.
- **Suppression:** a group isn't reported, and its size isn't shown, if n < 50 or it has fewer than 11 positives or negatives.
- **Output:** aggregate only. `reports/fairness/fairness_report.{json,md}` (gitignored) and one MLflow run per model in experiment `aki-fairness`. No training run is modified.
- **Tests:** `tests/test_fairness.py` uses synthetic predictions with a built-in gap. It checks that the gap is detected, that the overall numbers equal sklearn's, and that small groups are suppressed without revealing their size.

## Subgroup evaluation: limitations
These are written into every report:
- Demographics are as recorded in the EHR and have large "Unknown" groups.
- The inputs include eGFR codes from race-based equations, so race can reach the model through its inputs.
- Groups have small positive counts, so intervals are wide.
- There is one split and no external validation.
- One threshold is used for all groups.
- There is no intersectional analysis.
- Age comes from year of birth only.
- Focal-loss outputs are not calibrated.
- Many comparisons are made with no correction.
- This is a descriptive audit: it does **not** show that the models are fair or fit for clinical use.

## Not implemented
- **C++ in the app:** the API scores in Python. C++ inference exists as a separate, verified program (`docs/reconstruction/cpp_inference.md`) and isn't wired into the dashboard.
- **Serving real patients through the dashboard:** possible locally with `AKI_ALLOW_REAL=1`, but no real database has been prepared for the app, and real data must never appear in screenshots or public demos.
- **Transformer explanations:** none. Only RETAIN shows per-event attention, and attention is not a causal explanation.
- **Authentication:** none. The app is for local demos only and binds to 127.0.0.1.
