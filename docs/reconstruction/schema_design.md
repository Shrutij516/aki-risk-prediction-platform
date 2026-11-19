# PostgreSQL schema and ingestion design

*Reconstruction document (2026-11-17, public version: figures from the TriNetX extract removed; detailed version kept locally). Implemented in `migrations/` and `aki/`.*

## Layers

```
meta.schema_migrations   applied migrations + sha256 (runner refuses edited migrations)
meta.ingest_run          one row per ingestion: selection JSON, status, code version, error
meta.ingest_file         per file: rows loaded, bad dates, bad numerics, orphans, seconds

ehr.patient              ← patient.csv          PK patient_id
ehr.diagnosis            ← diagnosis.csv        FK patient_id
ehr.procedure            ← procedure.csv        FK patient_id
ehr.medication_drug      ← medication_drug.csv  FK patient_id
ehr.lab_result           ← lab_result_clean.csv (content-identical to lab_result.csv)

cohort.aki_cohort                  index date, windows, 3 labels (built by cohort.build_aki_cohort())
cohort.excluded_preexisting_esrd   audit trail for the ESRD exclusion
```

Planned later, not built yet: `model.*` (MLflow run ids, predictions, attention weights) and `app.*`.

## Design decisions

| Decision | Why |
|---|---|
| Raw tables mirror the CSV headers 1:1 (lower-cased) | Lossless and traceable. Every modelling transformation happens downstream and can be audited. |
| Codes, IDs and flags stay `TEXT` | Keeps leading zeros (NDC `00003089421`, CPT `00532`). Flags have 3 values (`U`/`F`/`T`), so they aren't booleans. |
| `YYYYMMDD` → `DATE`; values → `DOUBLE PRECISION`/`NUMERIC` | The profile showed 0 parse failures. Failures would still be stored as NULL and counted in `meta.ingest_file`. |
| No deduplication or filtering in `ehr.*` | Exact duplicate rows and NDC/RxNorm twin prescriptions exist in the extract; handling them is a modelling question, not an ingestion question. |
| Surrogate identity PKs on event tables | There is no natural key: exact duplicate rows exist. |
| Patient FK enforced | Patients load first; the profile shows 0 orphans. |
| Whole-patient sampling (`md5(patient_id ‖ seed)`) | Each loaded patient has complete history, so cohort and labels are exact per patient. Sampling is deterministic. |
| Cohort logic in SQL, rules cited to notebook cells | One definition serves both the API and the training pipeline. Validated against all 8 notebook counts. |
| One transaction per ingestion | A failed or cancelled run leaves no partial data and is marked `failed` with the error. This was exercised on real data when a slow query was cancelled. |
| DuckDB → Arrow batches → `COPY` | Streams multi-GB CSVs with bounded memory (4 GB DuckDB cap, 200K-row batches). |

## Ingestion modes

| Command | Loads | Measured on recovered data |
|---|---|---|
| `python -m aki.ingest --select all --rows cohort-codes --build-cohort` | All patients, but only AKI/ESRD diagnoses and dialysis-code procedures | Fast (seconds); reproduces all notebook cohort counts |
| `python -m aki.ingest --select aki-sample --n 200 --replace --build-cohort` | 200 AKI patients, full history in all tables | Seconds; small local database |
| `python -m aki.ingest --select all` | Everything | **Refused by the disk guard** on the development machine: the full extract needs more free disk than was available |

## Local server

`python -m aki.db.local_server start` runs PostgreSQL 16.2 (pgserver wheel) on a Unix socket only, with no TCP port. Its data directory is `~/.aki_reconstruction/pgdata` (override with `AKI_PGDATA`). It sits outside the project because pgserver can't start from a path containing spaces. It holds patient-level data and must stay on this machine. Any PostgreSQL 16 server can be used instead through `AKI_DATABASE_URL`.
