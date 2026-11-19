-- 001_raw_ehr: raw TriNetX tables, one per source CSV, plus ingestion bookkeeping.
--
-- Design (see docs/reconstruction/schema_design.md):
--  * Columns mirror the CSV headers 1:1 (lower-cased) so nothing is lost or renamed.
--  * Only types that the data profile proved safe are converted:
--      YYYYMMDD strings -> DATE (0 unparseable values in all files)
--      numeric value columns -> DOUBLE PRECISION / NUMERIC
--    Everything else (codes, flags like 'U'/'F'/'T', IDs) stays TEXT; codes keep
--    leading zeros (e.g. NDC '00003089421', CPT '00532').
--  * Rows are never deduplicated or filtered here; that is a modelling decision.
--  * patient_id foreign keys are enforced: ingestion loads ehr.patient first and
--    only loads events for patients present in it (orphans are counted and skipped).

CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS ehr;

CREATE TABLE meta.ingest_run (
    run_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'succeeded', 'failed')),
    data_dir        TEXT NOT NULL,
    selection       JSONB NOT NULL,          -- how patients were chosen (mode, n, seed, ...)
    code_version    TEXT,
    notes           TEXT
);

CREATE TABLE meta.ingest_file (
    run_id              BIGINT NOT NULL REFERENCES meta.ingest_run(run_id) ON DELETE CASCADE,
    source_file         TEXT NOT NULL,
    target_table        TEXT NOT NULL,
    file_bytes          BIGINT NOT NULL,
    file_mtime          TIMESTAMPTZ NOT NULL,
    rows_loaded         BIGINT NOT NULL,
    rows_bad_date       BIGINT NOT NULL,     -- non-empty date that failed YYYYMMDD parse (stored as NULL)
    rows_bad_numeric    BIGINT NOT NULL,     -- non-empty numeric field that failed to parse (stored as NULL)
    rows_orphan_patient BIGINT,              -- event rows whose patient_id is not in patient.csv (skipped); only computed when selection = all
    seconds             DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, source_file)
);

CREATE TABLE ehr.patient (
    patient_id                 TEXT PRIMARY KEY,
    sex                        TEXT,
    race                       TEXT,
    ethnicity                  TEXT,
    marital_status             TEXT,
    year_of_birth              SMALLINT,
    reason_yob_missing         TEXT,
    month_year_death           TEXT,         -- 'YYYYMM' in source (often empty); kept verbatim, no day available
    death_date_source_id       TEXT,
    patient_regional_location  TEXT,
    source_id                  TEXT,
    ingest_run_id              BIGINT NOT NULL REFERENCES meta.ingest_run(run_id)
);

CREATE TABLE ehr.diagnosis (
    diagnosis_id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    patient_id                     TEXT NOT NULL REFERENCES ehr.patient(patient_id),
    encounter_id                   TEXT,
    code_system                    TEXT,     -- 'ICD-10-CM' | 'ICD-9-CM' | NULL (a few TriNetX-derived rows)
    code                           TEXT NOT NULL,
    principal_diagnosis_indicator  TEXT,
    admitting_diagnosis            TEXT,
    reason_for_visit               TEXT,
    date                           DATE,
    derived_by_trinetx             TEXT,
    source_id                      TEXT,
    ingest_run_id                  BIGINT NOT NULL REFERENCES meta.ingest_run(run_id)
);

CREATE TABLE ehr.procedure (
    procedure_id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    patient_id                     TEXT NOT NULL REFERENCES ehr.patient(patient_id),
    encounter_id                   TEXT,
    code_system                    TEXT,     -- CPT | HCPCS | SNOMED | ICD-10-PCS | ICD-9-CM
    code                           TEXT NOT NULL,
    principal_procedure_indicator  TEXT,
    date                           DATE,
    derived_by_trinetx             TEXT,
    source_id                      TEXT,
    ingest_run_id                  BIGINT NOT NULL REFERENCES meta.ingest_run(run_id)
);

CREATE TABLE ehr.medication_drug (
    medication_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    patient_id           TEXT NOT NULL REFERENCES ehr.patient(patient_id),
    encounter_id         TEXT,
    unique_id            TEXT,
    code_system          TEXT,               -- NDC | RxNorm | ... (same order can appear under both)
    code                 TEXT NOT NULL,
    start_date           DATE,
    route                TEXT,
    brand                TEXT,
    strength             TEXT,
    quantity_dispensed   NUMERIC,
    days_supply          NUMERIC,
    derived_by_trinetx   TEXT,
    source_id            TEXT,
    ingest_run_id        BIGINT NOT NULL REFERENCES meta.ingest_run(run_id)
);

CREATE TABLE ehr.lab_result (
    lab_result_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    patient_id           TEXT NOT NULL REFERENCES ehr.patient(patient_id),
    encounter_id         TEXT,
    code_system          TEXT,               -- LOINC | TNX
    code                 TEXT NOT NULL,
    date                 DATE,
    lab_result_num_val   DOUBLE PRECISION,   -- raw; source includes implausible extremes, plausibility is downstream
    lab_result_text_val  TEXT,
    units_of_measure     TEXT,
    derived_by_trinetx   TEXT,
    source_id            TEXT,
    ingest_run_id        BIGINT NOT NULL REFERENCES meta.ingest_run(run_id)
);

-- Access paths used by cohort construction and per-patient timelines.
CREATE INDEX diagnosis_patient_date_idx   ON ehr.diagnosis (patient_id, date);
CREATE INDEX diagnosis_code_idx           ON ehr.diagnosis (code_system, code);
CREATE INDEX procedure_patient_date_idx   ON ehr.procedure (patient_id, date);
CREATE INDEX procedure_code_idx           ON ehr.procedure (code_system, code);
CREATE INDEX medication_patient_date_idx  ON ehr.medication_drug (patient_id, start_date);
CREATE INDEX lab_patient_date_idx         ON ehr.lab_result (patient_id, date);
CREATE INDEX lab_code_idx                 ON ehr.lab_result (code);
