-- 003_app: tables the web application reads.
--
--  app.dataset_info  one row saying whether this database holds SYNTHETIC or REAL data;
--                    the API shows it on every response and refuses REAL unless explicitly allowed.
--  app.sequence      model-input token sequences (built by aki.model.sequences, the same code
--                    used for training), so the API can score patients live.
--  app.model         registered model bundles + their aggregate validation/test metrics
--                    (test metrics are from the REAL held-out evaluation when source = 'real-trained').
--  app.prediction    batch-scored probabilities for the patients in THIS database.

CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE app.dataset_info (
    singleton    BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    kind         TEXT NOT NULL CHECK (kind IN ('synthetic', 'real')),
    description  TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE app.sequence (
    patient_id  TEXT NOT NULL REFERENCES ehr.patient(patient_id) ON DELETE CASCADE,
    variant     TEXT NOT NULL CHECK (variant IN ('nolabs', 'withlabs')),
    tokens      TEXT[] NOT NULL,
    dates       DATE[] NOT NULL,
    built_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (patient_id, variant),
    CHECK (cardinality(tokens) = cardinality(dates))
);

CREATE TABLE app.model (
    model_key       TEXT PRIMARY KEY,
    model           TEXT NOT NULL CHECK (model IN ('retain', 'transformer')),
    variant         TEXT NOT NULL CHECK (variant IN ('nolabs', 'withlabs')),
    source          TEXT NOT NULL CHECK (source IN ('real-trained', 'synthetic-trained')),
    mlflow_run_id   TEXT,
    threshold       DOUBLE PRECISION NOT NULL,
    maxlen          INTEGER NOT NULL,
    truncate        TEXT NOT NULL,
    focal_alpha     DOUBLE PRECISION,
    dropout         DOUBLE PRECISION,
    val_metrics     JSONB,
    test_metrics    JSONB,
    split           JSONB,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE app.prediction (
    patient_id       TEXT NOT NULL REFERENCES ehr.patient(patient_id) ON DELETE CASCADE,
    model_key        TEXT NOT NULL REFERENCES app.model(model_key) ON DELETE CASCADE,
    probability      DOUBLE PRECISION NOT NULL CHECK (probability BETWEEN 0 AND 1),
    threshold        DOUBLE PRECISION NOT NULL,
    above_threshold  BOOLEAN NOT NULL,
    n_events         INTEGER NOT NULL,
    n_used           INTEGER NOT NULL,
    n_oov            INTEGER NOT NULL,
    attention        JSONB,
    scored_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (patient_id, model_key)
);
CREATE INDEX prediction_model_prob_idx ON app.prediction (model_key, probability DESC);
