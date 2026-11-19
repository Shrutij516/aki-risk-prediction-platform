-- 002_cohort: AKI cohort + 2-year ESRD/dialysis labels, re-expressed in SQL.
--
-- Source of every rule: AKI_Dialysis_Prediction-main/RETAIN_WithLabs.ipynb
--   cells 5-8   AKI index = earliest dated diagnosis with
--               (ICD-10-CM, code LIKE 'N17%') or (ICD-9-CM, code LIKE '584%')
--   cells 10-14 exclude patients with ESRD (ICD-10-CM 'N18.6' | ICD-9-CM '585.6', exact)
--               dated ON OR BEFORE the index date
--   cell 18     prediction_end_date = index + 2 years (pd.DateOffset(years=2))
--   cell 20     y_esrd_2yr: ESRD date  > index AND <= end      (strictly after index)
--   cells 24-26 y_dialysis_2yr: dialysis procedure date >= index AND <= end (index day included)
--               CPT 90935/90937/90945/90947, HCPCS G0257, ICD-10-PCS prefix 5A1D, ICD-9 39.95;
--               code_system matched case-insensitively by substring, code trimmed
--   cell 27     y_esrd_or_dialysis_2yr = OR of the two
--   cell 29     blank_start_date = index - 90 days; model inputs are events dated < blank_start_date
--
-- Deliberate fidelity notes (kept as in the original, not "fixed"):
--   * the > vs >= asymmetry between the ESRD and dialysis windows;
--   * diagnosis codes are not trimmed (notebook did not trim them); procedure codes are.
-- Cohort membership is exact for every loaded patient because ingestion samples
-- whole patients, never individual rows.

CREATE SCHEMA IF NOT EXISTS cohort;

CREATE TABLE cohort.aki_cohort (
    patient_id              TEXT PRIMARY KEY REFERENCES ehr.patient(patient_id),
    aki_index_date          DATE NOT NULL,
    prediction_end_date     DATE NOT NULL,
    blank_start_date        DATE NOT NULL,
    y_esrd_2yr              BOOLEAN NOT NULL,
    y_dialysis_2yr          BOOLEAN NOT NULL,
    y_esrd_or_dialysis_2yr  BOOLEAN NOT NULL,
    definition              TEXT NOT NULL DEFAULT 'retain_withlabs_v1',
    built_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Patients with AKI who were excluded for pre-existing ESRD (kept for auditability).
CREATE TABLE cohort.excluded_preexisting_esrd (
    patient_id          TEXT PRIMARY KEY REFERENCES ehr.patient(patient_id),
    aki_index_date      DATE NOT NULL,
    first_esrd_date     DATE NOT NULL
);

CREATE OR REPLACE FUNCTION cohort.build_aki_cohort() RETURNS INTEGER
LANGUAGE plpgsql AS $$
DECLARE n INTEGER;
BEGIN
    TRUNCATE cohort.aki_cohort, cohort.excluded_preexisting_esrd;

    CREATE TEMP TABLE _aki_index ON COMMIT DROP AS
    SELECT patient_id, min(date) AS aki_index_date
    FROM ehr.diagnosis
    WHERE date IS NOT NULL
      AND ((code_system = 'ICD-10-CM' AND code LIKE 'N17%')
        OR (code_system = 'ICD-9-CM'  AND code LIKE '584%'))
    GROUP BY patient_id;

    CREATE TEMP TABLE _esrd ON COMMIT DROP AS
    SELECT patient_id, date
    FROM ehr.diagnosis
    WHERE date IS NOT NULL
      AND ((code_system = 'ICD-10-CM' AND code = 'N18.6')
        OR (code_system = 'ICD-9-CM'  AND code = '585.6'));

    INSERT INTO cohort.excluded_preexisting_esrd
    SELECT a.patient_id, a.aki_index_date, min(e.date)
    FROM _aki_index a JOIN _esrd e USING (patient_id)
    WHERE e.date <= a.aki_index_date
    GROUP BY a.patient_id, a.aki_index_date;

    CREATE TEMP TABLE _dialysis ON COMMIT DROP AS
    SELECT patient_id, date
    FROM (SELECT patient_id, date, upper(btrim(code_system)) AS cs, btrim(code) AS c
          FROM ehr.procedure WHERE date IS NOT NULL) p
    WHERE (cs LIKE '%CPT%'   AND c IN ('90935', '90937', '90945', '90947'))
       OR (cs LIKE '%HCPCS%' AND c = 'G0257')
       OR (cs LIKE '%ICD-10%' AND cs LIKE '%PCS%' AND c LIKE '5A1D%')
       OR (cs LIKE '%ICD-9%' AND c = '39.95');

    -- Set-based (hash joins); a correlated EXISTS per patient was quadratic on the full extract.
    CREATE TEMP TABLE _clean ON COMMIT DROP AS
    SELECT a.patient_id, a.aki_index_date, (a.aki_index_date + INTERVAL '2 years')::date AS end_date
    FROM _aki_index a
    LEFT JOIN cohort.excluded_preexisting_esrd x USING (patient_id)
    WHERE x.patient_id IS NULL;
    ANALYZE _clean; ANALYZE _esrd; ANALYZE _dialysis;

    INSERT INTO cohort.aki_cohort (patient_id, aki_index_date, prediction_end_date, blank_start_date,
                                   y_esrd_2yr, y_dialysis_2yr, y_esrd_or_dialysis_2yr)
    SELECT c.patient_id, c.aki_index_date, c.end_date, c.aki_index_date - 90,
           e.patient_id IS NOT NULL, d.patient_id IS NOT NULL,
           e.patient_id IS NOT NULL OR d.patient_id IS NOT NULL
    FROM _clean c
    LEFT JOIN (SELECT DISTINCT x.patient_id FROM _esrd x JOIN _clean c2 USING (patient_id)
               WHERE x.date > c2.aki_index_date AND x.date <= c2.end_date) e USING (patient_id)
    LEFT JOIN (SELECT DISTINCT x.patient_id FROM _dialysis x JOIN _clean c2 USING (patient_id)
               WHERE x.date >= c2.aki_index_date AND x.date <= c2.end_date) d USING (patient_id);

    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n;
END $$;
