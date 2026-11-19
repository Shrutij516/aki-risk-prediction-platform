import datetime as dt

import psycopg
import pytest

from aki.ingest.trinetx import IngestConfig, ingest
from tests.conftest import (DIAGNOSES, EXPECTED_COHORT, EXPECTED_EXCLUDED, LABS, MEDICATIONS, PATIENTS,
                            PROCEDURES)


def _count(dsn, table):
    with psycopg.connect(dsn) as c:
        return c.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_full_load_row_counts_and_bookkeeping(migrated_dsn, synth_dir):
    s = ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="all"))
    assert s["patients_selected"] == len(PATIENTS)
    assert _count(migrated_dsn, "ehr.patient") == len(PATIENTS)
    assert _count(migrated_dsn, "ehr.diagnosis") == len(DIAGNOSES)
    assert _count(migrated_dsn, "ehr.procedure") == len(PROCEDURES)
    assert _count(migrated_dsn, "ehr.medication_drug") == len(MEDICATIONS)
    assert _count(migrated_dsn, "ehr.lab_result") == len(LABS)
    files = {f["source_file"]: f for f in s["files"]}
    assert files["diagnosis.csv"]["rows_bad_date"] == 1            # P09 '20201345'
    assert files["lab_result_clean.csv"]["rows_bad_numeric"] == 1  # 'not-a-number'
    assert files["diagnosis.csv"]["rows_orphan_patient"] == 0
    with psycopg.connect(migrated_dsn) as c:
        assert c.execute("SELECT status FROM meta.ingest_run WHERE run_id=%s", (s["run_id"],)).fetchone()[0] == "succeeded"
        assert c.execute("SELECT count(*) FROM meta.ingest_file WHERE run_id=%s", (s["run_id"],)).fetchone()[0] == 5


def test_values_preserved(migrated_dsn, synth_dir):
    ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="all"))
    with psycopg.connect(migrated_dsn) as c:
        # leading zeros kept on codes
        assert c.execute("SELECT count(*) FROM ehr.procedure WHERE code='00532'").fetchone()[0] == 1
        assert c.execute("SELECT count(*) FROM ehr.medication_drug WHERE code='00003089421'").fetchone()[0] == 1
        # dates parsed, unparseable -> NULL
        assert c.execute("SELECT date FROM ehr.diagnosis WHERE patient_id='P01' AND code='N17.9'").fetchone()[0] == dt.date(2020, 1, 10)
        assert c.execute("SELECT date FROM ehr.diagnosis WHERE patient_id='P09'").fetchone()[0] is None
        # empty strings become NULL; numbers parsed
        assert c.execute("SELECT code_system FROM ehr.diagnosis WHERE code='3827|1'").fetchone()[0] is None
        assert c.execute("SELECT quantity_dispensed FROM ehr.medication_drug WHERE unique_id='u1'").fetchone()[0] == 60
        assert c.execute("SELECT lab_result_num_val FROM ehr.lab_result WHERE code='2160-0'").fetchone()[0] == 1.35
        assert c.execute("SELECT units_of_measure FROM ehr.lab_result WHERE code='33914-3'").fetchone()[0] == "mL/min/{1.73_m2}"
        # patient fields
        assert c.execute("SELECT year_of_birth FROM ehr.patient WHERE patient_id='P06'").fetchone()[0] is None
        assert c.execute("SELECT month_year_death FROM ehr.patient WHERE patient_id='P02'").fetchone()[0] == "202203"


def test_raw_and_clean_lab_files_load_identically(migrated_dsn, synth_dir):
    ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="all", tables=("lab_result",)))
    with psycopg.connect(migrated_dsn) as c:
        clean = c.execute("SELECT patient_id, code, date, lab_result_num_val, units_of_measure FROM ehr.lab_result ORDER BY 1,2").fetchall()
    ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="all", tables=("lab_result",), replace=True,
                                      files={"lab_result": "lab_result.csv"}))
    with psycopg.connect(migrated_dsn) as c:
        raw = c.execute("SELECT patient_id, code, date, lab_result_num_val, units_of_measure FROM ehr.lab_result ORDER BY 1,2").fetchall()
    assert raw == clean


def test_cohort_matches_notebook_rules(migrated_dsn, synth_dir):
    s = ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="all", build_cohort=True))
    assert s["cohort_rows"] == len(EXPECTED_COHORT)
    with psycopg.connect(migrated_dsn) as c:
        rows = c.execute("""SELECT patient_id, aki_index_date::text, prediction_end_date::text, blank_start_date::text,
                                   y_esrd_2yr, y_dialysis_2yr, y_esrd_or_dialysis_2yr FROM cohort.aki_cohort""").fetchall()
        excluded = {r[0] for r in c.execute("SELECT patient_id FROM cohort.excluded_preexisting_esrd")}
    assert {r[0]: tuple(r[1:]) for r in rows} == EXPECTED_COHORT
    assert excluded == EXPECTED_EXCLUDED


def test_cohort_codes_scope_gives_same_cohort(migrated_dsn, synth_dir):
    ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="all", build_cohort=True))
    with psycopg.connect(migrated_dsn) as c:
        full = c.execute("SELECT * FROM cohort.aki_cohort ORDER BY patient_id").fetchall()
    s = ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="all", rows="cohort-codes",
                                          build_cohort=True, replace=True))
    with psycopg.connect(migrated_dsn) as c:
        slim = c.execute("SELECT * FROM cohort.aki_cohort ORDER BY patient_id").fetchall()
    strip = lambda rows: [r[:-2] for r in rows]  # drop definition, built_at
    assert strip(slim) == strip(full)
    assert {f["source_file"] for f in s["files"]} == {"patient.csv", "diagnosis.csv", "procedure.csv"}
    assert _count(migrated_dsn, "ehr.medication_drug") == 0
    assert _count(migrated_dsn, "ehr.diagnosis") < len(DIAGNOSES)  # I10, E11.9, 3827|1 filtered out


def test_sampling_is_deterministic_and_whole_patient(migrated_dsn, synth_dir):
    ids = []
    for _ in range(2):
        ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="sample", n=4, seed=7, replace=True))
        with psycopg.connect(migrated_dsn) as c:
            ids.append(sorted(r[0] for r in c.execute("SELECT patient_id FROM ehr.patient")))
            # every diagnosis row of each sampled patient is present
            loaded = c.execute("SELECT count(*) FROM ehr.diagnosis").fetchone()[0]
    assert ids[0] == ids[1] and len(ids[0]) == 4
    assert loaded == sum(1 for r in DIAGNOSES if r[0] in ids[0])


def test_aki_sample_only_returns_aki_patients(migrated_dsn, synth_dir):
    ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="aki-sample", n=100))
    with psycopg.connect(migrated_dsn) as c:
        got = {r[0] for r in c.execute("SELECT patient_id FROM ehr.patient")}
    # P09 has an AKI code (unparseable date) so it is selected; P06/P07 are not AKI.
    assert got == {"P01", "P02", "P03", "P04", "P05", "P08", "P09", "P10"}


def test_second_load_requires_replace_and_failed_run_is_recorded(migrated_dsn, synth_dir):
    ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="all"))
    with pytest.raises(RuntimeError, match="already has"):
        ingest(migrated_dsn, IngestConfig(data_dir=synth_dir, select="all"))
    with psycopg.connect(migrated_dsn) as c:
        statuses = [r[0] for r in c.execute("SELECT status FROM meta.ingest_run ORDER BY run_id")]
        assert statuses == ["succeeded", "failed"]
        # failed run rolled back fully: original data intact
        assert c.execute("SELECT count(*) FROM ehr.patient").fetchone()[0] == len(PATIENTS)
