"""Test fixtures: a throwaway embedded PostgreSQL and synthetic TriNetX-shaped CSVs.

No real data is read by any test. Each test gets its own fresh database.
"""
from __future__ import annotations

import csv
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

# ------------------------------------------------------------ synthetic data
# Hand-written cases chosen to exercise every cohort rule from RETAIN_WithLabs.ipynb.
PATIENT_HDR = ["patient_id", "sex", "race", "ethnicity", "marital_status", "year_of_birth", "reason_yob_missing",
               "month_year_death", "death_date_source_id", "patient_regional_location", "source_id"]
DIAG_HDR = ["patient_id", "encounter_id", "code_system", "code", "principal_diagnosis_indicator",
            "admitting_diagnosis", "reason_for_visit", "date", "derived_by_TriNetX", "source_id"]
PROC_HDR = ["patient_id", "encounter_id", "code_system", "code", "principal_procedure_indicator", "date",
            "derived_by_TriNetX", "source_id"]
MED_HDR = ["patient_id", "encounter_id", "unique_id", "code_system", "code", "start_date", "route", "brand",
           "strength", "quantity_dispensed", "days_supply", "derived_by_TriNetX", "source_id"]
LAB_HDR = ["patient_id", "encounter_id", "code_system", "code", "date", "lab_result_num_val",
           "lab_result_text_val", "units_of_measure", "derived_by_TriNetX", "source_id"]

PATIENTS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]


def _dx(pid, system, code, date, enc="E1"):
    return [pid, enc, system, code, "Unknown", "U", "U", date, "F", "EHR"]


def _px(pid, system, code, date, enc="E1"):
    return [pid, enc, system, code, "Unknown", date, "F", "EHR"]


DIAGNOSES = [
    # P01: AKI ICD-10, index = earliest AKI date; ESRD 17 months later -> y_esrd
    _dx("P01", "ICD-10-CM", "N17.0", "20200501"), _dx("P01", "ICD-10-CM", "N17.9", "20200110"),
    _dx("P01", "ICD-10-CM", "N18.6", "20210601"), _dx("P01", "ICD-10-CM", "I10", "20190101"),
    # P02: AKI ICD-9 with ESRD on the index day -> excluded (<= index)
    _dx("P02", "ICD-9-CM", "584.9", "20120301"), _dx("P02", "ICD-9-CM", "585.6", "20120301"),
    # P03: AKI; dialysis on the index day (>= index) -> y_dialysis; ESRD after window -> no y_esrd
    _dx("P03", "ICD-10-CM", "N17.9", "20190201"), _dx("P03", "ICD-10-CM", "N18.6", "20210202"),
    # P04: ESRD exactly at index + 2y (inclusive) -> y_esrd; dialysis one day after window -> no
    _dx("P04", "ICD-10-CM", "N17.9", "20180101"), _dx("P04", "ICD-10-CM", "N18.6", "20200101"),
    # P05: leap-day index, end = 2018-02-28 (pd.DateOffset semantics)
    _dx("P05", "ICD-10-CM", "N17.9", "20160229"),
    # P06: no AKI
    _dx("P06", "ICD-10-CM", "E11.9", "20200101"),
    # P07: N17 code under the wrong code system -> not AKI
    _dx("P07", "ICD-9-CM", "N17.9", "20200101"),
    # P08: ESRD the day before AKI -> excluded
    _dx("P08", "ICD-10-CM", "N17.9", "20200101"), _dx("P08", "ICD-10-CM", "N18.6", "20191231"),
    # P09: only AKI row has an unparseable date -> stored NULL, counted, no cohort entry
    _dx("P09", "ICD-10-CM", "N17.9", "20201345"),
    # P10: AKI ICD-9; ICD-9 dialysis procedure 39.95 within window
    _dx("P10", "ICD-9-CM", "584.5", "20150505"),
    # row without code_system (exists in real data, e.g. TriNetX-derived '3827|1')
    ["P01", "", "", "3827|1", "Unknown", "U", "U", "20200110", "T", "TriNetX"],
]
PROCEDURES = [
    _px("P03", "CPT", "90935", "20190201"),
    _px("P04", "ICD-10-PCS", "5A1D70Z", "20200102"),
    _px("P05", "HCPCS", "G0257", "20180228"),
    _px("P10", "ICD-9-CM", "39.95", "20150601"),
    _px("P01", "CPT", "00532", "20190101"),  # leading zero must survive
]
MEDICATIONS = [
    ["P01", "E1", "u1", "NDC", "00003089421", "20210921", "Oral Product", "Eliquis", "Unknown", "60.000000", "", "F", "EHR"],
    ["P01", "E1", "u2", "RxNorm", "1364447", "20210921", "Oral Product", "Eliquis", "Unknown", "60.000000", "", "F", "EHR"],
    ["P03", "E2", "u3", "RxNorm", "310798", "20180101", "Oral Product", "Generic", "Unknown", "", "30", "F", "EHR"],
]
LABS = [
    ["P01", "E1", "LOINC", "2160-0", "20190101", "1.35", "", "mg/dL", "F", "EHR"],
    ["P01", "E1", "LOINC", "33914-3", "20190101", "41.2", "", "mL/min/{1.73_m2}", "F", "EHR"],
    ["P03", "E2", "LOINC", "10524-7", "20130129", "", "", "units", "F", "EHR"],
    ["P03", "E2", "LOINC", "3094-0", "20180101", "not-a-number", "", "mg/dL", "F", "EHR"],
]
PATIENT_ROWS = [[p, "F" if i % 2 else "M", "White", "Not Hispanic or Latino", "Unknown",
                 "" if p == "P06" else str(1950 + i), "Protected" if p == "P06" else "Present",
                 "202203" if p == "P02" else "", "EHR" if p == "P02" else "", "South", "EHR"]
                for i, p in enumerate(PATIENTS)]

EXPECTED_COHORT = {
    # patient: (index, end, blank_start, y_esrd, y_dialysis, y_any)
    "P01": ("2020-01-10", "2022-01-10", "2019-10-12", True, False, True),
    "P03": ("2019-02-01", "2021-02-01", "2018-11-03", False, True, True),
    "P04": ("2018-01-01", "2020-01-01", "2017-10-03", True, False, True),
    "P05": ("2016-02-29", "2018-02-28", "2015-12-01", False, True, True),
    "P10": ("2015-05-05", "2017-05-05", "2015-02-04", False, True, True),
}
EXPECTED_EXCLUDED = {"P02", "P08"}


def _write(path: Path, header, rows, quote_all: bool) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def synth_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    _write(d / "patient.csv", PATIENT_HDR, PATIENT_ROWS, True)
    _write(d / "diagnosis.csv", DIAG_HDR, DIAGNOSES, True)
    _write(d / "procedure.csv", PROC_HDR, PROCEDURES, True)
    _write(d / "medication_drug.csv", MED_HDR, MEDICATIONS, True)
    _write(d / "lab_result.csv", LAB_HDR, LABS, True)          # quoted, like the raw file
    _write(d / "lab_result_clean.csv", LAB_HDR, LABS, False)   # unquoted, like the clean file
    return d


# ------------------------------------------------------------ database
@pytest.fixture(scope="session")
def pg_server_uri(tmp_path_factory) -> str:
    pgserver = pytest.importorskip("pgserver")
    srv = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="delete")
    return srv.get_uri()


def _with_db(uri: str, db: str) -> str:
    parts = urlsplit(uri)
    return urlunsplit(parts._replace(path="/" + db))


@pytest.fixture
def dsn(pg_server_uri) -> str:
    name = "t_" + uuid.uuid4().hex[:12]
    with psycopg.connect(pg_server_uri, autocommit=True) as c:
        c.execute(f"CREATE DATABASE {name}")
    yield _with_db(pg_server_uri, name)
    with psycopg.connect(pg_server_uri, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


@pytest.fixture
def migrated_dsn(dsn) -> str:
    from aki.db.migrate import migrate

    migrate(dsn)
    return dsn
