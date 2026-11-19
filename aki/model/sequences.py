"""Build patient event-token sequences from the TriNetX CSVs, as the notebooks did.

Faithful to RETAIN_WithLabs.ipynb / RETAIN_NoLabs.ipynb (cells cited inline):
  * events = diagnosis (date), procedure (date), medication (start_date) rows
    of cohort patients dated strictly before blank_start_date (cells 64-69)
  * token = "<event_type>_<code>"; code_system is NOT part of the token (cell 81)
  * patients with >= 3 non-lab events are kept (cells 72-76)
  * with labs: curated kidney LOINC rows -> numeric, unit-validated, converted,
    plausibility-filtered, severity-binned (cell 80, aki/lab_codes.py), restricted
    to cohort patients and < blank_start_date, then appended to the filtered
    timeline. As in the notebook, lab events are appended for ALL cohort patients,
    so patients with < 3 non-lab events re-enter if they have labs.
  * order within a patient: date, then the original concat order
    (diagnosis, procedure, medication, lab), then source-file row order. This
    reproduces pandas' stable multi-column sort in cells 71/77/80, so truncation
    to the first 400 events selects the same events as the original.

DuckDB streams the CSVs; nothing is fully loaded into memory. Output is
patient-level (patient_id, label, tokens) and must stay in gitignored outputs/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import pandas as pd

from aki.lab_codes import (BUN_CODES, CREATININE_CODES, EGFR_CODES, PLAUSIBLE, UMOL_PER_MGDL_CREATININE,
                           VALID_UNITS)

ROOT = Path(__file__).resolve().parents[2]


def _lst(codes) -> str:
    return ", ".join(f"'{c}'" for c in sorted(codes))


def _read(path: Path) -> str:
    p = str(path).replace("'", "''")
    return f"read_csv('{p}', all_varchar=true, header=true)"


# SQL mirror of aki.lab_codes.severity_bin (equality checked in tests/test_sequences.py).
LAB_TEST_SQL = f"""CASE WHEN code IN ({_lst(CREATININE_CODES)}) THEN 'creatinine'
                        WHEN code IN ({_lst(EGFR_CODES)}) THEN 'egfr'
                        WHEN code IN ({_lst(BUN_CODES)}) THEN 'bun' END"""
LAB_BIN_SQL = """CASE test
  WHEN 'creatinine' THEN 'creatinine_' || CASE WHEN v < 1.2 THEN 'normal' WHEN v < 2.0 THEN 'mild'
                                               WHEN v < 4.0 THEN 'moderate' ELSE 'severe' END
  WHEN 'egfr' THEN 'egfr_' || CASE WHEN v >= 90 THEN 'g1' WHEN v >= 60 THEN 'g2' WHEN v >= 45 THEN 'g3a'
                                   WHEN v >= 30 THEN 'g3b' WHEN v >= 15 THEN 'g4' ELSE 'g5' END
  WHEN 'bun' THEN 'bun_' || CASE WHEN v < 20 THEN 'normal' WHEN v < 50 THEN 'elevated' ELSE 'high' END END"""


def _valid_units_sql() -> str:
    parts = [f"(test = '{t}' AND units IN ({_lst(u)}))" for t, u in VALID_UNITS.items()]
    return " OR ".join(parts)


def _plausible_sql() -> str:
    return " OR ".join(f"(test = '{t}' AND v BETWEEN {lo} AND {hi})" for t, (lo, hi) in PLAUSIBLE.items())


def build_sequences(data_dir: Path, cohort: pd.DataFrame, include_labs: bool,
                    lab_file: str = "lab_result_clean.csv", memory: str = "6GB") -> tuple[pd.DataFrame, dict]:
    """cohort: columns patient_id, blank_start_date (date), label (0/1).

    Returns (sequences DataFrame [patient_id, label, tokens, n_events], aggregate stats dict).
    """
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    con.execute("SET preserve_insertion_order=true")  # row order == file order (needed for tie-breaking)
    coh = cohort[["patient_id", "blank_start_date", "label"]].copy()
    coh["blank_start_date"] = pd.to_datetime(coh["blank_start_date"]).dt.date
    con.register("cohort_df", coh)
    con.execute("CREATE TEMP TABLE cohort AS SELECT patient_id, CAST(blank_start_date AS DATE) AS blank_start_date, "
                "CAST(label AS INTEGER) AS label FROM cohort_df")
    stats: dict = {"cohort_patients": len(coh), "include_labs": include_labs}

    sources = [("diagnosis", 0, "diagnosis.csv", "date"), ("procedure", 1, "procedure.csv", "date"),
               ("medication", 2, "medication_drug.csv", "start_date")]
    for etype, rank, fname, dcol in sources:
        con.execute(f"""
            CREATE TEMP TABLE ev_{etype} AS
            WITH src AS (
                SELECT patient_id, code, try_strptime({dcol}, '%Y%m%d')::DATE AS date,
                       row_number() OVER () AS rn
                FROM {_read(data_dir / fname)}
                WHERE patient_id IN (SELECT patient_id FROM cohort))
            SELECT s.patient_id, s.date, {rank} AS type_rank, s.rn, '{etype}_' || s.code AS token
            FROM src s JOIN cohort c USING (patient_id)
            WHERE s.date IS NOT NULL AND s.date < c.blank_start_date""")
        stats[f"{etype}_events_obs_window"] = con.sql(f"SELECT count(*) FROM ev_{etype}").fetchall()[0][0]

    con.execute("CREATE TEMP TABLE ev AS SELECT * FROM ev_diagnosis UNION ALL SELECT * FROM ev_procedure "
                "UNION ALL SELECT * FROM ev_medication")
    stats["events_total_no_labs"] = con.sql("SELECT count(*) FROM ev").fetchall()[0][0]
    stats["patients_with_any_event"] = con.sql("SELECT count(DISTINCT patient_id) FROM ev").fetchall()[0][0]
    con.execute("CREATE TEMP TABLE valid AS SELECT patient_id FROM ev GROUP BY 1 HAVING count(*) >= 3")
    stats["patients_ge3_events"] = con.sql("SELECT count(*) FROM valid").fetchall()[0][0]
    con.execute("CREATE TEMP TABLE timeline AS SELECT * FROM ev WHERE patient_id IN (SELECT patient_id FROM valid)")
    stats["events_after_ge3_filter"] = con.sql("SELECT count(*) FROM timeline").fetchall()[0][0]

    if include_labs:
        con.execute(f"""
            CREATE TEMP TABLE lab_k AS
            SELECT patient_id, code, try_strptime(date, '%Y%m%d')::DATE AS date,
                   try_cast(nullif(lab_result_num_val, '') AS DOUBLE) AS v, trim(units_of_measure) AS units,
                   {LAB_TEST_SQL} AS test, row_number() OVER () AS rn
            FROM (SELECT patient_id, trim(code) AS code, date, lab_result_num_val, units_of_measure
                  FROM {_read(data_dir / lab_file)} WHERE code_system = 'LOINC')
            WHERE code IN ({_lst(CREATININE_CODES | EGFR_CODES | BUN_CODES)})""")
        stats["lab_kidney_rows_loaded"] = con.sql("SELECT count(*) FROM lab_k").fetchall()[0][0]
        con.execute("DELETE FROM lab_k WHERE v IS NULL")
        stats["lab_after_drop_nan"] = con.sql("SELECT count(*) FROM lab_k").fetchall()[0][0]
        con.execute(f"DELETE FROM lab_k WHERE NOT ({_valid_units_sql()}) OR units IS NULL")
        stats["lab_after_units"] = con.sql("SELECT count(*) FROM lab_k").fetchall()[0][0]
        con.execute(f"UPDATE lab_k SET v = v / {UMOL_PER_MGDL_CREATININE} WHERE test = 'creatinine' AND units = 'umol/L'")
        con.execute(f"DELETE FROM lab_k WHERE NOT ({_plausible_sql()})")
        stats["lab_after_plausible"] = con.sql("SELECT count(*) FROM lab_k").fetchall()[0][0]
        con.execute(f"""
            CREATE TEMP TABLE ev_lab AS
            SELECT l.patient_id, l.date, 3 AS type_rank, l.rn, 'lab_' || ({LAB_BIN_SQL}) AS token
            FROM lab_k l JOIN cohort c USING (patient_id)
            WHERE l.date IS NOT NULL AND l.date < c.blank_start_date""")
        stats["lab_events_after_blanking"] = con.sql("SELECT count(*) FROM ev_lab").fetchall()[0][0]
        stats["lab_unique_tokens"] = con.sql("SELECT count(DISTINCT token) FROM ev_lab").fetchall()[0][0]
        con.execute("INSERT INTO timeline SELECT * FROM ev_lab")
        stats["events_total_with_labs"] = con.sql("SELECT count(*) FROM timeline").fetchall()[0][0]

    seqs = con.sql("""
        SELECT t.patient_id, c.label, list(t.token ORDER BY t.date, t.type_rank, t.rn) AS tokens,
               list(t.date ORDER BY t.date, t.type_rank, t.rn) AS dates, count(*) AS n_events
        FROM timeline t JOIN cohort c USING (patient_id)
        GROUP BY t.patient_id, c.label ORDER BY t.patient_id""").df()
    stats["sequence_patients"] = len(seqs)
    stats["sequence_positives"] = int(seqs["label"].sum())
    stats["vocab_all_patients"] = con.sql("SELECT count(DISTINCT token) FROM timeline").fetchall()[0][0]
    stats["median_sequence_length"] = float(seqs["n_events"].median())
    stats["seconds"] = round(time.time() - t0, 1)
    return seqs, stats


def cohort_from_postgres(dsn: str) -> pd.DataFrame:
    import psycopg

    with psycopg.connect(dsn) as c:
        rows = c.execute("SELECT patient_id, blank_start_date, y_esrd_or_dialysis_2yr::int FROM cohort.aki_cohort").fetchall()
    return pd.DataFrame(rows, columns=["patient_id", "blank_start_date", "label"])


def main() -> None:
    from aki.db.connection import get_dsn

    ap = argparse.ArgumentParser(description="Build event sequences (patient-level output: keep in outputs/).")
    ap.add_argument("--dsn", help="database holding cohort.aki_cohort (default: AKI_DATABASE_URL / local server)")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--labs", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stats-out", type=Path)
    a = ap.parse_args()
    if not str(a.out.resolve()).startswith(str((ROOT / "outputs").resolve())):
        raise SystemExit("--out must be under outputs/ (gitignored): sequences are patient-level data")
    cohort = cohort_from_postgres(get_dsn(a.dsn))
    seqs, stats = build_sequences(a.data_dir, cohort, a.labs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    seqs.to_parquet(a.out, index=False)
    text = json.dumps(stats, indent=2)
    if a.stats_out:
        a.stats_out.parent.mkdir(parents=True, exist_ok=True)
        a.stats_out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
