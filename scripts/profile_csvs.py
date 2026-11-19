"""Profile the recovered TriNetX CSVs without loading them into memory.

DuckDB streams each CSV in vectorized chunks under a fixed memory limit, so
multi-GB files are scanned, never materialised. Every column is read as text
(all_varchar) so parse failures are counted rather than crashing the scan.

Output contains aggregate statistics only (counts, null rates, value
frequencies of code systems / units). No patient_id or encounter_id values
are written. Output goes to reports/ which is gitignored.

Usage: .venv/bin/python scripts/profile_csvs.py [--files diagnosis.csv ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aki.lab_codes import ALL_KIDNEY_CODES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "reports"

DATE_COLS = {"date", "start_date"}
NUM_COLS = {"lab_result_num_val", "quantity_dispensed", "days_supply", "year_of_birth"}
# Low-cardinality columns worth listing value frequencies for.
FREQ_COLS = {
    "code_system", "principal_diagnosis_indicator", "admitting_diagnosis",
    "reason_for_visit", "derived_by_TriNetX", "source_id", "principal_procedure_indicator",
    "route", "sex", "race", "ethnicity", "marital_status", "reason_yob_missing",
    "patient_regional_location", "death_date_source_id",
}
# High-cardinality ID columns: approximate distinct counts only.
ID_COLS = {"patient_id", "encounter_id", "unique_id"}


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    return con


def src(path: Path) -> str:
    p = str(path).replace("'", "''")
    return f"read_csv('{p}', all_varchar=true, header=true, quote='\"', escape='\"')"


def profile_file(con, path: Path) -> dict:
    t0 = time.time()
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src(path)}").fetchall()]
    exprs = ["count(*) AS n_rows"]
    for c in cols:
        q = f'"{c}"'
        exprs.append(f"count(*) FILTER (WHERE {q} IS NULL OR {q} = '') AS \"{c}__empty\"")
        if c in ID_COLS or c == "code" or c == "brand" or c == "units_of_measure" or c == "strength":
            exprs.append(f'approx_count_distinct({q}) AS "{c}__approx_distinct"')
        if c in DATE_COLS:
            d = f"try_strptime({q}, '%Y%m%d')"
            exprs += [
                f'min({d}) AS "{c}__min"', f'max({d}) AS "{c}__max"',
                f"count(*) FILTER (WHERE {q} <> '' AND {d} IS NULL) AS \"{c}__unparseable\"",
            ]
        if c in NUM_COLS:
            v = f"try_cast({q} AS DOUBLE)"
            exprs += [
                f'min({v}) AS "{c}__min"', f'max({v}) AS "{c}__max"',
                f"count(*) FILTER (WHERE {q} <> '' AND {v} IS NULL) AS \"{c}__non_numeric\"",
            ]
    row = con.execute(f"SELECT {', '.join(exprs)} FROM {src(path)}").fetchone()
    names = ["n_rows"] + [e.rsplit(" AS ", 1)[1].strip('"') for e in exprs[1:]]
    stats = dict(zip(names, row))
    out = {"file": path.name, "bytes": path.stat().st_size, "columns": cols, "n_rows": stats.pop("n_rows"), "column_stats": {}}
    for k, v in stats.items():
        c, metric = k.rsplit("__", 1)
        out["column_stats"].setdefault(c, {})[metric] = str(v) if hasattr(v, "isoformat") else v
    for c in cols:
        if c in FREQ_COLS:
            out["column_stats"][c]["top_values"] = con.execute(
                f'SELECT "{c}", count(*) n FROM {src(path)} GROUP BY 1 ORDER BY n DESC LIMIT 25'
            ).fetchall()
    out["seconds"] = round(time.time() - t0, 1)
    return out


def relationships(con) -> dict:
    """Patient-key coverage of each event table against patient.csv."""
    pat = src(DATA / "patient.csv")
    res = {"patient_rows": con.execute(f"SELECT count(*), count(DISTINCT patient_id) FROM {pat}").fetchone()}
    con.execute(f"CREATE TEMP TABLE pats AS SELECT DISTINCT patient_id FROM {pat}")
    for f in ["diagnosis.csv", "procedure.csv", "medication_drug.csv", "lab_result_clean.csv"]:
        if not (DATA / f).exists():
            continue
        con.execute(f"CREATE OR REPLACE TEMP TABLE ids AS SELECT DISTINCT patient_id FROM {src(DATA / f)}")
        n, orphan = con.execute(
            "SELECT count(*), count(*) FILTER (WHERE patient_id NOT IN (SELECT patient_id FROM pats)) FROM ids"
        ).fetchone()
        res[f] = {"distinct_patients": n, "patients_missing_from_patient_csv": orphan}
    return res


def lab_raw_vs_clean(con) -> dict:
    """How lab_result_clean.csv differs from lab_result.csv (aggregates only)."""
    out = {}
    for f in ["lab_result.csv", "lab_result_clean.csv"]:
        out[f] = con.execute(
            f"""SELECT count(*), count(DISTINCT patient_id),
                       count(*) FILTER (WHERE lab_result_num_val <> ''),
                       count(*) FILTER (WHERE code_system = 'LOINC')
                FROM {src(DATA / f)}"""
        ).fetchone()
    # Kidney labs used by RETAIN_WithLabs (curated LOINC list, aki/lab_codes.py)
    lst = ",".join(f"'{c}'" for c in sorted(ALL_KIDNEY_CODES))
    out["kidney_loinc_units_clean"] = con.execute(
        f"""SELECT code, units_of_measure, count(*) n,
                   median(try_cast(lab_result_num_val AS DOUBLE)) med
            FROM {src(DATA / 'lab_result_clean.csv')} WHERE code IN ({lst})
            GROUP BY 1,2 ORDER BY 1, n DESC"""
    ).fetchall()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--skip-relationships", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    con = connect()
    files = args.files or sorted(p.name for p in DATA.glob("*.csv"))
    report = {"files": {}}
    for f in files:
        print(f"profiling {f} ...", flush=True)
        report["files"][f] = profile_file(con, DATA / f)
        print(f"  {report['files'][f]['n_rows']:,} rows in {report['files'][f]['seconds']}s", flush=True)
        (OUT / "data_profile.json").write_text(json.dumps(report, indent=2, default=str))
    if not args.skip_relationships:
        print("relationships ...", flush=True)
        report["relationships"] = relationships(con)
        report["lab_raw_vs_clean"] = lab_raw_vs_clean(con)
    (OUT / "data_profile.json").write_text(json.dumps(report, indent=2, default=str))
    print("wrote reports/data_profile.json")


if __name__ == "__main__":
    main()
