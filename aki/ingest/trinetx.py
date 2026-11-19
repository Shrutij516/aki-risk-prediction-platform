"""Chunked TriNetX CSV -> PostgreSQL ingestion.

DuckDB streams each CSV (never fully in memory), filters to the selected
patients, converts types, and hands Arrow record batches to PostgreSQL COPY.

Selection is always by whole patient, never by row, so every loaded patient
has their complete history (or complete cohort-defining rows) and the SQL
cohort logic gives the same answer for them as on the full extract.

Selections:
  all         every patient in patient.csv
  sample      N patients from patient.csv, deterministic order md5(patient_id || ':' || seed)
  aki-sample  N patients with an AKI diagnosis (notebook definition), same ordering

Row scopes:
  all           every row of each table for the selected patients
  cohort-codes  only rows needed to build the cohort/labels: AKI + ESRD diagnoses
                and dialysis procedures (superset filter; exact rules live in SQL).
                Medication and lab tables are skipped.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import psycopg
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

# ---------------------------------------------------------------- table specs
# (csv column, kind). Target column name = csv column lower-cased.
TEXT, DATE, SMALLINT, NUM = "text", "date", "smallint", "num"


@dataclass(frozen=True)
class Spec:
    table: str
    default_file: str
    columns: tuple[tuple[str, str], ...]

    @property
    def target_columns(self) -> list[str]:
        return [c.lower() for c, _ in self.columns]


PATIENT = Spec("ehr.patient", "patient.csv", (
    ("patient_id", TEXT), ("sex", TEXT), ("race", TEXT), ("ethnicity", TEXT), ("marital_status", TEXT),
    ("year_of_birth", SMALLINT), ("reason_yob_missing", TEXT), ("month_year_death", TEXT),
    ("death_date_source_id", TEXT), ("patient_regional_location", TEXT), ("source_id", TEXT),
))
DIAGNOSIS = Spec("ehr.diagnosis", "diagnosis.csv", (
    ("patient_id", TEXT), ("encounter_id", TEXT), ("code_system", TEXT), ("code", TEXT),
    ("principal_diagnosis_indicator", TEXT), ("admitting_diagnosis", TEXT), ("reason_for_visit", TEXT),
    ("date", DATE), ("derived_by_TriNetX", TEXT), ("source_id", TEXT),
))
PROCEDURE = Spec("ehr.procedure", "procedure.csv", (
    ("patient_id", TEXT), ("encounter_id", TEXT), ("code_system", TEXT), ("code", TEXT),
    ("principal_procedure_indicator", TEXT), ("date", DATE), ("derived_by_TriNetX", TEXT), ("source_id", TEXT),
))
MEDICATION = Spec("ehr.medication_drug", "medication_drug.csv", (
    ("patient_id", TEXT), ("encounter_id", TEXT), ("unique_id", TEXT), ("code_system", TEXT), ("code", TEXT),
    ("start_date", DATE), ("route", TEXT), ("brand", TEXT), ("strength", TEXT),
    ("quantity_dispensed", NUM), ("days_supply", NUM), ("derived_by_TriNetX", TEXT), ("source_id", TEXT),
))
# lab_result_clean.csv is content-identical to lab_result.csv (same order-independent
# fingerprint over all fields, reports/extra_checks.json) but unquoted, so it is the default.
LAB = Spec("ehr.lab_result", "lab_result_clean.csv", (
    ("patient_id", TEXT), ("encounter_id", TEXT), ("code_system", TEXT), ("code", TEXT), ("date", DATE),
    ("lab_result_num_val", NUM), ("lab_result_text_val", TEXT), ("units_of_measure", TEXT),
    ("derived_by_TriNetX", TEXT), ("source_id", TEXT),
))
EVENT_SPECS = {"diagnosis": DIAGNOSIS, "procedure": PROCEDURE, "medication_drug": MEDICATION, "lab_result": LAB}

# Superset row filters for --rows cohort-codes (exact matching is done in migrations/002_cohort.sql).
COHORT_ROW_FILTERS = {
    "diagnosis": "(code LIKE 'N17%' OR code LIKE '584%' OR code IN ('N18.6', '585.6'))",
    "procedure": "(trim(code) IN ('90935','90937','90945','90947','G0257','39.95') OR trim(code) LIKE '5A1D%')",
}
AKI_DX_FILTER = "((code_system = 'ICD-10-CM' AND code LIKE 'N17%') OR (code_system = 'ICD-9-CM' AND code LIKE '584%'))"


# ---------------------------------------------------------------- helpers
def _q(path: Path) -> str:
    return str(path).replace("'", "''")


def _read(path: Path) -> str:
    return f"read_csv('{_q(path)}', all_varchar=true, header=true)"


def _expr(col: str, kind: str) -> tuple[str, str | None]:
    """Return (value expression, bad-parse flag expression or None)."""
    raw = f'nullif("{col}", \'\')'
    out = col.lower()
    if kind == TEXT:
        return f'{raw} AS "{out}"', None
    if kind == DATE:
        v = f"try_strptime({raw}, '%Y%m%d')::DATE"
        return f'{v} AS "{out}"', f"({raw} IS NOT NULL AND {v} IS NULL)"
    cast = {"smallint": "SMALLINT", "num": "DOUBLE"}[kind]
    v = f"try_cast({raw} AS {cast})"
    return f'{v} AS "{out}"', f"({raw} IS NOT NULL AND {v} IS NULL)"


def _git_version(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------- main entry
@dataclass
class IngestConfig:
    data_dir: Path
    select: str = "sample"            # all | sample | aki-sample
    n: int = 100
    seed: int = 42
    rows: str = "all"                 # all | cohort-codes
    tables: tuple[str, ...] = ("diagnosis", "procedure", "medication_drug", "lab_result")
    replace: bool = False
    build_cohort: bool = False
    batch_rows: int = 200_000
    duckdb_memory: str = "4GB"
    files: dict[str, str] | None = None   # override file name per table key
    allow_large: bool = False


def _file_for(cfg: IngestConfig, key: str, spec: Spec) -> Path:
    name = (cfg.files or {}).get(key, spec.default_file)
    return cfg.data_dir / name


def _select_patients(con: duckdb.DuckDBPyConnection, cfg: IngestConfig) -> int:
    patients = _read(_file_for(cfg, "patient", PATIENT))
    order = f"md5(patient_id || ':' || '{int(cfg.seed)}')"
    if cfg.select == "all":
        q = f"SELECT DISTINCT patient_id FROM {patients}"
    elif cfg.select == "sample":
        q = f"SELECT patient_id FROM (SELECT DISTINCT patient_id FROM {patients}) ORDER BY {order} LIMIT {int(cfg.n)}"
    elif cfg.select == "aki-sample":
        dx = _read(_file_for(cfg, "diagnosis", DIAGNOSIS))
        q = (f"SELECT patient_id FROM (SELECT DISTINCT patient_id FROM {dx} WHERE {AKI_DX_FILTER} "
             f"AND patient_id IN (SELECT patient_id FROM {patients})) ORDER BY {order} LIMIT {int(cfg.n)}")
    else:
        raise ValueError(f"unknown select mode {cfg.select!r}")
    con.execute(f"CREATE OR REPLACE TEMP TABLE sel AS {q}")
    return con.sql("SELECT count(*) FROM sel").fetchall()[0][0]


def _check_disk(cfg: IngestConfig) -> None:
    if cfg.select != "all" or cfg.rows != "all" or cfg.allow_large:
        return
    need = sum(_file_for(cfg, k, s).stat().st_size for k, s in EVENT_SPECS.items() if k in cfg.tables)
    free = shutil.disk_usage(cfg.data_dir).free
    # PostgreSQL heap + indexes typically take ~1.5x the CSV size.
    if need * 1.5 > free:
        raise RuntimeError(
            f"Full load needs roughly {need * 1.5 / 1e9:.0f} GB but only {free / 1e9:.0f} GB is free. "
            "Use --select sample/aki-sample or --rows cohort-codes, or pass --allow-large."
        )


def _copy_stream(con, cur: psycopg.Cursor, spec: Spec, sql: str, flag_names: list[str],
                 batch_rows: int) -> tuple[int, int, int]:
    """Stream query result into spec.table via COPY. Returns (rows, bad_date, bad_numeric)."""
    rows = bad_date = bad_num = 0
    target = ", ".join(spec.target_columns + ["ingest_run_id"])
    reader = con.sql(sql).to_arrow_reader(batch_rows)
    with cur.copy(f"COPY {spec.table} ({target}) FROM STDIN WITH (FORMAT csv)") as cp:
        for batch in reader:
            tbl = pa.Table.from_batches([batch])
            for name in flag_names:
                n = pc.sum(tbl[name]).as_py() or 0
                if name.startswith("_baddate"):
                    bad_date += n
                else:
                    bad_num += n
            tbl = tbl.drop_columns(flag_names)
            buf = io.BytesIO()
            pacsv.write_csv(tbl, buf, pacsv.WriteOptions(include_header=False, quoting_style="needed"))
            cp.write(buf.getvalue())
            rows += tbl.num_rows
    return rows, bad_date, bad_num


def _load_table(con, cur, cfg: IngestConfig, key: str, spec: Spec, run_id: int) -> dict:
    path = _file_for(cfg, key, spec)
    t0 = time.time()
    exprs, flags, flag_names = [], [], []
    for col, kind in spec.columns:
        e, f = _expr(col, kind)
        exprs.append(e)
        if f:
            name = f"_baddate_{col.lower()}" if kind == DATE else f"_badnum_{col.lower()}"
            flags.append(f"{f} AS {name}")
            flag_names.append(name)
    where = ["patient_id IN (SELECT patient_id FROM sel)"]
    if cfg.rows == "cohort-codes" and key in COHORT_ROW_FILTERS:
        where.append(COHORT_ROW_FILTERS[key])
    sql = (f"SELECT {', '.join(exprs)}, {int(run_id)}::BIGINT AS ingest_run_id"
           f"{', ' + ', '.join(flags) if flags else ''} FROM {_read(path)} WHERE {' AND '.join(where)}")
    rows, bad_date, bad_num = _copy_stream(con, cur, spec, sql, flag_names, cfg.batch_rows)
    orphans = None
    if cfg.select == "all" and key != "patient":
        orphans = con.sql(
            f"SELECT count(*) FROM {_read(path)} WHERE patient_id NOT IN (SELECT patient_id FROM sel)"
        ).fetchall()[0][0]
    st = path.stat()
    rec = dict(source_file=path.name, target_table=spec.table, file_bytes=st.st_size,
               file_mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc), rows_loaded=rows,
               rows_bad_date=bad_date, rows_bad_numeric=bad_num, rows_orphan_patient=orphans,
               seconds=round(time.time() - t0, 2))
    cur.execute(
        """INSERT INTO meta.ingest_file (run_id, source_file, target_table, file_bytes, file_mtime, rows_loaded,
               rows_bad_date, rows_bad_numeric, rows_orphan_patient, seconds)
           VALUES (%(run_id)s, %(source_file)s, %(target_table)s, %(file_bytes)s, %(file_mtime)s, %(rows_loaded)s,
               %(rows_bad_date)s, %(rows_bad_numeric)s, %(rows_orphan_patient)s, %(seconds)s)""",
        dict(rec, run_id=run_id),
    )
    return rec


def ingest(dsn: str, cfg: IngestConfig) -> dict:
    """Run one ingestion. Returns a summary dict (aggregate counts only)."""
    _check_disk(cfg)
    unknown = set(cfg.tables) - set(EVENT_SPECS)
    if unknown:
        raise ValueError(f"unknown tables: {sorted(unknown)}")
    tables = [t for t in cfg.tables if not (cfg.rows == "cohort-codes" and t not in COHORT_ROW_FILTERS)]

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg.duckdb_memory}'")
    con.execute("SET preserve_insertion_order=false")

    selection = dict(select=cfg.select, n=cfg.n if cfg.select != "all" else None, seed=cfg.seed,
                     rows=cfg.rows, tables=tables, files=cfg.files or {})
    with psycopg.connect(dsn, autocommit=True) as conn:
        run_id = conn.execute(
            "INSERT INTO meta.ingest_run (data_dir, selection, code_version) VALUES (%s, %s, %s) RETURNING run_id",
            (str(cfg.data_dir), json.dumps(selection), _git_version(Path(__file__).resolve().parents[2])),
        ).fetchone()[0]
        summary = {"run_id": run_id, "selection": selection, "files": []}
        try:
            with conn.transaction(), conn.cursor() as cur:
                n_existing = cur.execute("SELECT count(*) FROM ehr.patient").fetchone()[0]
                if n_existing and not cfg.replace:
                    raise RuntimeError(f"ehr.patient already has {n_existing} rows; pass replace=True (--replace).")
                if cfg.replace:
                    cur.execute("TRUNCATE ehr.patient, ehr.diagnosis, ehr.procedure, ehr.medication_drug, "
                                "ehr.lab_result, cohort.aki_cohort, cohort.excluded_preexisting_esrd CASCADE")
                    # CASCADE also clears app.sequence / app.prediction (migration 003), which are
                    # derived from the patients being replaced.
                summary["patients_selected"] = _select_patients(con, cfg)
                summary["files"].append(_load_table(con, cur, cfg, "patient", PATIENT, run_id))
                for key in tables:
                    summary["files"].append(_load_table(con, cur, cfg, key, EVENT_SPECS[key], run_id))
                if cfg.build_cohort:
                    summary["cohort_rows"] = cur.execute("SELECT cohort.build_aki_cohort()").fetchone()[0]
            conn.execute("UPDATE meta.ingest_run SET status='succeeded', finished_at=now() WHERE run_id=%s", (run_id,))
        except Exception as exc:
            conn.execute("UPDATE meta.ingest_run SET status='failed', finished_at=now(), notes=%s WHERE run_id=%s",
                         (f"{type(exc).__name__}: {exc}"[:2000], run_id))
            raise
    for f in summary["files"]:
        f["file_mtime"] = f["file_mtime"].isoformat()
    return summary
