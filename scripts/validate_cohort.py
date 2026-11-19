"""Compare the SQL cohort in PostgreSQL with counts printed in RETAIN_WithLabs.ipynb (expected values read from a local file).

Requires a prior run of: python -m aki.ingest --dsn <db> --select all --rows cohort-codes --build-cohort
Usage: python scripts/validate_cohort.py [dsn]   (default: local database aki_full_cohort)
Writes reports/cohort_validation.json (aggregate counts only).
"""
import json
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aki.db.connection import dsn_for_database, get_dsn  # noqa: E402

# Expected values are the counts printed in the saved notebook outputs. They are specific to the
# TriNetX extract, so they live in a local, gitignored file rather than in this public script:
# local/reference_counts.json -> {"notebook_cohort_counts": {metric: [notebook cell, value], ...}}
REFERENCE = Path(__file__).resolve().parents[1] / "local" / "reference_counts.json"
if not REFERENCE.exists():
    sys.exit(f"missing {REFERENCE}: create it locally with the counts from the saved notebook outputs")
NOTEBOOK = {k: tuple(v) for k, v in json.loads(REFERENCE.read_text())["notebook_cohort_counts"].items()}
SQL = {
    "aki_diagnosis_rows": "SELECT count(*) FROM ehr.diagnosis WHERE (code_system='ICD-10-CM' AND code LIKE 'N17%') OR (code_system='ICD-9-CM' AND code LIKE '584%')",
    "esrd_diagnosis_rows": "SELECT count(*) FROM ehr.diagnosis WHERE (code_system='ICD-10-CM' AND code='N18.6') OR (code_system='ICD-9-CM' AND code='585.6')",
    "aki_patients": "SELECT (SELECT count(*) FROM cohort.aki_cohort) + (SELECT count(*) FROM cohort.excluded_preexisting_esrd)",
    "excluded_preexisting_esrd": "SELECT count(*) FROM cohort.excluded_preexisting_esrd",
    "cohort_patients": "SELECT count(*) FROM cohort.aki_cohort",
    "y_esrd_2yr": "SELECT count(*) FROM cohort.aki_cohort WHERE y_esrd_2yr",
    "y_dialysis_2yr": "SELECT count(*) FROM cohort.aki_cohort WHERE y_dialysis_2yr",
    "y_esrd_or_dialysis_2yr": "SELECT count(*) FROM cohort.aki_cohort WHERE y_esrd_or_dialysis_2yr",
}
# Database holding the full cohort (ingested with --select all --rows cohort-codes); override with argv[1].
DSN = sys.argv[1] if len(sys.argv) > 1 else dsn_for_database(get_dsn(), "aki_full_cohort")
with psycopg.connect(DSN) as c:
    rows = []
    for k, (cell, nb) in NOTEBOOK.items():
        got = c.execute(SQL[k]).fetchone()[0]
        rows.append({"metric": k, "notebook_cell": cell, "notebook": nb, "sql": got, "match": got == nb})
out = {"source_notebook": "AKI_Dialysis_Prediction-main/RETAIN_WithLabs.ipynb", "checks": rows,
       "all_match": all(r["match"] for r in rows)}
Path("reports/cohort_validation.json").write_text(json.dumps(out, indent=2))
for r in rows:
    print(f"{r['metric']:28s} {r['notebook_cell']:8s} notebook={r['notebook']:>7,} sql={r['sql']:>7,} {'OK' if r['match'] else 'MISMATCH'}")
print("ALL MATCH" if out["all_match"] else "MISMATCHES FOUND")
