"""CLI: python -m aki.ingest --select aki-sample --n 200 --build-cohort

Applies pending migrations first, then ingests. Prints aggregate counts only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aki.db.connection import get_dsn
from aki.db.migrate import migrate
from aki.ingest.trinetx import EVENT_SPECS, IngestConfig, ingest

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--select", choices=["all", "sample", "aki-sample"], default="sample")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows", choices=["all", "cohort-codes"], default="all")
    ap.add_argument("--tables", nargs="+", choices=list(EVENT_SPECS), default=list(EVENT_SPECS))
    ap.add_argument("--lab-file", help="override lab CSV (default lab_result_clean.csv)")
    ap.add_argument("--replace", action="store_true", help="truncate ehr.* and cohort.* before loading")
    ap.add_argument("--build-cohort", action="store_true")
    ap.add_argument("--allow-large", action="store_true")
    ap.add_argument("--summary-out", type=Path, help="write the JSON summary here (keep under reports/)")
    a = ap.parse_args()

    dsn = get_dsn(a.dsn)
    migrate(dsn)
    cfg = IngestConfig(data_dir=a.data_dir, select=a.select, n=a.n, seed=a.seed, rows=a.rows,
                       tables=tuple(a.tables), replace=a.replace, build_cohort=a.build_cohort,
                       files={"lab_result": a.lab_file} if a.lab_file else None, allow_large=a.allow_large)
    summary = ingest(dsn, cfg)
    text = json.dumps(summary, indent=2, default=str)
    if a.summary_out:
        a.summary_out.parent.mkdir(parents=True, exist_ok=True)
        a.summary_out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
