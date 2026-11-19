"""Minimal forward-only SQL migration runner.

Applies migrations/NNN_name.sql in order, each in its own transaction, and
records (version, name, sha256) in meta.schema_migrations. If an applied
migration file is later edited, the checksum mismatch is an error: add a new
migration instead of rewriting history.

The one exception is a comment-only edit. `reconcile` updates the stored checksum
after proving, for that database, that:
  1. the stored checksum equals the sha256 of the previous file text you supply, and
  2. the previous and current files have identical executable SQL (comments removed;
     string literals and $$-quoted bodies such as function code are kept intact).
Anything else is refused.

Usage: python -m aki.db.migrate [--dsn URL]   (default: $AKI_DATABASE_URL)
       python -m aki.db.migrate --dsn URL --reconcile 1 --previous path/to/old_001.sql
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import psycopg

from aki.db.connection import get_dsn

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_NAME = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")


class MigrationError(RuntimeError):
    pass


def discover(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, str, Path]]:
    found = []
    for p in sorted(directory.glob("*.sql")):
        m = _NAME.match(p.name)
        if not m:
            raise MigrationError(f"bad migration filename: {p.name}")
        found.append((int(m.group(1)), m.group(2), p))
    versions = [v for v, _, _ in found]
    if len(set(versions)) != len(versions):
        raise MigrationError("duplicate migration version numbers")
    return found


def executable_sql(sql: str) -> str:
    """SQL with comments removed and whitespace collapsed, for comparing executable content.

    Single-quoted strings, double-quoted identifiers and dollar-quoted bodies ($$...$$, $tag$...$tag$)
    are copied verbatim, including any '--' inside them (e.g. comments in plpgsql function bodies,
    which PostgreSQL stores as part of the function).
    """
    out, i, n = [], 0, len(sql)
    while i < n:
        c = sql[i]
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j
            out.append(" ")
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            if j < 0:
                raise MigrationError("unterminated block comment")
            i = j + 2
            out.append(" ")
        elif c in ("'", '"'):
            j = i + 1
            while True:
                j = sql.find(c, j)
                if j < 0:
                    raise MigrationError("unterminated quoted string")
                if sql.startswith(c * 2, j):   # escaped quote
                    j += 2
                    continue
                break
            out.append(sql[i:j + 1])
            i = j + 1
        elif c == "$":
            m = re.match(r"\$[A-Za-z_]*\$", sql[i:])
            if m:
                tag = m.group(0)
                j = sql.find(tag, i + len(tag))
                if j < 0:
                    raise MigrationError(f"unterminated dollar-quoted body {tag}")
                out.append(sql[i:j + len(tag)])
                i = j + len(tag)
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    text = "".join(out)
    # collapse whitespace outside the preserved literals is good enough: literals have no runs we alter
    return re.sub(r"\s+", " ", text).strip()


def _ensure_table(conn: psycopg.Connection) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS meta")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS meta.schema_migrations (
               version    INTEGER PRIMARY KEY,
               name       TEXT NOT NULL,
               sha256     TEXT NOT NULL,
               applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
    )


def migrate(dsn: str, directory: Path = MIGRATIONS_DIR) -> list[int]:
    """Apply pending migrations; return the versions applied in this call."""
    applied_now = []
    with psycopg.connect(dsn, autocommit=True) as conn:
        _ensure_table(conn)
        done = {v: (n, h) for v, n, h in conn.execute("SELECT version, name, sha256 FROM meta.schema_migrations")}
        for version, name, path in discover(directory):
            sql = path.read_text()
            digest = hashlib.sha256(sql.encode()).hexdigest()
            if version in done:
                if done[version][1] != digest:
                    raise MigrationError(f"migration {path.name} changed after being applied")
                continue
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO meta.schema_migrations (version, name, sha256) VALUES (%s, %s, %s)",
                    (version, name, digest),
                )
            applied_now.append(version)
    return applied_now


def reconcile(dsn: str, version: int, previous_sql: str, directory: Path = MIGRATIONS_DIR) -> str:
    """Accept a comment-only edit of an applied migration. Returns 'updated' or 'already current'."""
    path = next((p for v, _, p in discover(directory) if v == version), None)
    if path is None:
        raise MigrationError(f"no migration with version {version}")
    current_sql = path.read_text()
    old_digest = hashlib.sha256(previous_sql.encode()).hexdigest()
    new_digest = hashlib.sha256(current_sql.encode()).hexdigest()
    if executable_sql(previous_sql) != executable_sql(current_sql):
        raise MigrationError(f"{path.name}: executable SQL differs from the previous file; not a comment-only edit")
    with psycopg.connect(dsn) as conn:
        row = conn.execute("SELECT sha256 FROM meta.schema_migrations WHERE version = %s", (version,)).fetchone()
        if row is None:
            raise MigrationError(f"migration {version} has not been applied to this database")
        if row[0] == new_digest:
            return "already current"
        if row[0] != old_digest:
            raise MigrationError(f"stored checksum for {version} does not match the previous file you supplied")
        conn.execute("UPDATE meta.schema_migrations SET sha256 = %s WHERE version = %s AND sha256 = %s",
                     (new_digest, version, old_digest))
    return "updated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn")
    ap.add_argument("--reconcile", type=int, metavar="VERSION",
                    help="accept a comment-only edit of this applied migration (needs --previous)")
    ap.add_argument("--previous", type=Path, help="the migration file as it was when applied")
    args = ap.parse_args()
    if args.reconcile is not None:
        if not args.previous:
            ap.error("--reconcile needs --previous")
        print(reconcile(get_dsn(args.dsn), args.reconcile, args.previous.read_text()))
        return
    applied = migrate(get_dsn(args.dsn))
    print(f"applied: {applied}" if applied else "schema up to date")


if __name__ == "__main__":
    main()
