import shutil

import psycopg
import pytest

from aki.db.migrate import MIGRATIONS_DIR, MigrationError, migrate

EXPECTED_TABLES = {
    ("meta", "schema_migrations"), ("meta", "ingest_run"), ("meta", "ingest_file"),
    ("ehr", "patient"), ("ehr", "diagnosis"), ("ehr", "procedure"), ("ehr", "medication_drug"),
    ("ehr", "lab_result"), ("cohort", "aki_cohort"), ("cohort", "excluded_preexisting_esrd"),
    ("app", "dataset_info"), ("app", "sequence"), ("app", "model"), ("app", "prediction"),
}


def test_migrate_creates_schema_and_is_idempotent(dsn):
    first = migrate(dsn)
    assert first == [1, 2, 3]
    assert migrate(dsn) == []  # second run is a no-op
    with psycopg.connect(dsn) as c:
        tables = set(c.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_schema IN ('meta','ehr','cohort','app')").fetchall())
    assert EXPECTED_TABLES <= tables


def test_edited_migration_is_rejected(dsn, tmp_path):
    mdir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, mdir)
    migrate(dsn, mdir)
    p = mdir / "002_cohort.sql"
    p.write_text(p.read_text() + "\n-- edited\n")
    with pytest.raises(MigrationError, match="changed after being applied"):
        migrate(dsn, mdir)


def test_codes_are_text_and_dates_are_dates(migrated_dsn):
    with psycopg.connect(migrated_dsn) as c:
        types = dict(c.execute(
            "SELECT table_name || '.' || column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'ehr'").fetchall())
    assert types["diagnosis.code"] == "text"
    assert types["medication_drug.code"] == "text"
    assert types["diagnosis.date"] == "date"
    assert types["medication_drug.start_date"] == "date"
    assert types["lab_result.lab_result_num_val"] == "double precision"


def test_reconcile_accepts_comment_only_edits_and_nothing_else(dsn, tmp_path):
    from aki.db.migrate import executable_sql, reconcile

    mdir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, mdir)
    p = mdir / "001_raw_ehr.sql"
    original = p.read_text()
    migrate(dsn, mdir)

    # comment-only edit: migrate refuses until reconciled, then succeeds
    p.write_text(original.replace("-- 001_raw_ehr:", "-- 001_raw_ehr (comment edited):"))
    assert executable_sql(p.read_text()) == executable_sql(original)
    with pytest.raises(MigrationError):
        migrate(dsn, mdir)
    with pytest.raises(MigrationError, match="does not match the previous file"):
        reconcile(dsn, 1, original + "\n-- not what was applied\n", mdir)
    assert reconcile(dsn, 1, original, mdir) == "updated"
    assert reconcile(dsn, 1, original, mdir) == "already current"
    assert migrate(dsn, mdir) == []

    # executable change is refused even with the right previous file
    edited = p.read_text()
    p.write_text(edited.replace("CREATE INDEX lab_code_idx", "CREATE INDEX lab_code_idx2"))
    with pytest.raises(MigrationError, match="executable SQL differs"):
        reconcile(dsn, 1, edited, mdir)


def test_executable_sql_keeps_strings_and_dollar_bodies():
    from aki.db.migrate import executable_sql

    sql = "SELECT '--not a comment' AS a; -- real comment\nCREATE FUNCTION f() RETURNS int AS $$ -- body\nSELECT 1 $$ LANGUAGE sql; /* block */"
    out = executable_sql(sql)
    assert "'--not a comment'" in out and "-- body" in out and "real comment" not in out and "block" not in out
