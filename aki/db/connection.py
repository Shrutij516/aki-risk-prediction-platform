"""Database connection settings.

The DSN comes from (in order): an explicit argument, $AKI_DATABASE_URL, or the
local embedded PostgreSQL (see aki.db.local_server), which is
started on demand. Credentials never live in code.
"""
from __future__ import annotations

import os


def get_dsn(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("AKI_DATABASE_URL"):
        return os.environ["AKI_DATABASE_URL"]
    from aki.db import local_server

    if local_server.is_initialised():
        return local_server.start()
    raise RuntimeError(
        "No database configured: set AKI_DATABASE_URL or run `python -m aki.db.local_server start`."
    )


def dsn_for_database(dsn: str, dbname: str) -> str:
    """Same server/credentials, different database name."""
    from urllib.parse import urlsplit, urlunsplit

    if not dbname.replace("_", "").isalnum():
        raise ValueError(f"unsafe database name {dbname!r}")
    return urlunsplit(urlsplit(dsn)._replace(path="/" + dbname))


def ensure_database(dsn: str, dbname: str, reset: bool = False) -> str:
    """Create (or with reset=True, drop and recreate) a database on the server behind dsn."""
    import psycopg

    admin = dsn_for_database(dsn, "postgres")
    target = dsn_for_database(dsn, dbname)
    with psycopg.connect(admin, autocommit=True) as c:
        exists = c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if exists and reset:
            c.execute(f'DROP DATABASE "{dbname}" WITH (FORCE)')
            exists = None
        if not exists:
            c.execute(f'CREATE DATABASE "{dbname}"')
    return target
