"""Local embedded PostgreSQL 16 for development (pgserver; Unix socket only, no TCP).

    python -m aki.db.local_server start   # init/start server, print socket URI
    python -m aki.db.local_server stop

Data directory: $AKI_PGDATA, default ~/.aki_reconstruction/pgdata. It lives
outside the project because pgserver cannot start from a path containing
spaces ("NEW PROJECTS"). It holds patient-level data once real CSVs are
ingested and must stay on this machine. Any PostgreSQL 16 reachable through
$AKI_DATABASE_URL works the same way (e.g. a Docker container later).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PGDATA = Path(os.environ.get("AKI_PGDATA", Path.home() / ".aki_reconstruction" / "pgdata"))
if " " in str(PGDATA):
    raise RuntimeError(f"AKI_PGDATA must not contain spaces (pgserver limitation): {PGDATA}")


def is_initialised() -> bool:
    return (PGDATA / "PG_VERSION").exists()


def start() -> str:
    """Start (initialising if needed) and return the socket URI. Server keeps running."""
    import pgserver

    PGDATA.parent.mkdir(parents=True, exist_ok=True)
    return pgserver.get_server(PGDATA, cleanup_mode=None).get_uri()


def stop() -> None:
    import pgserver

    if is_initialised():
        pgserver.pg_ctl(["-w", "stop", "-m", "fast"], pgdata=PGDATA)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    if cmd == "start":
        print(start())
    elif cmd == "stop":
        stop()
    else:
        sys.exit("usage: python -m aki.db.local_server [start|stop]")
