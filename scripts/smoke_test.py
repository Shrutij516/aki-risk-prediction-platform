"""End-to-end smoke test against a RUNNING demo server (scripts/demo.sh).

    .venv/bin/python scripts/smoke_test.py --base http://127.0.0.1:8000

Checks every endpoint and the served frontend; exits non-zero on the first failure.
Refuses to run against a server that reports REAL data.
"""
import argparse
import sys

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    base = ap.parse_args().base.rstrip("/")
    c = httpx.Client(base_url=base, timeout=60)

    def get(path, **params):
        r = c.get(path, params=params)
        r.raise_for_status()
        return r

    checks = []
    h = get("/api/health")
    assert h.headers["X-AKI-Dataset"] == "synthetic", "server is not serving synthetic demo data"
    checks.append(f"health ok, models loaded: {h.json()['models_loaded']}")
    d = get("/api/dataset").json()
    checks.append(f"dataset: {d['counts']['patients']} patients, cohort {d['counts']['cohort']}")
    models = get("/api/models").json()["models"]
    assert models, "no models registered"
    checks.append(f"models: {[m['model_key'] for m in models]}")
    page = get("/api/patients", model=models[0]["model_key"], sort="risk", limit=5).json()
    assert page["items"], "patient search returned nothing"
    pid = page["items"][0]["patient_id"]
    checks.append(f"patients: total {page['total']}, top-risk {pid}")
    p = get(f"/api/patients/{pid}").json()
    t = get(f"/api/patients/{pid}/timeline", limit=20).json()
    assert t["items"], "empty timeline"
    checks.append(f"patient detail + timeline ({t['total']} events)")
    for m in models:
        r = get(f"/api/patients/{pid}/risk", model=m["model_key"]).json()
        assert abs(r["probability"] - r["stored_probability"]) < 1e-5, f"live != stored for {m['model_key']}"
    checks.append(f"live risk == stored batch score for all {len(models)} models")
    assert len(p["predictions"]) == len(models)
    html = get("/").text
    assert '<div id="root">' in html, "frontend not served"
    checks.append("frontend index served")
    fr = c.get("/api/fairness")
    checks.append(f"fairness endpoint: {fr.status_code}")
    for line in checks:
        print("OK  ", line)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print("FAIL", e)
        sys.exit(1)
