"""API tests against a small synthetic database built with the real pipeline code (no real data)."""
import json

import psycopg
import pytest
import torch
from fastapi.testclient import TestClient

from aki.api.main import create_app
from aki.demo.synthetic_ehr import generate
from aki.ingest.trinetx import IngestConfig, ingest
from aki.model.data import Vocab
from aki.model.models import build_model
from aki.model.sequences import build_sequences, cohort_from_postgres
from aki.serve.registry import load_all
from aki.serve.scoring import load_sequences, register_models, score_all, set_dataset_info


def _bundle(path, key, kind, variant, seqs):
    """Untrained (seeded) model bundle in the export format: tests the plumbing, not model quality."""
    torch.manual_seed(0)
    vocab = Vocab.build(list(seqs["tokens"]))
    m = build_model(kind, len(vocab), 64, 0.1)
    d = path / key
    d.mkdir(parents=True)
    torch.save(m.state_dict(), d / "weights.pt")
    (d / "vocab.json").write_text(json.dumps(vocab.token2id))
    (d / "meta.json").write_text(json.dumps({
        "key": key, "model": kind, "labs": variant == "withlabs", "source": "synthetic-trained", "mlflow_run_id": None,
        "maxlen": 64, "truncate": "keep_first", "mask_padding": False, "dropout": 0.1, "focal_alpha": 0.5,
        "threshold": 0.5, "vocab_size": len(vocab), "val_metrics": {"roc_auc": 0.5}, "test_metrics": {"roc_auc": 0.5}}))


@pytest.fixture
def demo(migrated_dsn, tmp_path):
    data = tmp_path / "demo_data"
    generate(data, n=60, seed=7)
    ingest(migrated_dsn, IngestConfig(data_dir=data, select="all", build_cohort=True))
    cohort = cohort_from_postgres(migrated_dsn)
    models_dir = tmp_path / "models"
    seqs = {}
    for variant in ("nolabs", "withlabs"):
        seqs[variant], _ = build_sequences(data, cohort, include_labs=variant == "withlabs")
    _bundle(models_dir, "retain_withlabs", "retain", "withlabs", seqs["withlabs"])
    _bundle(models_dir, "transformer_nolabs", "transformer", "nolabs", seqs["nolabs"])
    models = load_all(models_dir)
    with psycopg.connect(migrated_dsn) as conn:
        set_dataset_info(conn, "synthetic", "test")
        for v, df in seqs.items():
            load_sequences(conn, df, v)
        register_models(conn, models)
        score_all(conn, models)
    fairness = tmp_path / "fairness.json"
    fairness.write_text(json.dumps({"kind": "test-report"}))
    app = create_app(dsn=migrated_dsn, models_dir=models_dir, fairness_path=fairness, web_dist=tmp_path / "none")
    with TestClient(app) as client:
        yield client, migrated_dsn, seqs


def test_health_dataset_and_header(demo):
    client, _, _ = demo
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["dataset_kind"] == "synthetic"
    assert r.headers["X-AKI-Dataset"] == "synthetic"
    d = client.get("/api/dataset").json()
    assert d["counts"]["patients"] == 60 and d["counts"]["cohort"] > 0
    assert "no clinical meaning" in d["disclaimer"]


def test_models_include_original_published_results(demo):
    client, _, _ = demo
    ms = {m["model_key"]: m for m in client.get("/api/models").json()["models"]}
    assert set(ms) == {"retain_withlabs", "transformer_nolabs"}
    assert ms["retain_withlabs"]["original_published"]["roc_auc"] == 0.653
    assert ms["retain_withlabs"]["evaluation_label"] == "Synthetic data only"


def test_patient_search_paging_and_risk_sort(demo):
    client, _, seqs = demo
    page = client.get("/api/patients", params={"limit": 5}).json()
    assert page["total"] == len(seqs["withlabs"]) or page["total"] >= 5
    assert len(page["items"]) == 5 and page["items"][0]["patient_id"].startswith("SYN")
    one = client.get("/api/patients", params={"q": page["items"][0]["patient_id"]}).json()
    assert one["total"] == 1
    ranked = client.get("/api/patients", params={"model": "retain_withlabs", "sort": "risk", "limit": 20}).json()["items"]
    probs = [r["probability"] for r in ranked]
    assert probs == sorted(probs, reverse=True)
    assert client.get("/api/patients", params={"sort": "risk"}).status_code == 422


def test_patient_detail_and_timeline(demo):
    client, _, seqs = demo
    pid = seqs["withlabs"]["patient_id"].iloc[0]
    p = client.get(f"/api/patients/{pid}").json()
    assert p["in_cohort"] and {x["model_key"] for x in p["predictions"]} == {"retain_withlabs", "transformer_nolabs"}
    t = client.get(f"/api/patients/{pid}/timeline", params={"limit": 10}).json()
    assert t["total"] == sum(p["event_counts"].values()) and len(t["items"]) == 10
    dates = [i["date"] for i in t["items"]]
    assert dates == sorted(dates, reverse=True)
    obs = client.get(f"/api/patients/{pid}/timeline", params={"observation_only": True, "limit": 1000}).json()
    assert all(i["in_observation_window"] for i in obs["items"]) and obs["total"] < t["total"]
    labs = client.get(f"/api/patients/{pid}/timeline", params={"types": "lab"}).json()
    assert {i["type"] for i in labs["items"]} == {"lab"}
    assert client.get("/api/patients/NOPE/timeline").status_code == 404


def test_live_risk_matches_batch_prediction_and_has_attention(demo):
    client, _, seqs = demo
    pid = seqs["withlabs"]["patient_id"].iloc[0]
    r = client.get(f"/api/patients/{pid}/risk", params={"model": "retain_withlabs"}).json()
    assert abs(r["probability"] - r["stored_probability"]) < 1e-6
    assert r["above_threshold"] == (r["probability"] >= r["threshold"])
    assert 0 < len(r["attention"]) <= 10 and all("date" in a and "token" in a for a in r["attention"])
    t = client.get(f"/api/patients/{pid}/risk", params={"model": "transformer_nolabs"}).json()
    assert t["attention"] is None or t["model"] == "transformer"
    assert client.get(f"/api/patients/{pid}/risk", params={"model": "nope"}).status_code == 404


def test_fairness_endpoint(demo):
    client, _, _ = demo
    assert client.get("/api/fairness").json() == {"kind": "test-report"}


def test_refuses_real_dataset_unless_allowed(demo, tmp_path):
    _, dsn, _ = demo
    with psycopg.connect(dsn) as conn:
        set_dataset_info(conn, "real", "pretend real")
    with pytest.raises(RuntimeError, match="marked REAL"):
        with TestClient(create_app(dsn=dsn, models_dir=tmp_path, allow_real=False, web_dist=tmp_path)):
            pass
    with TestClient(create_app(dsn=dsn, models_dir=tmp_path, allow_real=True, web_dist=tmp_path)) as c:
        assert c.get("/api/health").headers["X-AKI-Dataset"] == "real"
