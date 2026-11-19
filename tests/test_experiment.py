"""End-to-end experiment on synthetic data: MLflow logging and validation-only selection."""
import mlflow
import numpy as np

from aki.model.data import split_indices, synthetic_sequences
from aki.model.experiment import run_experiment
from aki.model.train import TrainConfig

BASE = TrainConfig(model="retain", epochs=2, patience=2, maxlen=48, device="cpu", seed=3)


def _run(tmp_path, seqs, pids, y, name):
    return run_experiment(pids, y, seqs, base=BASE, alphas=(0.25, 0.5), dropouts=(0.1,),
                          dataset_name=name, tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
                          experiment_name="test", artifact_location=(tmp_path / "art").as_uri(), n_boot=50)


def test_experiment_logs_to_mlflow_without_patient_ids(tmp_path):
    pids, y, seqs = synthetic_sequences(400, prevalence=0.2, seed=1, mean_len=30)
    s = _run(tmp_path, seqs, pids, y, "syn")
    assert set(s["test"]) >= {"roc_auc", "pr_auc", "f1", "ci95", "at_val_fpr"}
    assert len(s["configs"]) == 2
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    runs = mlflow.search_runs(experiment_names=["test"], output_format="list")
    assert len(runs) == 3  # parent + 2 nested
    parent = next(r for r in runs if r.info.run_id == s["mlflow_run_id"])
    assert "test_roc_auc" in parent.data.metrics and parent.data.tags["reconstruction"] == "true"
    logged = " ".join(str(v) for r in runs for v in (*r.data.params.values(), *r.data.tags.values()))
    assert not any(pid in logged for pid in pids)


def test_test_set_does_not_influence_selection(tmp_path):
    pids, y, seqs = synthetic_sequences(400, prevalence=0.2, seed=2, mean_len=30)
    a = _run(tmp_path, seqs, pids, y, "a")
    # Scramble every TEST patient's sequence: selection must not change.
    _, _, te = split_indices(np.asarray(y), seed=BASE.seed)
    rng = np.random.default_rng(0)
    scrambled = [list(s) for s in seqs]
    for i in te:
        scrambled[i] = [f"diagnosis_NOISE{rng.integers(1000)}" for _ in range(10)]
    b = _run(tmp_path, scrambled, pids, y, "b")
    assert a["selected"] == b["selected"]
    assert [c["val"] for c in a["configs"]] == [c["val"] for c in b["configs"]]
    assert a["test"]["roc_auc"] != b["test"]["roc_auc"]  # the test set was actually used afterwards
