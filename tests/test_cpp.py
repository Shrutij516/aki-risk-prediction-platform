"""C++ (LibTorch AOTInductor) runner vs eager PyTorch, on a tiny untrained model and synthetic ids.

Skipped unless the executable has been built (scripts/build_cpp.sh). No real data is used.
"""
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from aki.model.models import RETAIN, TransformerClassifier
from aki.serve.export_cpp import MAX_BATCH, export_package

BIN = Path(__file__).resolve().parents[1] / "cpp" / "build" / "aki_infer"
pytestmark = pytest.mark.skipif(not BIN.exists(), reason="C++ runner not built (scripts/build_cpp.sh)")


def _run(pkg, X, tmp_path, batch=7):
    inp = tmp_path / "in.txt"
    inp.write_text("\n".join(" ".join(map(str, r)) for r in X.tolist()) + "\n")
    return subprocess.run([str(BIN), str(pkg), str(inp), str(batch)], capture_output=True, text=True)


@pytest.mark.parametrize("cls", [TransformerClassifier, RETAIN])
def test_cpp_matches_eager(cls, tmp_path):
    torch.manual_seed(0)
    maxlen = 16
    model = cls(60, maxlen=maxlen) if cls is TransformerClassifier else cls(60)
    pkg = export_package(model, maxlen, tmp_path / "m.pt2", cls.__name__)
    rng = np.random.default_rng(0)
    X = rng.integers(0, 60, size=(23, maxlen))
    X[0] = 0                      # all padding
    X[1, 3:] = 0                  # short sequence
    with torch.no_grad():
        ref = torch.sigmoid(model(torch.from_numpy(X))).numpy()
    r = _run(pkg, X, tmp_path)
    assert r.returncode == 0, r.stderr
    cpp = np.array([float(x) for x in r.stdout.split()])
    assert np.abs(cpp - ref).max() <= 1e-5


def test_cpp_rejects_wrong_width_and_batch(tmp_path):
    torch.manual_seed(0)
    pkg = export_package(TransformerClassifier(30, maxlen=8), 8, tmp_path / "m.pt2", "t")
    bad = _run(pkg, np.ones((2, 5), dtype=np.int64), tmp_path)
    assert bad.returncode == 1 and "model expects 8" in bad.stderr
    big = _run(pkg, np.ones((2, 8), dtype=np.int64), tmp_path, batch=MAX_BATCH + 1)
    assert big.returncode == 1 and "exceeds the exported limit" in big.stderr
