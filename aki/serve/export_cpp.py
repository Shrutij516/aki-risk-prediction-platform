"""Export a trained model for C++ inference with torch.export + AOTInductor (no retraining).

TorchScript (torch.jit.trace) is deprecated in the installed PyTorch 2.15, so this uses the
supported path: torch.export -> AOTInductor-compiled package (.pt2), loaded in C++ with
LibTorch's torch::inductor::AOTIModelPackageLoader (cpp/aki_infer.cpp).

Also writes parity inputs (already-encoded, padded token-id matrices: one patient per line)
and the eager-PyTorch reference probabilities for the same rows.

    python -m aki.serve.export_cpp --model transformer_withlabs [--real-test-rows 1000]

Output goes to outputs/cpp/ (gitignored). The real-test inputs are encoded patient
sequences (integer ids only) and stay local.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aki.model.data import pad_truncate, split_indices
from aki.serve.registry import load_bundle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "cpp"


def _encode(m, token_lists) -> np.ndarray:
    return pad_truncate([[m.token2id.get(t, 1) for t in toks] for toks in token_lists], m.meta["maxlen"], m.meta["truncate"])


def _write_matrix(path: Path, X: np.ndarray) -> None:
    with path.open("w") as f:
        for row in X:
            f.write(" ".join(map(str, row.tolist())) + "\n")


@torch.no_grad()
def eager_probs(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    return np.concatenate([torch.sigmoid(model(torch.from_numpy(X[i:i + batch]))).numpy() for i in range(0, len(X), batch)])


MAX_BATCH = 4096


def export_package(model: torch.nn.Module, maxlen: int, path: Path, name: str) -> Path:
    """torch.export with a dynamic batch dimension (1..MAX_BATCH) -> AOTInductor .pt2 package.

    maxlen and the batch limit are embedded as package metadata so the C++ runner can reject
    inputs the compiled graph was not built for (it does not validate shapes at run time).
    """
    model = model.eval()
    example = torch.ones(2, maxlen, dtype=torch.long)
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    ep = torch.export.export(model, (example,), dynamic_shapes={"ids": {0: batch}})
    torch._inductor.aoti_compile_and_package(ep, package_path=str(path), inductor_configs={
        "aot_inductor.metadata": {"aki_model": name, "aki_maxlen": str(maxlen), "aki_max_batch": str(MAX_BATCH),
                                  "aki_output": "logit; probability = sigmoid(logit)"}})
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="transformer_withlabs")
    ap.add_argument("--real-test-rows", type=int, default=1000, help="0 to skip real held-out inputs")
    a = ap.parse_args()

    m = load_bundle(ROOT / "outputs" / "models" / a.model)
    model = m.model.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    maxlen = m.meta["maxlen"]

    t0 = time.time()
    pkg = export_package(model, maxlen, OUT / f"{a.model}.pt2", a.model)
    export_seconds = round(time.time() - t0, 1)

    variant = "withlabs" if m.meta["labs"] else "nolabs"
    sets = {}
    demo = ROOT / "outputs" / "demo_sequences" / f"sequences_{variant}.parquet"
    if demo.exists():
        sets["synthetic_demo"] = _encode(m, [list(t) for t in pd.read_parquet(demo)["tokens"]])
    if a.real_test_rows:
        df = pd.read_parquet(ROOT / "outputs" / f"sequences_{variant}.parquet")
        labels = df["label"].to_numpy().astype(int)
        _, _, te = split_indices(labels, seed=42)
        te = te[: a.real_test_rows]
        sets["real_heldout"] = _encode(m, [list(df["tokens"].iloc[i]) for i in te])
    # Edge cases: all padding, a single event, every position filled with OOV.
    edge = np.zeros((3, maxlen), dtype=np.int64)
    edge[1, 0] = 2
    edge[2, :] = 1
    sets["edge_cases"] = edge

    manifest = {"model": a.model, "package": pkg.name, "maxlen": maxlen, "export": "torch.export + AOTInductor",
                "torch_version": torch.__version__, "export_seconds": export_seconds, "sets": {}}
    loaded = torch._inductor.aoti_load_package(str(pkg))
    for name, X in sets.items():
        _write_matrix(OUT / f"inputs_{name}.txt", X)
        ref = eager_probs(model, X)
        np.savetxt(OUT / f"ref_{name}.txt", ref, fmt="%.9g")
        aoti_py = torch.sigmoid(loaded(torch.from_numpy(X))).numpy()
        manifest["sets"][name] = {"rows": int(len(X)), "python_aoti_vs_eager_max_abs_diff": float(np.abs(aoti_py - ref).max())}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
