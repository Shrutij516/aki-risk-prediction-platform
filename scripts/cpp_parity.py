"""Python vs C++ numerical parity for the exported model.

Requires: python -m aki.serve.export_cpp  (package + inputs + eager references in outputs/cpp/)
          scripts/build_cpp.sh             (cpp/build/aki_infer)

Runs the C++ executable on every input set at two batch sizes, compares with the eager PyTorch
probabilities, and fails if any |difference| > 1e-5. Writes reports/cpp_parity.json (aggregates).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "cpp"
BIN = ROOT / "cpp" / "build" / "aki_infer"
TOL = 1e-5


def main() -> int:
    manifest = json.loads((OUT / "manifest.json").read_text())
    pkg = OUT / manifest["package"]
    report = {"model": manifest["model"], "tolerance": TOL, "export": manifest["export"],
              "torch_version": manifest["torch_version"], "sets": {}}
    ok = True
    for name in manifest["sets"]:
        ref = np.loadtxt(OUT / f"ref_{name}.txt", ndmin=1)
        for batch in (1, 256):
            t0 = time.time()
            r = subprocess.run([str(BIN), str(pkg), str(OUT / f"inputs_{name}.txt"), str(batch)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAIL {name} batch={batch}: {r.stderr.strip()}")
                return 1
            cpp = np.array([float(x) for x in r.stdout.split()])
            if cpp.shape != ref.shape:
                print(f"FAIL {name}: {cpp.shape} outputs vs {ref.shape} references")
                return 1
            diff = float(np.abs(cpp - ref).max())
            passed = diff <= TOL
            ok &= passed
            report["sets"][f"{name}_batch{batch}"] = {"rows": int(len(ref)), "max_abs_diff": diff,
                                                      "mean_abs_diff": float(np.abs(cpp - ref).mean()),
                                                      "passed": passed, "seconds": round(time.time() - t0, 2),
                                                      "cpp_timing": r.stderr.strip()}
            print(f"{'OK  ' if passed else 'FAIL'} {name:15s} batch={batch:<4d} rows={len(ref):5d} max|C++ - Python| = {diff:.3e}")
    report["passed"] = ok
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "cpp_parity.json").write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
