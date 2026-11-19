# C++ inference (LibTorch + AOTInductor)

*Reconstruction document (2026-11-19). New work: the original project had no C++ (see PROVENANCE.md).*

## What it does
`cpp/build/aki_infer` is a command-line program. It loads one exported trained model and prints a probability for each input row.

- **Model:** the reconstructed PyTorch **Transformer with binned labs** (`transformer_withlabs`), the best of the four by validation ROC-AUC. It is exported without retraining from the same weights the API serves.
- **Input:** a text file with one patient per line. Each line holds 400 integer token ids, already encoded with the model's training-only vocabulary and padded by Python.
- **Output:** one probability per line, `sigmoid(logit)`, to 9 significant digits.

**Scope:** the program only runs the model. Vocabulary lookup, truncation, padding, the threshold and attention all stay in Python. It never reads clinical codes or patient identifiers.

## How the export works
- `torch.jit.trace` / TorchScript prints a deprecation warning in the installed PyTorch 2.15 ("Please switch to `torch.compile` or `torch.export`"), so this uses the supported path instead:
  1. `torch.export.export` with a dynamic batch dimension (1–4,096) and a fixed sequence length of 400;
  2. `torch._inductor.aoti_compile_and_package`, which compiles ahead of time to a `.pt2` package;
  3. in C++, `torch::inductor::AOTIModelPackageLoader` from the LibTorch that ships in the PyTorch wheel.
- **No architecture changes were needed.** The custom Keras-style attention, the GRUs (RETAIN, checked in tests) and the -1e9 masking all export cleanly.
- **The package carries metadata:** model name, `aki_maxlen` and `aki_max_batch`. The compiled code doesn't check input shapes at run time, so the C++ program checks them and refuses a mismatched width or an oversized batch with exit code 1. Before this check was added, a 3-id input was silently scored, which is why it exists.

## Build and run

```bash
.venv/bin/python -m aki.serve.export_cpp --model transformer_withlabs   # → outputs/cpp/*.pt2 + parity inputs (local)
scripts/build_cpp.sh                                                    # → cpp/build/aki_infer (cmake + LibTorch from the venv)
cpp/build/aki_infer outputs/cpp/transformer_withlabs.pt2 outputs/cpp/inputs_synthetic_demo.txt 256 > probs.txt
.venv/bin/python scripts/cpp_parity.py                                  # C++ vs eager PyTorch, tolerance 1e-5
```

The build requirements are a C++17 compiler (Apple clang 17 was used), `cmake` (installed into the venv by the build script) and the venv's PyTorch 2.15 wheel. `cmake` prints `library kineto not found`: kineto is the profiler library, and inference doesn't use it.

## Parity results
Run on 2026-11-19; full output in `reports/cpp_parity.json`, which is local and gitignored.

| Input set | Rows | Batch 1: max \|C++ − Python\| | Batch 256: max \|C++ − Python\| |
|---|---|---|---|
| Synthetic demo patients | 447 | 8.9e-8 | 6.0e-8 |
| Real held-out test patients (first 1,000; encoded ids only, local) | 1,000 | 9.0e-8 | 9.0e-8 |
| Edge cases (all padding, one event, all unseen codes) | 3 | 3.0e-8 | 5.9e-8 |

Every set is well inside the 1e-5 tolerance. The Python AOTInductor package also matches eager PyTorch to within 9e-8.

`tests/test_cpp.py` runs automatically once the program is built. It exports tiny untrained Transformer and RETAIN models, checks parity on random synthetic ids, and checks that a wrong width or an oversized batch is refused.

## Limitations
- **One model.** Only `transformer_withlabs` is exported for C++; the other three can be exported the same way with `--model`.
- **Numbers in, numbers out.** The program only takes encoded ids, so it can't score a raw patient timeline without the Python preprocessing.
- **Not used by the web app.** The API scores in Python; the C++ program is a separate, verified runtime.
- **Tied to one PyTorch version.** A `.pt2` package is compiled for one PyTorch version and one platform (here PyTorch 2.15, macOS arm64, CPU). It has to be re-exported after a PyTorch upgrade or on another machine.
- **The text input format isn't built for speed.** For 1,000 patients, reading and inference take about 0.85 s, after about 0.4 s to load the package.
- **Same model, same limits.** It's the same research model, with no clinical validation.
