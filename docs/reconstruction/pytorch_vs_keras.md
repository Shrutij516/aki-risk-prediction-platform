# PyTorch reconstruction vs the original TensorFlow/Keras notebooks

*Reconstruction document (2026-11-17, public version: figures from the TriNetX extract removed; detailed version kept locally). The original models were written in Keras (`AKI_Dialysis_Prediction-main/*.ipynb`). The PyTorch code in `aki/model/` is a new implementation that follows them. Results it produces are new results; they never replace or merge with the original table.*

## Verified to match the original

| Item | How it was checked |
|---|---|
| Event extraction, observation window, ≥3-event filter, lab filtering and binning, lab re-entry quirk, vocabulary | All 17 checked counts from the saved notebook outputs (event counts per type, filter stages, lab pipeline stages, sequence patients, vocabulary sizes) reproduced exactly on the recovered data. The detailed comparison is a local-only report |
| Event order inside a patient (decides what truncation keeps) | Notebook cell 84 printed encoded prefixes for 5 patients. All 50 IDs match after rebuilding `token2id` by first appearance |
| Cohort and labels | All 8 counts reproduced in SQL (`reports/cohort_validation.json`) |
| RETAIN architecture | Parameter count equals the Keras `model.summary()` in the saved notebook outputs, for both the with-labs and no-labs vocabularies (`tests/test_models.py`) |
| Transformer architecture | Parameter count equals the Keras summary. This requires Keras `MultiHeadAttention(key_dim=128)` semantics, i.e. 4 heads × 128 dims projected back to 128. `torch.nn.MultiheadAttention` would give 4 × 32 and a different model |
| Focal loss | Same formula, ε = 1e-7 clipping and mean reduction. Numerically equal to a NumPy reference (`test_focal_loss_matches_keras_formula`) |
| Layer order, activations, LayerNorm ε = 1e-6, dropout placement, post-padding, `truncating='post'` (oldest 400 events kept) | Code review against the notebook cells; truncation test |
| Initialisation | Keras defaults reproduced: glorot-uniform kernels, zero biases, uniform(±0.05) embeddings, orthogonal GRU recurrent kernels |
| Optimiser and schedule | Adam lr 1e-3, ε 1e-7 (Keras default), batch 32, 7 epochs, early stopping on validation ROC-AUC with patience 3, final grid α ∈ {0.25, 0.5} × dropout ∈ {0.1, 0.2}, threshold grid 0.05–0.95 step 0.01, selection by F1 |

## Intentional differences (evaluation protocol)

| # | Original | Reconstruction | Why |
|---|---|---|---|
| 1 | α, dropout and decision threshold chosen on the **test** set | Chosen on **validation** only. The test set is scored once, after selection. `test_test_set_does_not_influence_selection` scrambles every test sequence and checks that selection doesn't change | Test-set tuning inflates F1, precision and sensitivity |
| 2 | 80/20 split; validation = Keras `validation_split=0.2`, the last 20% of the training array, unstratified | Stratified 64/16/20 split, seed 42 | Explicit and stratified with a ~10% positive rate. Same effective proportions |
| 3 | `token2id` built from all patients, test included; no OOV id | Vocabulary from training patients only; id 1 = OOV | No information from validation/test patients. The embedding table on real data is therefore smaller than in the Keras summaries |
| 4 | Best config retrained from scratch, then evaluated | The selected grid model (already early-stopped on validation) is evaluated directly | Avoids an extra unvalidated training run |
| 5 | Keras `AUC` metric for early stopping (200-threshold approximation) | Exact `sklearn.roc_auc_score` | Can pick a different best epoch in near-ties |
| 6 | `restore_best_weights=True` (Keras 3 restores at the end of training) | Always restores the best-epoch weights | Same as Keras 3; Keras 2 restored only if stopped early |
| 7 | Point estimates only | Also 95% bootstrap CIs on the test set (1,000 patient resamples) and test TPR at thresholds chosen on validation for FPR 5% / 1% | A single test split carries real variance |

## Framework-level differences (unavoidable)

- **Random number streams differ.** Weights, dropout masks and shuffles cannot match Keras bit for bit, so numbers will differ from the originals even with identical settings.
- **GRU.** Keras `GRU(reset_after=True)` has the same formulation as `torch.nn.GRU`, with input and recurrent biases kept separately; the matching parameter counts confirm it. Gate order differs (Keras z,r,h vs PyTorch r,z,n), but the initial weight distribution is the same.
- **Determinism.** Training is bitwise repeatable on CPU (`test_training_is_deterministic_on_cpu`). Apple MPS kernels are not, so MPS runs vary slightly between repeats. The device is recorded with each run.
- **Transformer padding mask (inferred, not executed in TensorFlow).** In the original, `mask_zero=True` sits inside the custom `TokenAndPositionEmbedding` layer. Under Keras 3 a custom layer without `compute_mask` doesn't pass the mask on, so attention and `GlobalAveragePooling1D` include padded positions. The default `mask_padding=False` reproduces that. I could not confirm it by running the original TensorFlow code, because the Colab environment is gone. `--mask-padding` gives the corrected variant.

## Corrected variants (opt-in, reported separately)

| Flag | Effect |
|---|---|
| `--truncate keep_last` | Keep the 400 most recent pre-blanking events instead of the oldest. The median sequence is close to 400 events, so many patients are affected |
| `--mask-padding` | Transformer: mask padding in attention and pooling |

## Not reconstructed
- `Transformer_WithLabs` naive-LOINC variant: it needs `standardized_terminology.csv`, which isn't in `data/`, and the README already marks it as not comparable. The PyTorch Transformer can be trained on the binned-lab sequences instead, as a new configuration.
- The logistic-regression / random-forest tabular baselines in `AKI_Project.ipynb` and the progress report.
- Kept as in the original and not "fixed": exact duplicate events, NDC/RxNorm twin prescriptions, tokens without a code system, and creatinine rows mislabelled as µmol/L (see `data_inventory.md`).

## How to run

```bash
python -m aki.model.sequences --dsn <db with full cohort> [--labs] --out outputs/sequences_withlabs.parquet
python -m aki.model.experiment --data outputs/sequences_withlabs.parquet --model retain
python -m aki.model.experiment --synthetic 3000 --model transformer        # no real data
mlflow ui --backend-store-uri sqlite:///mlflow.db                           # local only
```

The MLflow store (`mlflow.db`, `mlartifacts/`, including model weights), `outputs/` and `reports/` are gitignored.
