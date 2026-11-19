"""PyTorch models vs the original Keras architectures, plus loss/metric/split behaviour."""
import numpy as np
import pytest
import torch

from aki.model.data import OOV, Vocab, pad_truncate, split_indices, synthetic_sequences
from aki.model.losses import binary_focal_loss
from aki.model.metrics import THRESHOLD_GRID, compute_metrics, select_threshold, threshold_at_fpr
from aki.model.models import RETAIN, TransformerClassifier
from aki.model.train import TrainConfig, fit


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def keras_retain_params(vocab: int, e: int = 128, h: int = 128) -> int:
    """Parameter count of the notebooks' Keras RETAIN, layer by layer (GRU reset_after=True)."""
    gru = 3 * (e * h + h * h + 2 * h)
    return vocab * e + 2 * gru + (h + 1) + (h * e + e) + (e + 1)


def keras_transformer_params(vocab: int, maxlen: int = 400, e: int = 128, heads: int = 4, ff: int = 128) -> int:
    """Keras Transformer: token+position embeddings, MultiHeadAttention(key_dim=e), FFN, 2 LayerNorms, head."""
    mha = 3 * (e * heads * e + heads * e) + (heads * e * e + e)
    block = mha + (e * ff + ff) + (ff * e + e) + 2 * (2 * e)
    return vocab * e + maxlen * e + block + (e * 64 + 64) + (64 + 1)


@pytest.mark.parametrize("vocab", [50, 1000, 12345])
def test_param_counts_match_keras_architecture(vocab):
    assert n_params(RETAIN(vocab)) == keras_retain_params(vocab)
    assert n_params(TransformerClassifier(vocab, maxlen=400)) == keras_transformer_params(vocab)


def test_param_counts_match_saved_keras_summaries():
    """Exact totals printed by model.summary() in the saved notebooks. The vocabulary sizes are specific
    to the TriNetX extract, so they are read from local/reference_counts.json (gitignored)."""
    import json
    from pathlib import Path

    ref = Path(__file__).resolve().parents[1] / "local" / "reference_counts.json"
    if not ref.exists():
        pytest.skip("local reference counts not available (extract-specific values are not published)")
    for r in json.loads(ref.read_text())["keras_param_counts"]:
        model = RETAIN(r["vocab_with_pad"]) if r["model"] == "retain" else TransformerClassifier(r["vocab_with_pad"], maxlen=400)
        assert n_params(model) == r["params"], r["source"]


def test_retain_attention_is_a_distribution_over_real_events():
    torch.manual_seed(0)
    m = RETAIN(50).eval()
    ids = torch.tensor([[5, 6, 7, 0, 0, 0], [1, 2, 3, 4, 5, 6]])
    _, alpha, beta = m(ids, return_attention=True)
    assert torch.allclose(alpha.sum(1), torch.ones(2), atol=1e-6)
    assert alpha[0, 3:].max() < 1e-12
    assert beta.shape == (2, 6, 128) and beta.abs().max() <= 1


def test_transformer_mask_padding_makes_output_independent_of_pad_length():
    torch.manual_seed(0)
    m = TransformerClassifier(50, maxlen=20, mask_padding=True).eval()
    short = torch.tensor([[5, 6, 7] + [0] * 7])
    long = torch.tensor([[5, 6, 7] + [0] * 17])
    assert torch.allclose(m(short), m(long), atol=1e-5)
    # original behaviour (no mask) depends on padding, as in the Keras model
    m.mask_padding = False
    assert not torch.allclose(m(short), m(long), atol=1e-5)


def test_focal_loss_matches_keras_formula():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=64).astype(np.float32)
    y = (rng.random(64) < 0.3).astype(np.float32)
    p = np.clip(1 / (1 + np.exp(-logits)), 1e-7, 1 - 1e-7)
    pt = np.where(y == 1, p, 1 - p)
    a = np.where(y == 1, 0.25, 0.75)
    ref = np.mean(-a * (1 - pt) ** 2 * np.log(pt))
    got = binary_focal_loss(torch.from_numpy(logits), torch.from_numpy(y), alpha=0.25, gamma=2.0)
    assert got.item() == pytest.approx(ref, rel=1e-5)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(y))
    assert binary_focal_loss(torch.from_numpy(logits), torch.from_numpy(y), 0.5, 0.0).item() == pytest.approx(0.5 * bce.item(), rel=1e-4)


def test_truncation_modes():
    X = pad_truncate([[1, 2, 3, 4, 5], [7]], maxlen=3)
    assert X.tolist() == [[1, 2, 3], [7, 0, 0]]  # original: keep first (oldest), post-pad
    assert pad_truncate([[1, 2, 3, 4, 5]], 3, "keep_last").tolist() == [[3, 4, 5]]


def test_vocab_is_train_only_with_oov():
    v = Vocab.build([["a", "b", "a"], ["c"]])
    assert v.encode(["a", "zzz"]) == [2, OOV]
    assert len(v) == 3 + 2


def test_split_is_disjoint_stratified_and_deterministic():
    y = np.array([1] * 100 + [0] * 900)
    tr, va, te = split_indices(y, seed=42)
    assert len(set(tr) | set(va) | set(te)) == 1000 and not (set(tr) & set(te) or set(va) & set(te) or set(tr) & set(va))
    assert (len(tr), len(va), len(te)) == (640, 160, 200)
    assert y[te].sum() == 20 and y[va].sum() == 16
    assert all((a == b).all() for a, b in zip((tr, va, te), split_indices(y, seed=42)))


def test_threshold_selection_and_fpr_threshold():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    t, f1 = select_threshold(y, p)
    assert t in THRESHOLD_GRID and f1 == 1.0 and 0.6 < t <= 0.7
    t5 = threshold_at_fpr(y, p, 0.0)
    assert compute_metrics(y, p, t5)["fp"] == 0


def _small_synthetic(n=600, seed=0):
    pids, y, seqs = synthetic_sequences(n, prevalence=0.2, seed=seed, mean_len=40)
    tr, va, _ = split_indices(y, seed=seed)
    v = Vocab.build([seqs[i] for i in tr])
    enc = lambda idx: pad_truncate([v.encode(seqs[i]) for i in idx], 64)  # noqa: E731
    return len(v), enc(tr), y[tr], enc(va), y[va]


@pytest.mark.parametrize("kind", ["retain", "transformer"])
def test_training_is_deterministic_on_cpu(kind):
    V, Xtr, ytr, Xva, yva = _small_synthetic(300)
    cfg = TrainConfig(model=kind, epochs=2, maxlen=64, device="cpu", seed=1)
    h1 = fit(cfg, V, Xtr, ytr, Xva, yva)[1]["history"]
    h2 = fit(cfg, V, Xtr, ytr, Xva, yva)[1]["history"]
    assert h1 == h2


@pytest.mark.parametrize("kind", ["retain", "transformer"])
def test_models_learn_synthetic_signal(kind):
    V, Xtr, ytr, Xva, yva = _small_synthetic(800)
    cfg = TrainConfig(model=kind, epochs=6, patience=6, maxlen=64, device="cpu", seed=0, mask_padding=True)
    _, info = fit(cfg, V, Xtr, ytr, Xva, yva)
    assert info["best_val_roc_auc"] > 0.8, info["history"]
