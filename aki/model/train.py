"""Training loop for one configuration: Adam, focal loss, early stopping on validation ROC-AUC."""
from __future__ import annotations

import copy
import os
import random
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from aki.model.losses import binary_focal_loss
from aki.model.models import build_model


@dataclass
class TrainConfig:
    model: str = "retain"            # retain | transformer
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0
    dropout: float = 0.1
    lr: float = 1e-3
    adam_eps: float = 1e-7           # Keras Adam default (PyTorch default is 1e-8)
    batch_size: int = 32
    epochs: int = 7                  # final notebook grid used 7
    patience: int = 3
    maxlen: int = 400
    mask_padding: bool = False       # Transformer only; False = original behaviour
    seed: int = 42
    device: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(cfg: TrainConfig, vocab_size: int, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
        y_val: np.ndarray, on_epoch: Callable[[int, dict], None] | None = None):
    """Train with early stopping on validation ROC-AUC; returns (best model, history).

    Best-epoch weights are always restored at the end (Keras 3 restore_best_weights behaviour).
    """
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    model = build_model(cfg.model, vocab_size, cfg.maxlen, cfg.dropout, cfg.mask_padding).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    gen = torch.Generator().manual_seed(cfg.seed)
    Xt, yt = torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))

    best_auc, best_state, best_epoch, bad, history = -np.inf, None, -1, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total, n = 0.0, 0
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            xb, yb = Xt[b].to(device), yt[b].to(device)
            loss = binary_focal_loss(model(xb), yb, cfg.focal_alpha, cfg.focal_gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
            n += len(b)
        p_val = predict_proba(model, X_val, device)
        val_auc = float(roc_auc_score(y_val, p_val)) if 0 < y_val.sum() < len(y_val) else float("nan")
        rec = {"epoch": epoch, "train_loss": total / max(n, 1), "val_roc_auc": val_auc}
        history.append(rec)
        if on_epoch:
            on_epoch(epoch, rec)
        if val_auc > best_auc:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_epoch": best_epoch, "best_val_roc_auc": best_auc, "device": str(device)}
