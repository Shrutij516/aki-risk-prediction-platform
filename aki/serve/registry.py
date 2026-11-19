"""Load exported model bundles (outputs/models/<key>/) and score token sequences on CPU."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aki.model.data import OOV, pad_truncate
from aki.model.models import build_model

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = ROOT / "outputs" / "models"


@dataclass
class ServingModel:
    key: str
    meta: dict
    token2id: dict
    model: torch.nn.Module

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    def _encode(self, tokens: list[str]) -> tuple[np.ndarray, list[int]]:
        ids = [self.token2id.get(t, OOV) for t in tokens]
        return pad_truncate([ids], self.meta["maxlen"], self.meta["truncate"]), ids

    @torch.no_grad()
    def score(self, tokens: list[str], top_k: int = 10) -> dict:
        """Probability for one patient's token sequence (+ RETAIN alpha attention).

        Positions refer to the original sequence order; with truncate='keep_first'
        only the first maxlen events are seen by the model, and attention covers those.
        """
        X, ids = self._encode(tokens)
        x = torch.from_numpy(X)
        out = {"n_events": len(tokens), "n_used": int(min(len(tokens), self.meta["maxlen"])),
               "n_oov": int(sum(1 for i in ids[: self.meta["maxlen"]] if i == OOV))}
        if self.meta["model"] == "retain":
            logit, alpha, _ = self.model(x, return_attention=True)
            a = alpha[0, : out["n_used"]].numpy()
            offset = 0 if self.meta["truncate"] == "keep_first" else max(0, len(tokens) - self.meta["maxlen"])
            order = np.argsort(-a)[:top_k]
            out["attention"] = [{"position": int(offset + i), "token": tokens[offset + i], "alpha": float(a[i])}
                                for i in order]
        else:
            logit = self.model(x)
            out["attention"] = None
        p = float(torch.sigmoid(logit)[0])
        out.update(probability=p, threshold=self.threshold, above_threshold=p >= self.threshold)
        return out


def load_bundle(path: Path) -> ServingModel:
    meta = json.loads((path / "meta.json").read_text())
    token2id = json.loads((path / "vocab.json").read_text())
    model = build_model(meta["model"], meta["vocab_size"], meta["maxlen"], meta["dropout"], meta["mask_padding"])
    model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu"))
    model.eval()
    return ServingModel(meta["key"], meta, token2id, model)


def load_all(models_dir: Path = DEFAULT_MODELS_DIR) -> dict[str, ServingModel]:
    if not models_dir.exists():
        return {}
    return {p.name: load_bundle(p) for p in sorted(models_dir.iterdir()) if (p / "meta.json").exists()}
