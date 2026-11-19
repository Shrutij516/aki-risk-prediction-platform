"""Vocabulary, encoding, splits, and a synthetic sequence generator."""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

PAD, OOV = 0, 1


@dataclass
class Vocab:
    """Built from TRAINING sequences only. 0 = padding, 1 = out-of-vocabulary.

    (The notebooks built token2id from all patients, test included, with no OOV id.)
    """
    token2id: dict[str, int]

    @classmethod
    def build(cls, sequences, min_count: int = 1) -> "Vocab":
        counts = Counter(t for seq in sequences for t in seq)
        kept = sorted((t for t, c in counts.items() if c >= min_count), key=lambda t: (-counts[t], t))
        return cls({t: i + 2 for i, t in enumerate(kept)})

    def __len__(self) -> int:  # size of the embedding table
        return len(self.token2id) + 2

    def encode(self, seq) -> list[int]:
        return [self.token2id.get(t, OOV) for t in seq]


def pad_truncate(ids_list, maxlen: int, truncate: str = "keep_first") -> np.ndarray:
    """Post-padding with 0. truncate='keep_first' matches the notebooks' fast_pad_sequences(truncating='post')
    (keeps the OLDEST maxlen events); 'keep_last' keeps the most recent maxlen events."""
    if truncate not in ("keep_first", "keep_last"):
        raise ValueError(truncate)
    out = np.zeros((len(ids_list), maxlen), dtype=np.int64)
    for i, ids in enumerate(ids_list):
        ids = list(ids)
        if len(ids) > maxlen:
            ids = ids[:maxlen] if truncate == "keep_first" else ids[-maxlen:]
        out[i, :len(ids)] = ids
    return out


def split_indices(labels: np.ndarray, test_frac: float = 0.2, val_frac: float = 0.2, seed: int = 42):
    """Stratified patient-level train/val/test split.

    Test = 20% (the original test size, stratified, seed 42). Val = 20% of the
    remainder, stratified (the original used Keras validation_split=0.2, i.e. the
    last 20% of the training array, unstratified). Final proportions 64/16/20.
    """
    idx = np.arange(len(labels))
    rest, test = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=labels)
    train, val = train_test_split(rest, test_size=val_frac, random_state=seed, stratify=labels[rest])
    return np.sort(train), np.sort(val), np.sort(test)


def fingerprint(patient_ids, labels) -> str:
    """Order-independent hash of (patient_id, label) pairs. Identifies a dataset without exposing IDs."""
    h = hashlib.sha256()
    for pid, y in sorted(zip(map(str, patient_ids), map(int, labels))):
        h.update(f"{pid}\t{y}\n".encode())
    return h.hexdigest()[:16]


def synthetic_sequences(n: int = 2000, prevalence: float = 0.1, seed: int = 0, mean_len: int = 120):
    """Synthetic EHR token sequences with a learnable signal. No real data involved.

    Positives get CKD-progression tokens (N18.4, eGFR G4/G5, high creatinine) more often.
    Returns (patient_ids, labels, sequences).
    """
    rng = np.random.default_rng(seed)
    background = ([f"diagnosis_D{i:03d}" for i in range(300)] + [f"procedure_{90000 + i}" for i in range(150)]
                  + [f"medication_{100000 + i}" for i in range(150)]
                  + ["lab_creatinine_normal", "lab_egfr_g1", "lab_egfr_g2", "lab_bun_normal"])
    risk = ["diagnosis_N18.4", "diagnosis_N18.5", "lab_egfr_g4", "lab_egfr_g5", "lab_creatinine_severe", "lab_bun_high"]
    labels = (rng.random(n) < prevalence).astype(int)
    seqs = []
    for y in labels:
        length = max(3, int(rng.lognormal(np.log(mean_len), 0.8)))
        seq = list(rng.choice(background, size=length))
        n_risk = rng.poisson(6 if y else 1)
        for _ in range(n_risk):
            seq.insert(int(rng.integers(0, len(seq) + 1)), str(rng.choice(risk)))
        seqs.append(seq)
    pids = [f"SYN{i:06d}" for i in range(n)]
    return pids, labels, seqs
