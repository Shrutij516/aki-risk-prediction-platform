"""PyTorch re-implementations of the original Keras Transformer and RETAIN.

These are NEW implementations (the originals were TensorFlow/Keras). They follow
the Keras code in the notebooks layer by layer; parameter counts equal the saved
Keras model summaries (tests/test_models.py). Differences are listed in
docs/reconstruction/pytorch_vs_keras.md.

Both models return logits; probabilities are sigmoid(logits).
"""
from __future__ import annotations

import math

import torch
from torch import nn

# ----------------------------------------------------------- Keras-style init
# Keras defaults: Dense/MHA kernels glorot_uniform, biases zeros, Embedding
# uniform(-0.05, 0.05), GRU kernel glorot_uniform, recurrent kernel orthogonal.


def _keras_dense(layer: nn.Linear) -> nn.Linear:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _keras_embedding(layer: nn.Embedding) -> nn.Embedding:
    nn.init.uniform_(layer.weight, -0.05, 0.05)
    return layer


def _keras_gru(gru: nn.GRU) -> nn.GRU:
    h = gru.hidden_size
    for name, p in gru.named_parameters():
        if name.startswith("weight_ih"):
            # Keras kernel is (in, 3h) initialised as one glorot matrix.
            fan_in, fan_out = p.shape[1], p.shape[0]
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(p, -bound, bound)
        elif name.startswith("weight_hh"):
            # Keras recurrent kernel is (h, 3h) orthogonal.
            w = torch.empty(h, 3 * h)
            nn.init.orthogonal_(w)
            with torch.no_grad():
                p.copy_(w.t())
        else:
            nn.init.zeros_(p)
    return gru


# ----------------------------------------------------------- Transformer
class KerasMultiHeadAttention(nn.Module):
    """keras.layers.MultiHeadAttention(num_heads, key_dim): each head has key_dim
    dims (NOT embed_dim / num_heads), then projects back to embed_dim."""

    def __init__(self, embed_dim: int, num_heads: int, key_dim: int):
        super().__init__()
        self.h, self.d = num_heads, key_dim
        inner = num_heads * key_dim
        self.q = _keras_dense(nn.Linear(embed_dim, inner))
        self.k = _keras_dense(nn.Linear(embed_dim, inner))
        self.v = _keras_dense(nn.Linear(embed_dim, inner))
        self.o = _keras_dense(nn.Linear(inner, embed_dim))

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = (f(x).view(b, t, self.h, self.d).transpose(1, 2) for f in (self.q, self.k, self.v))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d)
        if key_mask is not None:  # (B, T) True = real token
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.h * self.d)
        return self.o(out)


class TransformerBlock(nn.Module):
    """Post-norm block from the notebook: LN(x + Drop(MHA(x))) then LN(out1 + Drop(FFN(out1)))."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float):
        super().__init__()
        self.att = KerasMultiHeadAttention(embed_dim, num_heads, key_dim=embed_dim)
        self.ffn = nn.Sequential(_keras_dense(nn.Linear(embed_dim, ff_dim)), nn.ReLU(),
                                 _keras_dense(nn.Linear(ff_dim, embed_dim)))
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.drop1 = nn.Dropout(rate)
        self.drop2 = nn.Dropout(rate)

    def forward(self, x, key_mask=None):
        out1 = self.ln1(x + self.drop1(self.att(x, key_mask)))
        return self.ln2(out1 + self.drop2(self.ffn(out1)))


class TransformerClassifier(nn.Module):
    """build_transformer_model() from Transformer_*.ipynb.

    mask_padding=False reproduces the original's effective behaviour: the
    TokenAndPositionEmbedding custom layer does not propagate Keras' mask_zero
    mask, so attention and GlobalAveragePooling1D include padded positions.
    mask_padding=True masks padding in attention and pooling (corrected variant).
    """

    def __init__(self, vocab_size: int, maxlen: int = 400, embed_dim: int = 128, num_heads: int = 4,
                 ff_dim: int = 128, rate: float = 0.1, mask_padding: bool = False):
        super().__init__()
        self.mask_padding = mask_padding
        self.token_emb = _keras_embedding(nn.Embedding(vocab_size, embed_dim))
        self.pos_emb = _keras_embedding(nn.Embedding(maxlen, embed_dim))
        self.block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
        self.drop1 = nn.Dropout(rate)
        self.dense = _keras_dense(nn.Linear(embed_dim, 64))
        self.drop2 = nn.Dropout(rate)
        self.out = _keras_dense(nn.Linear(64, 1))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.token_emb(ids) + self.pos_emb(pos)[None]
        mask = ids != 0
        x = self.block(x, mask if self.mask_padding else None)
        if self.mask_padding:
            m = mask.unsqueeze(-1).float()
            x = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            x = x.mean(1)
        x = self.drop2(torch.relu(self.dense(self.drop1(x))))
        return self.out(x).squeeze(-1)


# ----------------------------------------------------------- RETAIN
class RETAIN(nn.Module):
    """build_retain_model() from RETAIN_*.ipynb (Choi et al. 2016 two-level attention).

    Same as the Keras graph: embeddings zeroed at padding, dropout, sequence
    reversed, GRU over the full reversed padded sequence (post-padding becomes
    leading zeros), alpha logits masked with -1e9 before softmax over time,
    beta = tanh(Dense(E)), context = sum_t alpha_t * beta_t * v_t, dropout, Dense(1).
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, alpha_hidden: int = 128,
                 beta_hidden: int = 128, rate: float = 0.1):
        super().__init__()
        self.embedding = _keras_embedding(nn.Embedding(vocab_size, embed_dim))
        self.drop_in = nn.Dropout(rate)
        self.alpha_gru = _keras_gru(nn.GRU(embed_dim, alpha_hidden, batch_first=True))
        self.alpha_dense = _keras_dense(nn.Linear(alpha_hidden, 1))
        self.beta_gru = _keras_gru(nn.GRU(embed_dim, beta_hidden, batch_first=True))
        self.beta_dense = _keras_dense(nn.Linear(beta_hidden, embed_dim))
        self.drop_ctx = nn.Dropout(rate)
        self.out = _keras_dense(nn.Linear(embed_dim, 1))

    def forward(self, ids: torch.Tensor, return_attention: bool = False):
        mask = (ids != 0).float().unsqueeze(-1)                      # (B, T, 1)
        v = self.drop_in(self.embedding(ids) * mask)                 # (B, T, E)
        v_rev = torch.flip(v, dims=[1])
        g, _ = self.alpha_gru(v_rev)
        alpha_logits = torch.flip(self.alpha_dense(g), dims=[1]) + (1.0 - mask) * -1e9
        alpha = torch.softmax(alpha_logits, dim=1)                   # (B, T, 1)
        h, _ = self.beta_gru(v_rev)
        beta = torch.flip(torch.tanh(self.beta_dense(h)), dims=[1])  # (B, T, E)
        context = (alpha * (beta * v) * mask).sum(1)
        logit = self.out(self.drop_ctx(context)).squeeze(-1)
        if return_attention:
            return logit, alpha.squeeze(-1), beta
        return logit


def build_model(kind: str, vocab_size: int, maxlen: int, dropout: float, mask_padding: bool = False) -> nn.Module:
    if kind == "transformer":
        return TransformerClassifier(vocab_size, maxlen=maxlen, rate=dropout, mask_padding=mask_padding)
    if kind == "retain":
        return RETAIN(vocab_size, rate=dropout)
    raise ValueError(f"unknown model {kind!r}")
