"""LWC-former: patch-wise dimensionality reduction + transformer encoder + MLP regression head."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


class PatchDimReduction(nn.Module):
    """Patch-based dimensionality reduction (paper eq. 3): H_i = b_i + X_i W_i, own W_i per patch."""

    def __init__(self, n_bands: int, patch_size: int, embed_dim: int, tail: str = "drop"):
        super().__init__()
        if tail not in ("drop", "pad"):
            raise ValueError("tail must be 'drop' or 'pad'")
        if patch_size < 1 or patch_size > n_bands:
            raise ValueError(f"patch_size must be in 1..{n_bands}, got {patch_size}")
        self.n_bands, self.patch_size, self.tail = n_bands, patch_size, tail
        self.n_patches = math.ceil(n_bands / patch_size) if tail == "pad" else n_bands // patch_size
        self.pad = self.n_patches * patch_size - n_bands if tail == "pad" else 0
        self.weight = nn.Parameter(torch.empty(self.n_patches, patch_size, embed_dim))
        self.bias = nn.Parameter(torch.empty(self.n_patches, embed_dim))
        bound = 1.0 / math.sqrt(patch_size)  # Kaiming-uniform (a=sqrt(5)), same as nn.Linear
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tail == "pad":
            x = nn.functional.pad(x, (0, self.pad))
        else:
            x = x[:, : self.n_patches * self.patch_size]
        x = x.reshape(x.shape[0], self.n_patches, self.patch_size)
        return torch.einsum("bnp,npd->bnd", x, self.weight) + self.bias


def sinusoidal_encoding(n_tokens: int, dim: int) -> torch.Tensor:
    pos = torch.arange(n_tokens, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_tokens, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : dim // 2]
    return pe


class LWCFormer(nn.Module):
    def __init__(self, n_bands: int, n_targets: int = 1, patch_size: int = 13, embed_dim: int = 8,
                 n_layers: int = 3, n_heads: int = 2, ff_dim: int = 32, attn_dropout: float = 0.1,
                 head_hidden: Sequence[int] = (64, 64, 32), head_dropout: float = 0.4,
                 pooling: str = "flatten", use_cls: bool = False, tail: str = "drop",
                 return_hidden: bool = False):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})")
        if pooling not in ("flatten", "mean", "cls"):
            raise ValueError("pooling must be flatten, mean or cls")
        if pooling == "cls" and not use_cls:
            raise ValueError("pooling='cls' requires use_cls=true")
        self.return_hidden, self.pooling, self.use_cls = return_hidden, pooling, use_cls
        self.pdr = PatchDimReduction(n_bands, patch_size, embed_dim, tail)
        self.n_patches = self.pdr.n_patches
        self.n_tokens = self.n_patches + (1 if use_cls else 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls else None
        if use_cls:
            nn.init.normal_(self.cls_token, std=0.02)
        self.register_buffer("pos", sinusoidal_encoding(self.n_tokens, embed_dim)[None], persistent=False)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
                                           dropout=attn_dropout, activation="relu", batch_first=True,
                                           norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        feat = self.n_patches * embed_dim if pooling == "flatten" else embed_dim
        layers: List[nn.Module] = []
        prev = feat
        for w in head_hidden:
            layers += [nn.Linear(prev, int(w)), nn.BatchNorm1d(int(w)), nn.ReLU(), nn.Dropout(head_dropout)]
            prev = int(w)
        layers.append(nn.Linear(prev, n_targets))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        h = self.pdr(x)
        if self.use_cls:
            h = torch.cat([self.cls_token.expand(h.shape[0], -1, -1), h], dim=1)
        enc = self.encoder(h + self.pos)
        if self.pooling == "cls":
            z = enc[:, 0]
        elif self.pooling == "mean":
            z = enc.mean(dim=1)
        else:
            z = (enc[:, 1:] if self.use_cls else enc).flatten(1)
        out = self.head(z)
        return (out, enc) if self.return_hidden else out

    def patch_boundaries(self, wavelengths) -> List[Tuple[float, float]]:
        """(first, last) wavelength in nm covered by each patch (CLS excluded)."""
        wl = np.asarray(wavelengths, dtype=float)
        P = self.pdr.patch_size
        return [(float(wl[i * P]), float(wl[min((i + 1) * P, len(wl)) - 1])) for i in range(self.n_patches)]
