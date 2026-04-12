"""3D patch-grid Manhattan distance and per-head additive attention bias (DFormer-style decay)."""
from __future__ import annotations

import torch
import torch.nn as nn


def patch_manhattan_distance_matrix(
    grid_t: int, grid_h: int, grid_w: int, *, device=None, dtype=torch.float32
) -> torch.Tensor:
    """
    Tokens in row-major (t, h, w): index k maps to
    t = k // (H*W), rem = k % (H*W), h = rem // W, w = rem % W.
    Returns D[i,j] = |Δt|+|Δh|+|Δw| with shape (L, L), L = T*H*W.
    """
    L = grid_t * grid_h * grid_w
    idx = torch.arange(L, device=device)
    hw = grid_h * grid_w
    t = idx // hw
    rem = idx % hw
    h = rem // grid_w
    w = rem % grid_w
    pts = torch.stack([t.float(), h.float(), w.float()], dim=1)
    if dtype != pts.dtype:
        pts = pts.to(dtype=dtype)
    d = (pts.unsqueeze(0) - pts.unsqueeze(1)).abs().sum(-1)
    return d


def dformerv2_style_head_decays(num_heads: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Per-head negative scale: log(1 - 2^{-(h+1)}), h = 0..H-1.
    Multiplied by Manhattan distance -> additive bias (more negative for far pairs).
    """
    h = torch.arange(1, num_heads + 1, device=device, dtype=torch.float32)
    v = torch.log(1.0 - torch.pow(2.0, -h))
    if dtype != v.dtype:
        v = v.to(dtype=dtype)
    return v


class GeomBiasTransformerEncoderLayer(nn.Module):
    """
    Pre-LN Transformer encoder layer; self-attention logits get additive bias (B, H, L, L) or (1, H, L, L).
    When attn_bias is None, reduces to standard scaled dot-product attention.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        *,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} not divisible by nhead {nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation != "gelu":
            raise ValueError("Only gelu supported")
        self._act = nn.GELU()

    def _sa_block(self, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        b, l, d = x.shape
        h, dh = self.nhead, self.head_dim
        q = self.q_proj(x).view(b, l, h, dh).transpose(1, 2)
        k = self.k_proj(x).view(b, l, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(b, l, h, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (dh**-0.5)
        if attn_bias is not None:
            scores = scores + attn_bias
        attn = scores.softmax(dim=-1)
        attn = self.dropout1(attn)
        o = attn @ v
        o = o.transpose(1, 2).reshape(b, l, d)
        return self.dropout(self.out_proj(o))

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self._act(self.linear1(x))))
        return self.dropout(x)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_bias)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_bias))
            x = self.norm2(x + self._ff_block(x))
        return x


class GeomBiasTransformerEncoder(nn.Module):
    def __init__(self, layers: list[GeomBiasTransformerEncoderLayer] | nn.ModuleList):
        super().__init__()
        self.layers = layers if isinstance(layers, nn.ModuleList) else nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_bias)
        return x
