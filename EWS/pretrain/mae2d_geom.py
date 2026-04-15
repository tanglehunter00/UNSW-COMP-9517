"""2D MAE with optional geometric prior bias on self-attention (patch-grid Manhattan × per-head decay)."""
from __future__ import annotations

import torch
import torch.nn as nn

from geom_attn_bias import (
    GeomBiasTransformerEncoder,
    GeomBiasTransformerEncoderLayer,
    dformerv2_style_head_decays,
    patch_manhattan_distance_matrix,
)
from mae2d import _trunc_normal_


class MaskedAutoencoder2DGeom(nn.Module):
    """
    Same outer API as MaskedAutoencoder2D, but encoder/decoder use GeomBiasTransformerEncoder
    when use_geom_bias=True. Additive bias: attn_logits += dist * decay_h (per head).

    Single-frame2D images: grid_t=1; distance reduces to |Δh|+|Δw| on the patch grid.
    """

    def __init__(
        self,
        img_size: int = 350,
        patch_size: int = 25,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        *,
        grid_t: int = 1,
        use_geom_bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.grid = img_size // patch_size
        if grid_t != 1:
            raise ValueError("MaskedAutoencoder2DGeom currently supports grid_t=1 (single-frame 2D patches).")
        self.grid_t = grid_t
        self.grid_h = self.grid
        self.grid_w = self.grid
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = patch_size * patch_size * in_chans
        self.embed_dim = embed_dim
        self.use_geom_bias = use_geom_bias

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        enc_layers = [
            GeomBiasTransformerEncoderLayer(
                embed_dim,
                num_heads,
                int(embed_dim * mlp_ratio),
                dropout=dropout,
                norm_first=True,
            )
            for _ in range(depth)
        ]
        self.encoder = GeomBiasTransformerEncoder(enc_layers)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))

        dec_layers = [
            GeomBiasTransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(decoder_embed_dim * mlp_ratio),
                dropout=dropout,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ]
        self.decoder = GeomBiasTransformerEncoder(dec_layers)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)

        D = patch_manhattan_distance_matrix(grid_t, self.grid_h, self.grid_w)
        self.register_buffer("patch_dist", D)
        self.register_buffer("enc_head_decay", dformerv2_style_head_decays(num_heads))
        self.register_buffer("dec_head_decay", dformerv2_style_head_decays(decoder_num_heads))

        dec_bias = D.unsqueeze(0) * self.dec_head_decay.view(decoder_num_heads, 1, 1)
        self.register_buffer("_dec_bias_full", dec_bias.unsqueeze(0))

        _trunc_normal_(self.pos_embed, std=0.02)
        _trunc_normal_(self.decoder_pos_embed, std=0.02)
        _trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        g = self.grid
        x = imgs.reshape(imgs.shape[0], self.in_chans, g, p, g, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], self.num_patches, self.patch_dim)

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, d = x.shape
        keep = int(l * (1 - mask_ratio))
        noise = torch.rand(b, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))
        mask = torch.ones(b, l, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_visible, mask, ids_restore, ids_keep

    def _encoder_attn_bias(self, ids_keep: torch.Tensor) -> torch.Tensor | None:
        if not self.use_geom_bias:
            return None
        # ids_keep: (B, Lv); patch_dist: (L, L)
        d_sub = self.patch_dist[ids_keep.unsqueeze(-1), ids_keep.unsqueeze(-2)].float()
        h = self.enc_head_decay.numel()
        return d_sub.unsqueeze(1) * self.enc_head_decay.view(1, h, 1, 1)

    def _decoder_attn_bias(self) -> torch.Tensor | None:
        if not self.use_geom_bias:
            return None
        return self._dec_bias_full.float()

    def encode_all_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        b, l, _ = x.shape
        idx = torch.arange(l, device=x.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        bias = self._encoder_attn_bias(idx)
        return self.encoder(x, bias)

    def encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        t = self.encode_all_tokens(x)
        b, _l, d = t.shape
        g = self.grid
        return t.transpose(1, 2).reshape(b, d, g, g)

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x_vis, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        bias = self._encoder_attn_bias(ids_keep)
        x_vis = self.encoder(x_vis, bias)
        return x_vis, mask, ids_restore

    def forward_decoder(self, x_vis: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x_vis)
        b, l_full = ids_restore.shape[0], ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(b, l_full - x.shape[1], x.shape[2])
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2]).long()
        )
        x_full = x_full + self.decoder_pos_embed
        x_full = self.decoder(x_full, self._decoder_attn_bias())
        return self.decoder_pred(x_full)

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.5):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
