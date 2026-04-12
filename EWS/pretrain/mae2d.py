"""Native 2D Masked Autoencoder (ViT-style patches) for self-supervised RGB (3,H,W) patches."""
from __future__ import annotations

import torch
import torch.nn as nn


def _trunc_normal_(t: torch.Tensor, std: float = 0.02) -> None:
    nn.init.normal_(t, std=std)
    with torch.no_grad():
        t.clamp_(-2 * std, 2 * std)


class MaskedAutoencoder2D(nn.Module):
    """
    imgs: (B, 3, H, W); H and W must be divisible by patch_size.
    mask_ratio fraction of patches are masked; reconstruction MSE is on masked patches only.
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
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_dim = patch_size * patch_size * in_chans
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))

        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_depth)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)

        _trunc_normal_(self.pos_embed, std=0.02)
        _trunc_normal_(self.decoder_pos_embed, std=0.02)
        _trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, L, patch_dim)"""
        p = self.patch_size
        g = self.grid
        x = imgs.reshape(imgs.shape[0], self.in_chans, g, p, g, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], self.num_patches, self.patch_dim)

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, L, D). Returns visible tokens, mask (B,L) with 1=masked, ids_restore."""
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
        return x_visible, mask, ids_restore

    def encode_all_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Downstream use after pretrain: no random masking; all patch tokens (B, L, embed_dim)."""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return self.encoder(x)

    def encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, embed_dim, grid, grid) for 2D heads (e.g. segmentation)."""
        t = self.encode_all_tokens(x)
        b, _l, d = t.shape
        g = self.grid
        return t.transpose(1, 2).reshape(b, d, g, g)

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x_vis, mask, ids_restore = self.random_masking(x, mask_ratio)
        x_vis = self.encoder(x_vis)
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
        x_full = self.decoder(x_full)
        return self.decoder_pred(x_full)

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
