"""MAE (geometric-attention backbone) + UNet-style decoder for EWS segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mae2d_geom import MaskedAutoencoder2DGeom


class _ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        g = min(8, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class _MergeUp(nn.Module):
    """Upsample to skip spatial size, concat, conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = _ConvGNAct(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MAEGeomEncoderUNetDecoder(nn.Module):
    """
    Encoder: MaskedAutoencoder2DGeom (patch_embed + geom-bias Transformer) → (B, D, g, g).
    Same UNet-style decoder as MAEEncoderUNetDecoder.
    """

    def __init__(
        self,
        mae: MaskedAutoencoder2DGeom,
        *,
        num_classes: int = 1,
        base_ch: int = 48,
    ):
        super().__init__()
        self.mae = mae
        C = base_ch

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, C), C),
            nn.GELU(),
        )
        self.d1 = nn.Sequential(nn.MaxPool2d(2), _ConvGNAct(C, C * 2))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), _ConvGNAct(C * 2, C * 4))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), _ConvGNAct(C * 4, C * 4))
        self.d4 = nn.Sequential(nn.MaxPool2d(2), _ConvGNAct(C * 4, C * 4))

        self.fuse = _ConvGNAct(mae.embed_dim + C * 4, C * 4)
        self.m3 = _MergeUp(C * 4, C * 4, C * 4)
        self.m2 = _MergeUp(C * 4, C * 4, C * 2)
        self.m1 = _MergeUp(C * 2, C * 2, C)
        self.m0 = _MergeUp(C, C, max(C // 2, 8))
        self.head = nn.Conv2d(max(C // 2, 8), num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, _c, h, w = x.shape
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)

        z = self.mae.encode_spatial(x)
        s4p = F.adaptive_avg_pool2d(s4, output_size=z.shape[-2:])
        u = self.fuse(torch.cat([z, s4p], dim=1))

        u = self.m3(u, s3)
        u = self.m2(u, s2)
        u = self.m1(u, s1)
        u = self.m0(u, s0)
        if u.shape[-2:] != (h, w):
            u = F.interpolate(u, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(u)


def build_model_from_mae_ckpt(
    ckpt_path: str,
    *,
    img_size: int = 350,
    patch_size: int = 25,
    num_classes: int = 1,
    freeze_mae: bool = False,
) -> MAEGeomEncoderUNetDecoder:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    use_geom = bool(ckpt.get("use_geom_bias", True))
    mae = MaskedAutoencoder2DGeom(
        img_size=img_size,
        patch_size=patch_size,
        use_geom_bias=use_geom,
    )
    mae.load_state_dict(ckpt["model"], strict=True)
    model = MAEGeomEncoderUNetDecoder(mae, num_classes=num_classes)
    if freeze_mae:
        for p in model.mae.parameters():
            p.requires_grad = False
    return model


def build_model_random_mae(
    *,
    img_size: int = 350,
    patch_size: int = 25,
    num_classes: int = 1,
    freeze_mae: bool = False,
    use_geom_bias: bool = True,
) -> MAEGeomEncoderUNetDecoder:
    mae = MaskedAutoencoder2DGeom(
        img_size=img_size,
        patch_size=patch_size,
        use_geom_bias=use_geom_bias,
    )
    model = MAEGeomEncoderUNetDecoder(mae, num_classes=num_classes)
    if freeze_mae:
        for p in model.mae.parameters():
            p.requires_grad = False
    return model
