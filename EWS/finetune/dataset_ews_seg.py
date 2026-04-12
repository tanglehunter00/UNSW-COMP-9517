"""EWS-Dataset semantic segmentation: train / validation / test folders, paired RGB + _mask.png."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SPLIT_DIRS = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


def _list_image_stems(split_dir: Path) -> list[str]:
    stems: list[str] = []
    for p in sorted(split_dir.glob("*.png")):
        name = p.name
        if name.endswith("_mask.png"):
            continue
        mask_path = p.with_name(p.stem + "_mask.png")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask file: {mask_path}")
        stems.append(p.stem)
    if not stems:
        raise FileNotFoundError(f"No RGB images found in: {split_dir}")
    return stems


def load_binary_mask(mask_path: Path) -> torch.Tensor:
    """PIL LA/RGBA: use L channel, foreground where L > 0."""
    im = Image.open(mask_path)
    if im.mode not in ("L", "LA", "RGBA", "RGB"):
        im = im.convert("RGBA")
    arr = np.array(im)
    if arr.ndim == 2:
        L = arr
    else:
        L = arr[..., 0]
    fg = (L > 0).astype(np.float32)
    return torch.from_numpy(fg).unsqueeze(0)


class EWSSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        *,
        img_size: int | None = 350,
    ):
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(SPLIT_DIRS)}, got {split!r}")
        root = Path(dataset_root)
        split_dir = root / SPLIT_DIRS[split]
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
        self.split_dir = split_dir
        self.stems = _list_image_stems(split_dir)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        img_path = self.split_dir / f"{stem}.png"
        mask_path = self.split_dir / f"{stem}_mask.png"
        img = Image.open(img_path).convert("RGB")
        if self.img_size is not None and img.size != (self.img_size, self.img_size):
            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:
                resample = Image.BILINEAR
            img = img.resize((self.img_size, self.img_size), resample)
        arr = np.array(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        m = load_binary_mask(mask_path)
        if self.img_size is not None and m.shape[-1] != self.img_size:
            m = torch.nn.functional.interpolate(
                m.unsqueeze(0), size=(self.img_size, self.img_size), mode="nearest"
            ).squeeze(0)
        return x, m
