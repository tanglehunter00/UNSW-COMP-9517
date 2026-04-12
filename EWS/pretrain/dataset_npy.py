"""Load `.npy` patches from a text list (one path per line); returns 2D RGB tensors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_paths(list_path: str | Path) -> list[str]:
    p = Path(list_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


class NpyPatchListDataset(Dataset):
    """One `.npy` path per line; array float32 (H,W,3) ~[0,1], returns (3,H,W)."""

    def __init__(
        self,
        list_path: str | Path,
        *,
        split: str = "train",
        val_every: int = 500,
    ):
        paths = read_paths(list_path)
        if not paths:
            raise FileNotFoundError(f"Empty list file: {list_path}")
        n = len(paths)
        if split == "train":
            self.paths = [paths[i] for i in range(n) if i % val_every != 0]
        else:
            self.paths = [paths[i] for i in range(n) if i % val_every == 0]
        if not self.paths:
            raise RuntimeError(
                f"Empty split after filtering: split={split}, total={n}. Try smaller val_every (now {val_every})."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = np.load(self.paths[idx])
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"{self.paths[idx]}: expected (H,W,3), got {arr.shape}")
        # (H,W,C) -> (C,H,W)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
