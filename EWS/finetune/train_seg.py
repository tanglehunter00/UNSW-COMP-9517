"""EWS-Dataset segmentation finetune: MAE encoder + UNet-style decoder. Train/val only; test for final eval."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_FIN = Path(__file__).resolve().parent
_PRETRAIN = _FIN.parent / "pretrain"
SEGMENTATION_TRAIN_LOG_CSV = _FIN / "segmentation_training_log.csv"
sys.path.insert(0, str(_FIN))
sys.path.insert(0, str(_PRETRAIN))

from dataset_ews_seg import EWSSegmentationDataset  # noqa: E402
from model_unet_mae import build_model_from_mae_ckpt  # noqa: E402


def dice_coeff(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """prob/target: (B,1,H,W)."""
    p = prob.reshape(prob.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    inter = (p * t).sum(dim=1)
    return ((2 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)).mean()


def pixel_accuracy(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Per-pixel accuracy at threshold thr; can be high with large background—use with Dice/AUC."""
    pred = (prob >= thr).to(target.dtype)
    return (pred == target).float().mean()


def _binary_roc_auc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pixel ROC-AUC without sklearn (Mann–Whitney form)."""
    y = y_true.astype(np.int64).ravel()
    s = y_score.astype(np.float64).ravel()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_auc_pixels_flat(prob_1d: torch.Tensor, target_1d: torch.Tensor) -> float:
    """ROC-AUC on flattened pixels (positive = foreground)."""
    y = target_1d.detach().cpu().numpy()
    p = prob_1d.detach().cpu().numpy()
    if y.min() == y.max():
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, p))
    except ImportError:
        return _binary_roc_auc_numpy(y, p)


def roc_auc_pixels(prob: torch.Tensor, target: torch.Tensor) -> float:
    return roc_auc_pixels_flat(prob.reshape(-1), target.reshape(-1))


def _csv_float_cell(v: float | None) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{float(v):.6f}"


def reset_segmentation_train_log_csv(path: Path | None = None) -> Path:
    """Overwrite CSV with header only; default path is EWS/finetune/segmentation_training_log.csv."""
    path = path or SEGMENTATION_TRAIN_LOG_CSV
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_dice",
                "val_acc",
                "val_auc",
                "test_loss",
                "test_dice",
                "test_acc",
                "test_auc",
            ]
        )
    return path


def append_segmentation_train_log_epoch(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_dice: float,
    val_acc: float,
    val_auc: float,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                str(epoch),
                _csv_float_cell(train_loss),
                _csv_float_cell(val_loss),
                _csv_float_cell(val_dice),
                _csv_float_cell(val_acc),
                _csv_float_cell(val_auc),
                "",
                "",
                "",
                "",
            ]
        )


def append_segmentation_train_log_test(
    path: Path,
    test_loss: float,
    test_dice: float,
    test_acc: float,
    test_auc: float,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "test",
                "",
                "",
                "",
                "",
                "",
                _csv_float_cell(test_loss),
                _csv_float_cell(test_dice),
                _csv_float_cell(test_acc),
                _csv_float_cell(test_auc),
            ]
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, float, float]:
    """Returns (loss, dice, pixel_acc, roc_auc). AUC over all pixels in the loader."""
    model.eval()
    loss_acc = 0.0
    dice_acc = 0.0
    acc_acc = 0.0
    n = 0
    ys: list[torch.Tensor] = []
    ps: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        prob = torch.sigmoid(logits)
        loss_acc += loss.item() * x.size(0)
        dice_acc += dice_coeff(prob, y).item() * x.size(0)
        acc_acc += pixel_accuracy(prob, y).item() * x.size(0)
        n += x.size(0)
        ys.append(y.reshape(-1))
        ps.append(prob.reshape(-1))
    y_cat = torch.cat(ys, dim=0)
    p_cat = torch.cat(ps, dim=0)
    auc = roc_auc_pixels_flat(p_cat, y_cat)
    return loss_acc / max(n, 1), dice_acc / max(n, 1), acc_acc / max(n, 1), auc


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="EWS-Dataset root (contains train/validation/test)",
    )
    p.add_argument("--mae_ckpt", type=str, required=True, help="mae2d_best.pt or epoch checkpoint")
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Where to save finetuned weights; default next to dataset_root",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--freeze_mae",
        action="store_true",
        help="Freeze MAE encoder; train decoder and shallow CNN only",
    )
    p.add_argument("--img_size", type=int, default=350)
    p.add_argument("--patch_size", type=int, default=25)
    p.add_argument("--amp", action="store_true", help="Mixed precision (CUDA)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--log_csv",
        type=str,
        default="",
        help="Per-epoch + test metrics CSV (default: EWS/finetune/segmentation_training_log.csv)",
    )
    args = p.parse_args()

    repo_ews = Path(__file__).resolve().parent.parent
    dataset_root = Path(args.dataset_root) if args.dataset_root else repo_ews / "data" / "EWS-Dataset"
    out_dir = Path(args.out_dir) if args.out_dir else repo_ews / "data" / "finetune_seg_ckpts"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type == "cpu" and args.device.startswith("cuda"):
        print("Warning: CUDA not available, using CPU")

    ds_tr = EWSSegmentationDataset(dataset_root, "train", img_size=args.img_size)
    ds_va = EWSSegmentationDataset(dataset_root, "val", img_size=args.img_size)
    ds_te = EWSSegmentationDataset(dataset_root, "test", img_size=args.img_size)

    pin = device.type == "cuda"
    loader_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    loader_te = DataLoader(
        ds_te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = build_model_from_mae_ckpt(
        args.mae_ckpt,
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=1,
        freeze_mae=args.freeze_mae,
    ).to(device)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    log_csv = Path(args.log_csv) if args.log_csv else SEGMENTATION_TRAIN_LOG_CSV
    reset_segmentation_train_log_csv(log_csv)
    print(f"Metrics CSV: {log_csv.resolve()}")

    best_dice = -1.0
    best_path = out_dir / "unet_mae_best.pt"
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, loader_tr, opt, device, scaler)
        va_loss, va_dice, va_acc, va_auc = evaluate(model, loader_va, device)
        append_segmentation_train_log_epoch(
            log_csv, epoch, tr_loss, va_loss, va_dice, va_acc, va_auc
        )
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  "
            f"val_loss={va_loss:.4f}  val_dice={va_dice:.4f}  val_acc={va_acc:.4f}  val_auc={va_auc:.4f}"
        )
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "val_dice": va_dice,
                "val_acc": va_acc,
                "val_auc": va_auc,
            },
            out_dir / f"unet_mae_epoch{epoch}.pt",
        )
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "val_dice": va_dice,
                    "val_acc": va_acc,
                    "val_auc": va_auc,
                },
                best_path,
            )

    print("Evaluating best val checkpoint on test (test used only here)...")
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_dice, te_acc, te_auc = evaluate(model, loader_te, device)
    append_segmentation_train_log_test(log_csv, te_loss, te_dice, te_acc, te_auc)
    print(
        f"test_loss={te_loss:.4f}  test_dice={te_dice:.4f}  test_acc={te_acc:.4f}  test_auc={te_auc:.4f}"
    )
    with open(out_dir / "test_metrics.txt", "w", encoding="utf-8") as f:
        f.write(
            f"test_loss={te_loss}\ntest_dice={te_dice}\ntest_acc={te_acc}\ntest_auc={te_auc}\n"
        )


if __name__ == "__main__":
    main()
