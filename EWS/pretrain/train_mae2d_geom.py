from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_npy import NpyPatchListDataset
from mae2d_geom import MaskedAutoencoder2DGeom
_CUDA_INSTALL_HINT = 'GPU training required but torch.cuda.is_available() is False.\nCommon causes: 1) CPU-only PyTorch (default pip install torch); 2) Driver / CUDA runtime issues.\nInstall a CUDA build of PyTorch, e.g.:\n  RTX 50 / Blackwell (cu128): python -m pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128\n  Other GPUs: python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124\nMatch CUDA to your driver: https://pytorch.org/get-started/locally/\nFor CPU debugging only, pass device="cpu" or --device cpu.'

def resolve_device(device_str: str) -> torch.device:
    s = (device_str or '').strip().lower()
    if s == 'cpu':
        return torch.device('cpu')
    if s:
        dev = torch.device(device_str)
        if dev.type == 'cuda' and (not torch.cuda.is_available()):
            raise RuntimeError(f'device={device_str!r} requested but CUDA is not available.\n{_CUDA_INSTALL_HINT}')
        return dev
    if torch.cuda.is_available():
        return torch.device('cuda')
    print("Warning: CUDA ， CPU （）。  GPU， CUDA  PyTorch； device='cpu' 。")
    return torch.device('cpu')

def train_one_epoch(model: MaskedAutoencoder2DGeom, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, mask_ratio: float) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc='train', leave=False):
        batch = batch.to(device, non_blocking=device.type == 'cuda')
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model(batch, mask_ratio=mask_ratio)
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.size(0)
        n += batch.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(model: MaskedAutoencoder2DGeom, loader: DataLoader, device: torch.device, mask_ratio: float) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == 'cuda')
        loss, _, _ = model(batch, mask_ratio=mask_ratio)
        total += loss.item() * batch.size(0)
        n += batch.size(0)
    return total / max(n, 1)

def parse_args():
    p = argparse.ArgumentParser(description='EWS 2D MAE pretraining (geometric attention bias)')
    p.add_argument('--list_path', type=str, required=True, help='all_patches.txt')
    p.add_argument('--ckpt_dir', type=str, required=True)
    p.add_argument('--img_size', type=int, default=350)
    p.add_argument('--patch_size', type=int, default=25)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.00015)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--mask_ratio', type=float, default=0.5)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--val_every', type=int, default=500)
    p.add_argument('--no_geom_bias', action='store_true', help='Disable distance-based attention bias (ablation; same stack as vanilla MHA).')
    p.add_argument('--device', type=str, default='', help='Empty: CUDA if available else CPU. Pass cpu to force CPU; cuda:0 to force GPU.')
    return p.parse_args()

def run_training(*, list_path: str, ckpt_dir: str, img_size: int=350, patch_size: int=25, epochs: int=10, batch_size: int=8, lr: float=0.00015, weight_decay: float=0.05, mask_ratio: float=0.5, num_workers: int=0, val_every: int=500, device: str='', use_geom_bias: bool=True) -> None:

    class A:
        pass
    args = A()
    args.list_path = list_path
    args.ckpt_dir = ckpt_dir
    args.img_size = img_size
    args.patch_size = patch_size
    args.epochs = epochs
    args.batch_size = batch_size
    args.lr = lr
    args.weight_decay = weight_decay
    args.mask_ratio = mask_ratio
    args.num_workers = num_workers
    args.val_every = val_every
    args.device = device
    args.use_geom_bias = use_geom_bias
    args.no_geom_bias = not use_geom_bias
    _run_from_args(args)

def _run_from_args(args):
    device = resolve_device(args.device)
    if device.type == 'cuda':
        print(f'Training device: {device} ({torch.cuda.get_device_name(0)})')
    else:
        print(f'Training device: {device} (CPU)')
    use_geom = not getattr(args, 'no_geom_bias', False)
    print(f'Geometric attention bias: {use_geom}')
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ds_tr = NpyPatchListDataset(args.list_path, split='train', val_every=args.val_every)
    ds_va = NpyPatchListDataset(args.list_path, split='val', val_every=args.val_every)
    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    model = MaskedAutoencoder2DGeom(img_size=args.img_size, patch_size=args.patch_size, use_geom_bias=use_geom).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, loader_tr, optimizer, device, args.mask_ratio)
        va = eval_epoch(model, loader_va, device, args.mask_ratio)
        print(f'epoch {epoch}/{args.epochs}  train_loss={tr:.6f}  val_loss={va:.6f}')
        path = ckpt_dir / f'mae2d_geom_epoch{epoch}.pt'
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args), 'arch': 'MaskedAutoencoder2DGeom', 'use_geom_bias': use_geom}, path)
        if va < best:
            best = va
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args), 'arch': 'MaskedAutoencoder2DGeom', 'use_geom_bias': use_geom}, ckpt_dir / 'mae2d_geom_best.pt')

def main():
    args = parse_args()
    args.use_geom_bias = not args.no_geom_bias
    _run_from_args(args)
if __name__ == '__main__':
    main()
