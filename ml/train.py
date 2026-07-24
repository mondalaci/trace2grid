#!/usr/bin/env python3
"""Train TinyUNet on prepared dataset with leave-one-out or full fit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TRAIN_LONG_SIDE
from dataset import ToolSegDataset, load_manifest
from model import TinyUNet, count_params

ML_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ML_ROOT / "dataset"
DEFAULT_RUNS = ML_ROOT / "runs"


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (den + eps)
    return 1 - dice.mean()


def bce_dice(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


@torch.no_grad()
def mask_iou(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * targets).sum().item()
    union = (pred + targets).clamp(0, 1).sum().item()
    return 1.0 if union == 0 else inter / union


def train_one_epoch(
    model: TinyUNet,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = bce_dice(logits, y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def eval_loader(model: TinyUNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    ious = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        logits = model(x)
        for i in range(x.size(0)):
            ious.append(mask_iou(logits[i : i + 1], y[i : i + 1]))
    return float(np.mean(ious)) if ious else 0.0


def fit(
    train_ids: list[str],
    val_ids: list[str],
    *,
    data_dir: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    repeats: int,
    long_side: int,
    base: int,
) -> tuple[TinyUNet, dict]:
    train_ds = ToolSegDataset(
        data_dir, train_ids, augment=True, long_side=long_side, repeats=repeats
    )
    val_ds = ToolSegDataset(data_dir, val_ids, augment=False, long_side=long_side, repeats=1)
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = TinyUNet(base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_iou = -1.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, opt, device)
        iou = eval_loader(model, val_loader, device)
        sched.step()
        history.append({"epoch": epoch, "loss": loss, "val_iou": iou})
        if iou >= best_iou:
            best_iou = iou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"  epoch {epoch:03d}/{epochs}  loss={loss:.4f}  "
            f"val_iou={iou * 100:.2f}%  ({time.time() - t0:.1f}s)"
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_val_iou": best_iou, "history": history}


def run_loo(args: argparse.Namespace, device: torch.device, ids: list[str]) -> dict:
    folds = []
    for holdout in ids:
        train_ids = [i for i in ids if i != holdout]
        print(f"\n=== LOO holdout: {holdout} (train {train_ids}) ===")
        model, info = fit(
            train_ids,
            [holdout],
            data_dir=args.data,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            repeats=args.repeats,
            long_side=args.long_side,
            base=args.base,
        )
        folds.append({"holdout": holdout, "val_iou": info["best_val_iou"]})
        out = args.runs / "loo" / holdout
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "base": args.base, "long_side": args.long_side},
            out / "best.pt",
        )
        (out / "metrics.json").write_text(json.dumps(folds[-1], indent=2) + "\n")

    mean = float(np.mean([f["val_iou"] for f in folds]))
    summary = {"mean_loo_iou": mean, "folds": folds}
    print(f"\nLOO mean IoU: {mean * 100:.2f}%")
    return summary


def run_full(args: argparse.Namespace, device: torch.device, ids: list[str]) -> dict:
    print(f"\n=== Full train on {ids} (val = train, monitor overfit) ===")
    model, info = fit(
        ids,
        ids,
        data_dir=args.data,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        repeats=args.repeats,
        long_side=args.long_side,
        base=args.base,
    )
    out = args.runs / "full"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "base": args.base, "long_side": args.long_side},
        out / "best.pt",
    )
    summary = {"train_iou": info["best_val_iou"], "history": info["history"]}
    (out / "metrics.json").write_text(
        json.dumps({"train_iou": summary["train_iou"]}, indent=2) + "\n"
    )
    print(f"Full-fit train IoU: {summary['train_iou'] * 100:.2f}%")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--mode", choices=("loo", "full", "both"), default="both")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--repeats", type=int, default=48, help="Augmented copies per image / epoch")
    ap.add_argument("--long-side", type=int, default=TRAIN_LONG_SIDE)
    ap.add_argument("--base", type=int, default=32, help="UNet base channels")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}  params≈{count_params(TinyUNet(base=args.base)) / 1e6:.2f}M")

    manifest = load_manifest(args.data)
    ids = [m["id"] for m in manifest]
    if len(ids) < 2 and args.mode in ("loo", "both"):
        raise SystemExit("Need ≥2 labeled samples for leave-one-out")

    args.runs.mkdir(parents=True, exist_ok=True)
    result: dict = {"ids": ids, "device": str(device)}
    if args.mode in ("loo", "both"):
        result["loo"] = run_loo(args, device, ids)
    if args.mode in ("full", "both"):
        result["full"] = run_full(args, device, ids)

    (args.runs / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.runs / 'summary.json'}")


if __name__ == "__main__":
    main()
