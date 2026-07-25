#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX for onnxruntime-web."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import TRAIN_LONG_SIDE
from model import TinyUNet

ML_ROOT = Path(__file__).resolve().parent
TASK_DEFAULTS = {
    "tools": {
        "checkpoint": ML_ROOT / "runs" / "full" / "best.pt",
        "out": ML_ROOT / "export" / "toolseg.onnx",
    },
    "paper": {
        "checkpoint": ML_ROOT / "runs" / "paper" / "full" / "best.pt",
        "out": ML_ROOT / "export" / "paperseg.onnx",
    },
}


def export_one(checkpoint: Path, out: Path, long_side: int | None, opset: int) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    base = int(ckpt.get("base", 32))
    side = int(long_side or ckpt.get("long_side", TRAIN_LONG_SIDE))
    model = TinyUNet(base=base)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, 3, side, side)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Legacy exporter emits a single .onnx (easier for onnxruntime-web fetch).
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset,
        dynamo=False,
    )
    meta = out.with_suffix(".json")
    meta.write_text(
        "{\n"
        f'  "longSide": {side},\n'
        f'  "base": {base},\n'
        f'  "input": "image",\n'
        f'  "output": "logits",\n'
        f'  "normalize": "rgb_to_[-1,1]",\n'
        f'  "checkpoint": "{checkpoint.as_posix()}"\n'
        "}\n"
    )
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"Wrote {meta}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("tools", "paper", "both"), default="both")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--long-side", type=int, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    tasks = ("tools", "paper") if args.task == "both" else (args.task,)
    for task in tasks:
        defaults = TASK_DEFAULTS[task]
        checkpoint = args.checkpoint if args.checkpoint and args.task != "both" else defaults["checkpoint"]
        out = args.out if args.out and args.task != "both" else defaults["out"]
        if args.task == "both" and (args.checkpoint or args.out):
            # Per-task defaults when exporting both; ignore single-path overrides.
            checkpoint = defaults["checkpoint"]
            out = defaults["out"]
        export_one(checkpoint, out, args.long_side, args.opset)


if __name__ == "__main__":
    main()
