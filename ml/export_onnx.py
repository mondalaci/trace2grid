#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX for onnxruntime-web."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import TRAIN_LONG_SIDE
from model import TinyUNet

ML_ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ML_ROOT / "runs" / "full" / "best.pt",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ML_ROOT / "export" / "toolseg.onnx",
    )
    ap.add_argument("--long-side", type=int, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = int(ckpt.get("base", 32))
    long_side = int(args.long_side or ckpt.get("long_side", TRAIN_LONG_SIDE))
    model = TinyUNet(base=base)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, 3, long_side, long_side)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Legacy exporter emits a single .onnx (easier for onnxruntime-web fetch).
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    meta = args.out.with_suffix(".json")
    meta.write_text(
        "{\n"
        f'  "longSide": {long_side},\n'
        f'  "base": {base},\n'
        f'  "input": "image",\n'
        f'  "output": "logits",\n'
        f'  "normalize": "rgb_to_[-1,1]",\n'
        f'  "checkpoint": "{args.checkpoint.as_posix()}"\n'
        "}\n"
    )
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    print(f"Wrote {meta}")


if __name__ == "__main__":
    main()
