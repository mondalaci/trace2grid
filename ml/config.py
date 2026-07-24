"""Shared constants matching the Trace2Grid training / eval pipeline."""

from __future__ import annotations

PAPER_SIZES_MM = {
    "a4": (210.0, 297.0),
    "letter": (215.9, 279.4),
    "legal": (215.9, 355.6),
}

# Same scale as src/train/evalRunner.ts
PX_PER_MM = 4.0
MAX_PHOTO_DIM = 1800
# Drop degenerate truth polys (scribbles) below this area.
MIN_TRUTH_AREA_MM2 = 20.0

# Inference long side used in training / ONNX (full rectified is ~840×1188).
TRAIN_LONG_SIDE = 768
