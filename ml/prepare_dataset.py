#!/usr/bin/env python3
"""Build rectified RGB + binary masks from training/<photo>.jpg(+.json)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from config import (
    MAX_PHOTO_DIM,
    MIN_TRUTH_AREA_MM2,
    PAPER_SIZES_MM,
    PX_PER_MM,
)

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
OUT_DEFAULT = Path(__file__).resolve().parent / "dataset"


def shoelace_mm2(ring: list[list[float]]) -> float:
    if len(ring) < 3:
        return 0.0
    a = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def load_photo_scaled(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    scale = min(1.0, MAX_PHOTO_DIM / max(w, h))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return bgr


def rectify(bgr: np.ndarray, ann: dict) -> tuple[np.ndarray, float, float]:
    paper_id = ann["paperSizeId"]
    w_mm, h_mm = PAPER_SIZES_MM[paper_id]
    if ann.get("landscape"):
        w_mm, h_mm = h_mm, w_mm
    out_w = int(round(w_mm * PX_PER_MM))
    out_h = int(round(h_mm * PX_PER_MM))
    ph, pw = bgr.shape[:2]
    corners = np.array(
        [[c[0] * pw, c[1] * ph] for c in ann["corners"]],
        dtype=np.float32,
    )
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(
        bgr,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, w_mm, h_mm


def rasterize_truth(ann: dict, out_w: int, out_h: int) -> np.ndarray:
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    for poly in ann.get("truth") or []:
        outer = poly.get("outerMm") or []
        if shoelace_mm2(outer) < MIN_TRUTH_AREA_MM2:
            continue
        rings = [outer, *(poly.get("holesMm") or [])]
        for i, ring in enumerate(rings):
            if len(ring) < 3:
                continue
            pts = np.array(
                [[int(round(x * PX_PER_MM)), int(round(y * PX_PER_MM))] for x, y in ring],
                dtype=np.int32,
            )
            color = 255 if i == 0 else 0
            cv2.fillPoly(mask, [pts], color)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    photos = sorted(TRAINING.glob("*.jpg"))
    if not photos:
        raise SystemExit(f"No photos in {TRAINING}")

    meta = []
    for photo in photos:
        ann_path = Path(str(photo) + ".json")
        if not ann_path.exists():
            print(f"skip {photo.name}: no annotation")
            continue
        ann = json.loads(ann_path.read_text())
        if not ann.get("truth") or not ann.get("corners"):
            print(f"skip {photo.name}: incomplete annotation")
            continue
        bgr = load_photo_scaled(photo)
        rgb_rect, w_mm, h_mm = rectify(bgr, ann)
        mask = rasterize_truth(ann, rgb_rect.shape[1], rgb_rect.shape[0])
        stem = photo.stem
        sample_dir = args.out / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / "image.png"), rgb_rect)
        cv2.imwrite(str(sample_dir / "mask.png"), mask)
        fg = int((mask > 127).sum())
        entry = {
            "id": stem,
            "photo": photo.name,
            "widthPx": int(rgb_rect.shape[1]),
            "heightPx": int(rgb_rect.shape[0]),
            "widthMm": w_mm,
            "heightMm": h_mm,
            "fgPixels": fg,
            "fgMm2": fg / (PX_PER_MM * PX_PER_MM),
        }
        meta.append(entry)
        print(
            f"{stem}: {entry['widthPx']}x{entry['heightPx']} "
            f"fg={entry['fgMm2']:.0f} mm²"
        )

    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(meta)} samples → {args.out}")


if __name__ == "__main__":
    main()
