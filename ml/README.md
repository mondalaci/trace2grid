# Neural tool segmentation (train on GPU, infer in browser via ONNX)

Train a small U-Net on labeled `training/` photos, export ONNX, and (next)
run it client-side with `onnxruntime-web` (WebGPU).

## npm shortcuts (from repo root)

Prefer these over raw Python once `ml/.venv` exists:

```bash
npm run ml:prepare       # rebuild dataset from training/
npm run ml:train:quick   # LOO+full, 40 ep / 32 repeats (~20 min)
npm run ml:train         # LOO+full, 80 ep / 48 repeats (~45–60 min)
npm run ml:export        # ONNX → ml/export/ and public/models/
npm run ml:quick         # prepare + train-quick + export
npm run ml               # prepare + train + export
```

Same thing via `ml/run.sh <prepare|train|train-quick|export|all|all-quick>`.
Extra `train.py` flags pass through, e.g. `npm run ml:train -- --epochs 20`.

## Setup (RTX / CUDA)

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
# Pick the CUDA wheel matching your driver: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Prepare dataset

Rectifies each labeled photo (same corners / `pxPerMm=4` as the app) and
writes `image.png` + `mask.png`. Degenerate truth polys under 20 mm² are dropped.

```bash
python prepare_dataset.py
# → ml/dataset/<id>/{image,mask}.png + manifest.json
```

## Train

Leave-one-out (honest score with n=4) and a full overfit fit:

```bash
python train.py --mode both --epochs 80 --repeats 48
# faster smoke test:
python train.py --mode loo --epochs 20 --repeats 16
```

Artifacts: `ml/runs/loo/<holdout>/best.pt`, `ml/runs/full/best.pt`, `ml/runs/summary.json`.

With only 4 images, **LOO mean IoU** is the number to trust; full-fit train IoU
will look almost perfect (memorization).

## Export ONNX

```bash
python export_onnx.py --checkpoint runs/full/best.pt
# → ml/export/toolseg.onnx (+ toolseg.json metadata)
```

Copy `toolseg.onnx` (+ `toolseg.json`) into `public/models/` for the browser
loader in `src/lib/scan/nnSeg.ts` (`segmentToolsNn`). ONNX weights are
gitignored (large); rebuild with `npm run ml:export` after training.

## Expected client performance

| Runtime | Latency @ 768² |
| --- | --- |
| WebGPU (desktop) | ~15–60 ms |
| WASM CPU | ~0.3–1.5 s |

Model is TinyUNet (`--base 32`, ~8 M params, ~30 MB FP32 ONNX).

## First run on the 4 labeled photos (RTX 5080)

| Metric | IoU |
| --- | --- |
| Leave-one-out mean | **79.8%** |
| LOO knifes / nose-plier / precision / red | 78.3% / 58.5% / 88.3% / 94.2% |
| Full-fit (train=val, overfit monitor) | **91.3%** |

LOO below the classical ~83% mean is expected with n=4 — the held-out photo is a different tool family. Add more labels, then re-run `train.py`; trust LOO, not full-fit.
