# Neural segmentation (train on GPU, infer in browser via ONNX)

Two TinyUNet models, both trained from `training/` labels and run
client-side with `onnxruntime-web` (WebGPU):

| Model | Input | Label source | Browser |
| --- | --- | --- | --- |
| `toolseg.onnx` | Rectified paper | Tool outlines (`truth`) | `extractToolContoursNn` |
| `paperseg.onnx` | Full photo | Paper corners polygon | `detectPaperQuadNn` |

Capture uses both by default. Classical OpenCV Lab paper detection remains as
a fallback if `paperseg.onnx` is missing; classical tool contours stay on
`eval.html` for A/B.

## npm shortcuts (from repo root)

Prefer these over raw Python once `ml/.venv` exists:

```bash
npm run ml:prepare       # rebuild tool + paper datasets from training/
npm run ml:train:quick   # LOO+full for both tasks (~20 min each)
npm run ml:train         # LOO+full for both (~45–60 min each)
npm run ml:export        # ONNX → ml/export/ and public/models/
npm run ml:quick         # prepare + train-quick + export
npm run ml               # prepare + train + export
```

Same thing via `ml/run.sh <prepare|train|train-quick|export|all|all-quick>`.
Extra `train.py` flags pass through, e.g. `npm run ml:train -- --epochs 20`.
Train a single task with `python train.py --task paper …`.

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

```bash
python prepare_dataset.py --task both
# tools → ml/dataset/<id>/{image,mask}.png   (rectified RGB + tool masks)
# paper → ml/dataset_paper/<id>/{image,mask}.png  (full photo + corner-quad mask)
```

Tool masks drop degenerate truth polys under 20 mm².

## Train

Leave-one-out (honest score with n=4) and a full overfit fit:

```bash
python train.py --task tools --mode both --epochs 80 --repeats 48
python train.py --task paper --mode both --epochs 80 --repeats 48
# faster smoke test:
python train.py --task paper --mode loo --epochs 20 --repeats 16
```

Artifacts:

- tools: `ml/runs/loo/<holdout>/best.pt`, `ml/runs/full/best.pt`
- paper: `ml/runs/paper/loo/…`, `ml/runs/paper/full/best.pt`

With only 4 images, **LOO mean IoU** is the number to trust; full-fit train IoU
will look almost perfect (memorization).

## Export ONNX

```bash
python export_onnx.py --task both
# → ml/export/toolseg.onnx + paperseg.onnx (+ .json metadata)
```

Copy both into `public/models/` (`npm run ml:export` does this). The `.onnx`
weights are committed so capture works after a fresh clone; re-export and
commit after retraining.

## Expected client performance

| Runtime | Latency @ 768² |
| --- | --- |
| WebGPU (desktop) | ~15–60 ms |
| WASM CPU | ~0.3–1.5 s |

Model is TinyUNet (`--base 32`, ~8 M params, ~30 MB FP32 ONNX each).

## First run on the 4 labeled photos (RTX 5080, tools)

| Metric | IoU |
| --- | --- |
| Leave-one-out mean | **79.8%** |
| LOO knifes / nose-plier / precision / red | 78.3% / 58.5% / 88.3% / 94.2% |
| Full-fit (train=val, overfit monitor) | **91.3%** |

LOO below classical tool scores is expected with n=4 — the held-out photo is a
different tool family. Add more labels, then re-run; trust LOO, not full-fit.
