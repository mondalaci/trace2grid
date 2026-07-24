/**
 * Neural tool segmentation via ONNX Runtime Web (WebGPU / WASM).
 *
 * Train/export: see ml/README.md. Place the ONNX file at
 * `public/models/toolseg.onnx` (and optional toolseg.json).
 *
 * Not wired into CaptureView yet — call `segmentToolsNn` after paper
 * rectification when you want to A/B against the OpenCV contour path.
 */

import type { InferenceSession, Tensor } from 'onnxruntime-web'

export interface NnSegMeta {
  longSide: number
  normalize: 'rgb_to_[-1,1]'
}

export interface NnSegResult {
  /** Binary mask, 0/255, same size as the rectified canvas. */
  mask: Uint8Array
  width: number
  height: number
  inferenceMs: number
}

const DEFAULT_MODEL_URL = '/models/toolseg.onnx'
const DEFAULT_META: NnSegMeta = { longSide: 768, normalize: 'rgb_to_[-1,1]' }

let sessionPromise: Promise<InferenceSession> | null = null
let meta: NnSegMeta = DEFAULT_META

async function loadOrt() {
  return import('onnxruntime-web')
}

/** Lazily create an ORT session (WebGPU if available, else WASM). */
export function loadToolSegSession(modelUrl = DEFAULT_MODEL_URL): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      try {
        const res = await fetch(modelUrl.replace(/\.onnx$/, '.json'))
        if (res.ok) meta = { ...DEFAULT_META, ...(await res.json()) }
      } catch {
        /* keep defaults */
      }
      const ort = await loadOrt()
      return ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu', 'wasm'],
      })
    })()
  }
  return sessionPromise
}

function letterboxRgb(
  source: HTMLCanvasElement,
  longSide: number,
): { tensor: Float32Array; pad: { top: number; left: number; scale: number; nh: number; nw: number } } {
  const sw = source.width
  const sh = source.height
  const scale = longSide / Math.max(sw, sh)
  const nw = Math.round(sw * scale)
  const nh = Math.round(sh * scale)
  const top = Math.floor((longSide - nh) / 2)
  const left = Math.floor((longSide - nw) / 2)

  const canvas = document.createElement('canvas')
  canvas.width = longSide
  canvas.height = longSide
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, longSide, longSide)
  ctx.drawImage(source, left, top, nw, nh)
  const { data } = ctx.getImageData(0, 0, longSide, longSide)

  const tensor = new Float32Array(3 * longSide * longSide)
  const plane = longSide * longSide
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    tensor[i] = (data[p] / 255 - 0.5) / 0.5
    tensor[plane + i] = (data[p + 1] / 255 - 0.5) / 0.5
    tensor[2 * plane + i] = (data[p + 2] / 255 - 0.5) / 0.5
  }
  return { tensor, pad: { top, left, scale, nh, nw } }
}

/**
 * Run TinyUNet ONNX on a rectified paper canvas; returns a full-resolution mask.
 */
export async function segmentToolsNn(
  rectified: HTMLCanvasElement,
  options?: { modelUrl?: string; threshold?: number },
): Promise<NnSegResult> {
  const ort = await loadOrt()
  const session = await loadToolSegSession(options?.modelUrl)
  const longSide = meta.longSide
  const { tensor, pad } = letterboxRgb(rectified, longSide)
  const input = new ort.Tensor('float32', tensor, [1, 3, longSide, longSide])

  const t0 = performance.now()
  const out = await session.run({ image: input })
  const inferenceMs = performance.now() - t0

  const logits = (out.logits ?? Object.values(out)[0]) as Tensor
  const data = logits.data as Float32Array
  const thr = options?.threshold ?? 0.5
  const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, z))))

  const { width: w, height: h } = rectified
  const mask = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    const sy = Math.min(pad.nh - 1, Math.max(0, Math.round(y * pad.scale)))
    for (let x = 0; x < w; x++) {
      const sx = Math.min(pad.nw - 1, Math.max(0, Math.round(x * pad.scale)))
      const li = (pad.top + sy) * longSide + (pad.left + sx)
      mask[y * w + x] = sigmoid(data[li]) > thr ? 255 : 0
    }
  }
  return { mask, width: w, height: h, inferenceMs }
}
