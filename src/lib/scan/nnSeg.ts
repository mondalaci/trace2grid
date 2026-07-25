/**
 * Neural tool segmentation via ONNX Runtime Web (WebGPU / WASM).
 *
 * Train/export: see ml/README.md. Weights live at `public/models/toolseg.onnx`
 * (gitignored — run `npm run ml:export` after training).
 */

import type { InferenceSession, Tensor } from 'onnxruntime-web'
import type { ExtractedContour, ExtractOptions } from './contours'
import { contoursFromBinaryMask } from './contours'
import type { CV } from './opencv'

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

const DEFAULT_MODEL_URL = `${import.meta.env.BASE_URL}models/toolseg.onnx`
const DEFAULT_META: NnSegMeta = { longSide: 768, normalize: 'rgb_to_[-1,1]' }

let sessionPromise: Promise<InferenceSession> | null = null
let meta: NnSegMeta = DEFAULT_META
let ortReady: Promise<typeof import('onnxruntime-web')> | null = null

async function loadOrt() {
  if (!ortReady) {
    // Default export is the WASM-inlined bundle (ort.bundle.min.mjs).
    ortReady = import('onnxruntime-web')
  }
  return ortReady
}

/** Map CaptureView sensitivity (−60…60) to sigmoid threshold (more sensitive → lower thr). */
export function thresholdFromSensitivity(sensitivity: number): number {
  return Math.min(0.85, Math.max(0.15, 0.5 - sensitivity * (0.35 / 60)))
}

/** Lazily create an ORT session (WebGPU if available, else WASM). */
export function loadToolSegSession(modelUrl = DEFAULT_MODEL_URL): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const url = modelUrl
      const probe = await fetch(url, { method: 'HEAD' }).catch(() => null)
      if (!probe?.ok) {
        const get = await fetch(url).catch(() => null)
        if (!get?.ok) {
          throw new Error(
            `Missing tool segmentation model at ${url}. Run \`npm run ml:export\` (or \`npm run ml\`) to place toolseg.onnx in public/models/.`,
          )
        }
      }
      try {
        const res = await fetch(url.replace(/\.onnx$/, '.json'))
        if (res.ok) meta = { ...DEFAULT_META, ...(await res.json()) }
      } catch {
        /* keep defaults */
      }
      const ort = await loadOrt()
      try {
        return await ort.InferenceSession.create(url, {
          executionProviders: ['webgpu', 'wasm'],
        })
      } catch {
        return ort.InferenceSession.create(url, {
          executionProviders: ['wasm'],
        })
      }
    })().catch((err) => {
      sessionPromise = null
      throw err
    })
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

/** Production detector: ONNX mask → OpenCV contours (same outline format as classical). */
export async function extractToolContoursNn(
  cv: CV,
  rectified: HTMLCanvasElement,
  options: ExtractOptions,
): Promise<ExtractedContour[]> {
  const threshold = thresholdFromSensitivity(options.sensitivity ?? 0)
  const { mask, width, height } = await segmentToolsNn(rectified, { threshold })
  return contoursFromBinaryMask(cv, mask, width, height, options)
}
