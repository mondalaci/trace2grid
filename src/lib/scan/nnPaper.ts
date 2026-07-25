/**
 * Neural paper-sheet segmentation via ONNX Runtime Web (WebGPU / WASM).
 *
 * Train/export: see ml/README.md. Weights: `public/models/paperseg.onnx`
 * (gitignored — run `npm run ml:export` after training).
 */

import type { InferenceSession, Tensor } from 'onnxruntime-web'

export interface NnPaperMeta {
  longSide: number
  normalize: 'rgb_to_[-1,1]'
}

export interface NnPaperResult {
  /** Binary mask, 0/255, same size as the source photo canvas. */
  mask: Uint8Array
  width: number
  height: number
  inferenceMs: number
}

const DEFAULT_MODEL_URL = `${import.meta.env.BASE_URL}models/paperseg.onnx`
const DEFAULT_META: NnPaperMeta = { longSide: 768, normalize: 'rgb_to_[-1,1]' }

let sessionPromise: Promise<InferenceSession> | null = null
let meta: NnPaperMeta = DEFAULT_META
let ortReady: Promise<typeof import('onnxruntime-web')> | null = null

async function loadOrt() {
  if (!ortReady) {
    ortReady = import('onnxruntime-web')
  }
  return ortReady
}

/** Lazily create an ORT session for paper segmentation. */
export function loadPaperSegSession(modelUrl = DEFAULT_MODEL_URL): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const url = modelUrl
      const probe = await fetch(url, { method: 'HEAD' }).catch(() => null)
      if (!probe?.ok) {
        const get = await fetch(url).catch(() => null)
        if (!get?.ok) {
          throw new Error(
            `Missing paper segmentation model at ${url}. Run \`npm run ml:export\` (or \`npm run ml\`) to place paperseg.onnx in public/models/.`,
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
  fillStyle = '#000000',
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
  ctx.fillStyle = fillStyle
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

/** Run TinyUNet ONNX on a full photo; returns a paper mask at photo resolution. */
export async function segmentPaperNn(
  photo: HTMLCanvasElement,
  options?: { modelUrl?: string; threshold?: number },
): Promise<NnPaperResult> {
  const ort = await loadOrt()
  const session = await loadPaperSegSession(options?.modelUrl)
  const longSide = meta.longSide
  const { tensor, pad } = letterboxRgb(photo, longSide, '#000000')
  const input = new ort.Tensor('float32', tensor, [1, 3, longSide, longSide])

  const t0 = performance.now()
  const out = await session.run({ image: input })
  const inferenceMs = performance.now() - t0

  const logits = (out.logits ?? Object.values(out)[0]) as Tensor
  const data = logits.data as Float32Array
  const thr = options?.threshold ?? 0.5
  const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, z))))

  const { width: w, height: h } = photo
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
