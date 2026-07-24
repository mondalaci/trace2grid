import type { Vec2 } from '../types'
import type { TruthPolygon } from './format'

/** Rasterize truth polygons (mm) onto a binary mask at the given px/mm. */
export function rasterizeTruth(
  truth: TruthPolygon[],
  widthPx: number,
  heightPx: number,
  pxPerMm: number,
): Uint8ClampedArray {
  const canvas = document.createElement('canvas')
  canvas.width = widthPx
  canvas.height = heightPx
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.fillStyle = '#fff'
  for (const poly of truth) {
    const path = new Path2D()
    const rings: Vec2[][] = [
      poly.outerMm.map(([x, y]) => [x * pxPerMm, y * pxPerMm]),
      ...poly.holesMm.map((h) => h.map(([x, y]) => [x * pxPerMm, y * pxPerMm] as Vec2)),
    ]
    for (const ring of rings) {
      ring.forEach(([x, y], i) => (i ? path.lineTo(x, y) : path.moveTo(x, y)))
      path.closePath()
    }
    ctx.fill(path, 'evenodd')
  }
  return ctx.getImageData(0, 0, widthPx, heightPx).data
}

/** Rasterize detected polygons (px) onto a binary mask. */
export function rasterizePolygonsPx(
  polygons: Vec2[][],
  widthPx: number,
  heightPx: number,
): Uint8ClampedArray {
  const canvas = document.createElement('canvas')
  canvas.width = widthPx
  canvas.height = heightPx
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.fillStyle = '#fff'
  for (const poly of polygons) {
    ctx.beginPath()
    poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
    ctx.closePath()
    ctx.fill()
  }
  return ctx.getImageData(0, 0, widthPx, heightPx).data
}

/** Intersection-over-union of two alpha masks (alpha > 127 = foreground). */
export function maskIoU(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  let inter = 0
  let union = 0
  for (let i = 0; i < a.length; i += 4) {
    const ta = a[i + 3] > 127
    const tb = b[i + 3] > 127
    if (ta && tb) inter++
    if (ta || tb) union++
  }
  return union === 0 ? 1 : inter / union
}
