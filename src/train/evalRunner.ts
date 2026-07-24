import { extractToolContours } from '../lib/scan/contours'
import { loadOpenCV } from '../lib/scan/opencv'
import { paperById, rectifyPaper } from '../lib/scan/paper'
import type { ContourParams } from '../lib/scan/params'
import type { Vec2 } from '../types'
import type { TrainingAnnotation } from './format'
import { maskIoU, rasterizePolygonsPx, rasterizeTruth } from './score'

const PX_PER_MM = 4
const MAX_DIM = 1800

export interface PhotoScore {
  photo: string
  iou: number
  detectedCount: number
  truthCount: number
}

export interface EvalResult {
  timestamp: string
  meanIoU: number
  scores: PhotoScore[]
  params?: Partial<ContourParams>
  note?: string
}

async function loadPhotoCanvas(name: string): Promise<HTMLCanvasElement> {
  const blob = await (await fetch(`/__training/photo/${encodeURIComponent(name)}`)).blob()
  const bitmap = await createImageBitmap(blob)
  const scale = Math.min(1, MAX_DIM / Math.max(bitmap.width, bitmap.height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(bitmap.width * scale)
  canvas.height = Math.round(bitmap.height * scale)
  canvas.getContext('2d')!.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  bitmap.close()
  return canvas
}

async function loadAnnotation(name: string): Promise<TrainingAnnotation | null> {
  const res = await fetch(`/__training/data/${encodeURIComponent(name)}`)
  if (!res.ok) return null
  return res.json()
}

/** Score one labeled photo using ground-truth corners (isolates contour quality). */
export async function scorePhoto(
  name: string,
  params?: Partial<ContourParams>,
  sensitivity = 0,
): Promise<PhotoScore | null> {
  const annotation = await loadAnnotation(name)
  if (!annotation?.truth?.length || !annotation.corners) return null

  const cv = await loadOpenCV()
  const photo = await loadPhotoCanvas(name)
  const corners = annotation.corners.map(
    ([x, y]) => [x * photo.width, y * photo.height] as Vec2,
  ) as [Vec2, Vec2, Vec2, Vec2]
  const paper = paperById(annotation.paperSizeId)
  const rectified = rectifyPaper(cv, photo, corners, paper, annotation.landscape, PX_PER_MM)
  const detected = extractToolContours(cv, rectified, {
    pxPerMm: PX_PER_MM,
    sensitivity,
    params,
  })
  const truthMask = rasterizeTruth(annotation.truth, rectified.width, rectified.height, PX_PER_MM)
  const detMask = rasterizePolygonsPx(
    detected.map((c) => c.pointsPx),
    rectified.width,
    rectified.height,
  )
  return {
    photo: name,
    iou: maskIoU(truthMask, detMask),
    detectedCount: detected.length,
    truthCount: annotation.truth.length,
  }
}

/** Score every labeled photo in training/. */
export async function scoreAllLabeled(
  params?: Partial<ContourParams>,
  sensitivity = 0,
): Promise<EvalResult> {
  const { photos } = await (await fetch('/__training/photos')).json()
  const scores: PhotoScore[] = []
  for (const name of photos as string[]) {
    const score = await scorePhoto(name, params, sensitivity)
    if (score) scores.push(score)
  }
  const meanIoU = scores.length ? scores.reduce((s, x) => s + x.iou, 0) / scores.length : 0
  return {
    timestamp: new Date().toISOString(),
    meanIoU,
    scores,
    params,
  }
}

/** Append one eval result to training/accuracy-log.csv via the dev server. */
export async function logAccuracy(result: EvalResult, note = ''): Promise<void> {
  await fetch('/__training/accuracy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      timestamp: result.timestamp,
      meanIoU: result.meanIoU,
      note,
      scores: result.scores,
      params: result.params ?? null,
    }),
  })
}
