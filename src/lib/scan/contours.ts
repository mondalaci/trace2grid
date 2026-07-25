import type { Vec2 } from '../../types'
import type { CV } from './opencv'
import {
  chromaThreshAt,
  darkRatioAt,
  DEFAULT_CONTOUR_PARAMS,
  type ContourParams,
} from './params'

export interface ExtractedContour {
  /** Simplified outline in rectified-image pixels. */
  pointsPx: Vec2[]
  areaMm2: number
}

export interface ExtractOptions {
  pxPerMm: number
  marginMm?: number
  sensitivity?: number
  minAreaMm2?: number
  params?: Partial<ContourParams>
}

/**
 * Turn a binary mask (0 / 255) into simplified tool outlines.
 * Shared by classical Lab thresholding and neural segmentation.
 */
export function contoursFromBinaryMask(
  cv: CV,
  mask: Uint8Array,
  width: number,
  height: number,
  options: ExtractOptions,
): ExtractedContour[] {
  const params: ContourParams = { ...DEFAULT_CONTOUR_PARAMS, ...options.params }
  const { pxPerMm } = options
  const marginPx = Math.round((options.marginMm ?? params.marginMm) * pxPerMm)
  const minAreaPx = (options.minAreaMm2 ?? params.minAreaMm2) * pxPerMm * pxPerMm
  const openK = Math.max(1, params.openKernel | 1)
  const closeK = Math.max(1, params.closeKernel | 1)
  const kernelOpen = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(openK, openK))
  const kernelClose = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(closeK, closeK))
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  const maskMat = cv.matFromArray(height, width, cv.CV_8UC1, mask as unknown as number[])
  try {
    const black = new cv.Scalar(0)
    cv.rectangle(maskMat, new cv.Point(0, 0), new cv.Point(width, marginPx), black, -1)
    cv.rectangle(maskMat, new cv.Point(0, height - marginPx), new cv.Point(width, height), black, -1)
    cv.rectangle(maskMat, new cv.Point(0, 0), new cv.Point(marginPx, height), black, -1)
    cv.rectangle(maskMat, new cv.Point(width - marginPx, 0), new cv.Point(width, height), black, -1)

    cv.morphologyEx(maskMat, maskMat, cv.MORPH_OPEN, kernelOpen)
    cv.morphologyEx(maskMat, maskMat, cv.MORPH_CLOSE, kernelClose)
    cv.findContours(maskMat, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    const results: ExtractedContour[] = []
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i)
      const areaPx = cv.contourArea(contour)
      if (areaPx < minAreaPx) continue

      const rect = cv.boundingRect(contour)
      const touchesEdge =
        rect.x <= marginPx + 1 ||
        rect.y <= marginPx + 1 ||
        rect.x + rect.width >= width - marginPx - 1 ||
        rect.y + rect.height >= height - marginPx - 1
      if (touchesEdge) continue

      const approx = new cv.Mat()
      try {
        cv.approxPolyDP(contour, approx, params.approxEpsMm * pxPerMm, true)
        if (approx.rows < 3) continue
        const pointsPx: Vec2[] = []
        for (let p = 0; p < approx.rows; p++) {
          pointsPx.push([approx.data32S[p * 2], approx.data32S[p * 2 + 1]])
        }
        results.push({ pointsPx, areaMm2: areaPx / (pxPerMm * pxPerMm) })
      } finally {
        approx.delete()
      }
    }
    results.sort((a, b) => b.areaMm2 - a.areaMm2)
    return results
  } finally {
    maskMat.delete()
    kernelOpen.delete()
    kernelClose.delete()
    contours.delete()
    hierarchy.delete()
  }
}

/**
 * Classical tool silhouettes on rectified paper (Lab chroma / darkness / highlight).
 * Kept for eval/tuning; production capture uses neural segmentation (`extractToolContoursNn`).
 */
export function extractToolContours(
  cv: CV,
  rectified: HTMLCanvasElement,
  options: ExtractOptions,
): ExtractedContour[] {
  const params: ContourParams = { ...DEFAULT_CONTOUR_PARAMS, ...options.params }
  const { pxPerMm } = options
  const sensitivity = options.sensitivity ?? 0

  const darkRatio = darkRatioAt(params, sensitivity)
  const chromaThresh = chromaThreshAt(params, sensitivity)
  const highlightDelta = params.highlightDelta

  const src = cv.imread(rectified)
  const rgb = new cv.Mat()
  const lab = new cv.Mat()
  let bgSmall: InstanceType<CV['Mat']> | null = null
  let bgMat: InstanceType<CV['Mat']> | null = null
  let lMat: InstanceType<CV['Mat']> | null = null
  try {
    cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB)
    cv.GaussianBlur(rgb, rgb, new cv.Size(3, 3), 0)
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab)

    const w = lab.cols
    const h = lab.rows
    const data = lab.data

    const sampleA: number[] = []
    const sampleB: number[] = []
    for (let i = 0; i < w * h; i += 37) {
      sampleA.push(data[i * 3 + 1])
      sampleB.push(data[i * 3 + 2])
    }
    sampleA.sort((x, y) => x - y)
    sampleB.sort((x, y) => x - y)
    const paperA = sampleA[sampleA.length >> 1]
    const paperB = sampleB[sampleB.length >> 1]

    lMat = new cv.Mat(h, w, cv.CV_8UC1)
    for (let i = 0; i < w * h; i++) lMat.data[i] = data[i * 3]
    bgSmall = new cv.Mat()
    cv.resize(lMat, bgSmall, new cv.Size(Math.max(1, w >> 2), Math.max(1, h >> 2)), 0, 0, cv.INTER_AREA)
    const bgKernelPx = Math.max(3, 2 * Math.round((params.bgKernelMm * pxPerMm) / 4 / 2) + 1)
    const bgKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(bgKernelPx, bgKernelPx))
    cv.morphologyEx(bgSmall, bgSmall, cv.MORPH_CLOSE, bgKernel)
    bgKernel.delete()
    bgMat = new cv.Mat()
    cv.resize(bgSmall, bgMat, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR)
    const bgData = bgMat.data

    const chromaThreshSq = chromaThresh * chromaThresh
    const mask = new Uint8Array(w * h)
    for (let i = 0; i < w * h; i++) {
      const L = data[i * 3]
      const da = data[i * 3 + 1] - paperA
      const db = data[i * 3 + 2] - paperB
      const chromaSq = da * da + db * db
      const bg = bgData[i]
      if (chromaSq > chromaThreshSq || L < bg * darkRatio || L > bg + highlightDelta) {
        mask[i] = 255
      }
    }

    return contoursFromBinaryMask(cv, mask, w, h, options)
  } finally {
    src.delete()
    rgb.delete()
    lab.delete()
    lMat?.delete()
    bgSmall?.delete()
    bgMat?.delete()
  }
}
