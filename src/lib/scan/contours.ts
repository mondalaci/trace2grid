import type { Vec2 } from '../../types'
import type { CV } from './opencv'

export interface ExtractedContour {
  /** Simplified outline in rectified-image pixels. */
  pointsPx: Vec2[]
  areaMm2: number
}

export interface ExtractOptions {
  pxPerMm: number
  /** Ignore a border strip of the paper, mm (shadows/curled edges). */
  marginMm?: number
  /** -60..60. Higher catches lighter grays (risking shadows), lower is stricter. */
  sensitivity?: number
  /** Discard blobs smaller than this, mm². */
  minAreaMm2?: number
}

/**
 * Find tool silhouettes on the rectified (white) paper.
 *
 * Works in Lab space with three shadow-aware cues, because a global gray
 * threshold cannot tell a gray shadow from a gray tool:
 *  - chroma: pixel color differs from the paper color (colored handles).
 *    Shadows keep the paper's chroma, so they never trigger this.
 *  - darkness: pixel is much darker than the *local* background brightness
 *    (estimated with a large morphological closing, so lighting gradients
 *    cancel out). The sensitivity slider tunes only this threshold.
 *  - highlight: pixel is clearly brighter than the background (chrome
 *    speculars). Shadows are never brighter than the paper.
 */
export function extractToolContours(
  cv: CV,
  rectified: HTMLCanvasElement,
  options: ExtractOptions,
): ExtractedContour[] {
  const { pxPerMm } = options
  const marginPx = Math.round((options.marginMm ?? 5) * pxPerMm)
  const minAreaPx = (options.minAreaMm2 ?? 50) * pxPerMm * pxPerMm
  const sensitivity = options.sensitivity ?? 0

  // Sensitivity mapping, calibrated on real photos: shadows live at
  // chroma < ~25 and brightness ratio > ~0.5, tools outside that box.
  const darkRatio = Math.min(0.92, Math.max(0.15, 0.5 + sensitivity / 150))
  const chromaThresh = Math.min(40, Math.max(15, 27 - sensitivity / 5))
  const highlightDelta = 25

  const src = cv.imread(rectified)
  const rgb = new cv.Mat()
  const lab = new cv.Mat()
  const kernelOpen = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3))
  const kernelClose = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7))
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  let maskMat: InstanceType<CV['Mat']> | null = null
  let bgSmall: InstanceType<CV['Mat']> | null = null
  let bgMat: InstanceType<CV['Mat']> | null = null
  let lMat: InstanceType<CV['Mat']> | null = null
  try {
    cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB)
    cv.GaussianBlur(rgb, rgb, new cv.Size(3, 3), 0)
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab)

    const w = lab.cols
    const h = lab.rows
    const data = lab.data // interleaved L, a, b (8-bit)

    // Paper chroma reference: median a/b over a sparse sample (paper dominates).
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

    // Local background brightness: large closing of L at 1/4 resolution.
    lMat = new cv.Mat(h, w, cv.CV_8UC1)
    const lData = lMat.data
    for (let i = 0; i < w * h; i++) lData[i] = data[i * 3]
    bgSmall = new cv.Mat()
    const smallSize = new cv.Size(Math.max(1, w >> 2), Math.max(1, h >> 2))
    cv.resize(lMat, bgSmall, smallSize, 0, 0, cv.INTER_AREA)
    // Kernel spans ~45 mm at quarter resolution, wider than typical tools.
    const bgKernelPx = Math.max(3, 2 * Math.round((45 * pxPerMm) / 4 / 2) + 1)
    const bgKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(bgKernelPx, bgKernelPx))
    cv.morphologyEx(bgSmall, bgSmall, cv.MORPH_CLOSE, bgKernel)
    bgKernel.delete()
    bgMat = new cv.Mat()
    cv.resize(bgSmall, bgMat, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR)
    const bgData = bgMat.data

    // Per-pixel classification.
    const chromaThreshSq = chromaThresh * chromaThresh
    const mask = new Uint8Array(w * h)
    for (let i = 0; i < w * h; i++) {
      const L = data[i * 3]
      const da = data[i * 3 + 1] - paperA
      const db = data[i * 3 + 2] - paperB
      const bg = bgData[i]
      if (
        da * da + db * db > chromaThreshSq ||
        L < bg * darkRatio ||
        L > bg + highlightDelta
      ) {
        mask[i] = 255
      }
    }
    maskMat = cv.matFromArray(h, w, cv.CV_8UC1, mask as unknown as number[])

    // Blank out the paper border: edge shadows and page curl create false blobs.
    const black = new cv.Scalar(0)
    cv.rectangle(maskMat, new cv.Point(0, 0), new cv.Point(w, marginPx), black, -1)
    cv.rectangle(maskMat, new cv.Point(0, h - marginPx), new cv.Point(w, h), black, -1)
    cv.rectangle(maskMat, new cv.Point(0, 0), new cv.Point(marginPx, h), black, -1)
    cv.rectangle(maskMat, new cv.Point(w - marginPx, 0), new cv.Point(w, h), black, -1)

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
        rect.x + rect.width >= w - marginPx - 1 ||
        rect.y + rect.height >= h - marginPx - 1
      if (touchesEdge) continue

      const approx = new cv.Mat()
      try {
        // ~0.3 mm simplification: keeps curves smooth, drops pixel noise.
        cv.approxPolyDP(contour, approx, 0.3 * pxPerMm, true)
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
    src.delete()
    rgb.delete()
    lab.delete()
    lMat?.delete()
    bgSmall?.delete()
    bgMat?.delete()
    maskMat?.delete()
    kernelOpen.delete()
    kernelClose.delete()
    contours.delete()
    hierarchy.delete()
  }
}
