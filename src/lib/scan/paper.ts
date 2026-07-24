import type { PaperSize, PaperSizeId, Vec2 } from '../../types'
import type { CV } from './opencv'

export const PAPER_SIZES: PaperSize[] = [
  { id: 'a4', name: 'A4 · 210 × 297 mm (ISO, most of the world)', widthMm: 210, heightMm: 297 },
  { id: 'letter', name: 'US Letter · 8.5 × 11 in', widthMm: 215.9, heightMm: 279.4 },
  { id: 'legal', name: 'US Legal · 8.5 × 14 in', widthMm: 215.9, heightMm: 355.6 },
]

export function paperById(id: PaperSizeId): PaperSize {
  return PAPER_SIZES.find((p) => p.id === id)!
}

function dist(a: Vec2, b: Vec2): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1])
}

/** Order 4 corners as top-left, top-right, bottom-right, bottom-left. */
export function orderCorners(pts: Vec2[]): [Vec2, Vec2, Vec2, Vec2] {
  const bySum = [...pts].sort((a, b) => a[0] + a[1] - (b[0] + b[1]))
  const tl = bySum[0]
  const br = bySum[3]
  const byDiff = [...pts].sort((a, b) => a[0] - a[1] - (b[0] - b[1]))
  const bl = byDiff[0]
  const tr = byDiff[3]
  return [tl, tr, br, bl]
}

/**
 * Find the paper sheet as the large bright *neutral-colored* region around
 * the image center. Color similarity separates white paper from light wood
 * or other warm backgrounds far more reliably than edge detection.
 * Returns ordered corners in image pixel coordinates, or null if not found.
 */
export function detectPaperQuad(cv: CV, canvas: HTMLCanvasElement): [Vec2, Vec2, Vec2, Vec2] | null {
  const maxDim = 900
  const scale = Math.min(1, maxDim / Math.max(canvas.width, canvas.height))

  const src = cv.imread(canvas)
  const small = new cv.Mat()
  const lab = new cv.Mat()
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(9, 9))
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  let mask: InstanceType<CV['Mat']> | null = null
  try {
    const w = Math.round(canvas.width * scale)
    const h = Math.round(canvas.height * scale)
    cv.resize(src, small, new cv.Size(w, h), 0, 0, cv.INTER_AREA)
    cv.cvtColor(small, small, cv.COLOR_RGBA2RGB)
    cv.GaussianBlur(small, small, new cv.Size(5, 5), 0)
    cv.cvtColor(small, lab, cv.COLOR_RGB2Lab)
    const data = lab.data

    // Reference paper color: median Lab of the central quarter (the user is
    // told to center the sheet, and tools cover well under half of it).
    const sampleL: number[] = []
    const sampleA: number[] = []
    const sampleB: number[] = []
    for (let y = h >> 2; y < h - (h >> 2); y += 3) {
      for (let x = w >> 2; x < w - (w >> 2); x += 3) {
        const i = (y * w + x) * 3
        sampleL.push(data[i])
        sampleA.push(data[i + 1])
        sampleB.push(data[i + 2])
      }
    }
    const med = (arr: number[]) => {
      arr.sort((a, b) => a - b)
      return arr[arr.length >> 1]
    }
    const paperL = med(sampleL)
    const paperA = med(sampleA)
    const paperB = med(sampleB)

    // Paper-similarity mask: close in brightness and chroma to the reference.
    // The lower brightness bound is looser: paper corners can be strongly
    // vignetted/shadowed, while nothing relevant is brighter than the paper.
    const lTolUp = Math.max(25, paperL * 0.15)
    const lTolDown = Math.max(45, paperL * 0.3)
    const abTol = 13
    const maskData = new Uint8Array(w * h)
    for (let i = 0; i < w * h; i++) {
      const dl = data[i * 3] - paperL
      if (
        dl < lTolUp &&
        dl > -lTolDown &&
        Math.abs(data[i * 3 + 1] - paperA) < abTol &&
        Math.abs(data[i * 3 + 2] - paperB) < abTol
      ) {
        maskData[i] = 255
      }
    }
    mask = cv.matFromArray(h, w, cv.CV_8UC1, maskData as unknown as number[])
    cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, kernel)
    cv.morphologyEx(mask, mask, cv.MORPH_OPEN, kernel)

    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    let bestIndex = -1
    let bestArea = w * h * 0.12
    for (let i = 0; i < contours.size(); i++) {
      const area = cv.contourArea(contours.get(i))
      if (area > bestArea) {
        bestArea = area
        bestIndex = i
      }
    }
    if (bestIndex < 0) return null

    // Convex hull, then relax the polygon approximation until 4 corners
    // remain. If the resulting quad lost noticeable area vs the hull (e.g. a
    // vignetted corner was missing from the mask and got cut diagonally),
    // fall back to the minimum-area rotated rectangle around the hull.
    const hull = new cv.Mat()
    const approx = new cv.Mat()
    try {
      cv.convexHull(contours.get(bestIndex), hull)
      const hullArea = cv.contourArea(hull)
      const peri = cv.arcLength(hull, true)
      let pts: Vec2[] | null = null
      for (const eps of [0.02, 0.03, 0.045, 0.06, 0.08]) {
        cv.approxPolyDP(hull, approx, eps * peri, true)
        if (approx.rows === 4 && cv.contourArea(approx) > hullArea * 0.97) {
          pts = []
          for (let p = 0; p < 4; p++) {
            pts.push([approx.data32S[p * 2], approx.data32S[p * 2 + 1]])
          }
          break
        }
      }
      if (!pts) {
        const box = cv.minAreaRect(contours.get(bestIndex))
        const rad = (box.angle * Math.PI) / 180
        const cos = Math.cos(rad)
        const sin = Math.sin(rad)
        const hw = box.size.width / 2
        const hh = box.size.height / 2
        pts = ([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]] as Vec2[]).map(([x, y]) => [
          box.center.x + x * cos - y * sin,
          box.center.y + x * sin + y * cos,
        ])
      }
      const ordered = orderCorners(pts)
      return ordered.map(([x, y]) => [x / scale, y / scale]) as [Vec2, Vec2, Vec2, Vec2]
    } finally {
      hull.delete()
      approx.delete()
    }
  } finally {
    src.delete()
    small.delete()
    lab.delete()
    mask?.delete()
    kernel.delete()
    contours.delete()
    hierarchy.delete()
  }
}

/**
 * Guess the paper size from the quad's side-length ratio, and whether the
 * sheet lies landscape in the photo.
 */
export function classifyPaper(corners: [Vec2, Vec2, Vec2, Vec2]): {
  sizeId: PaperSizeId
  landscape: boolean
} {
  const [tl, tr, br, bl] = corners
  const w = (dist(tl, tr) + dist(bl, br)) / 2
  const h = (dist(tl, bl) + dist(tr, br)) / 2
  const ratio = Math.max(w, h) / Math.min(w, h)
  let best: PaperSizeId = 'a4'
  let bestErr = Infinity
  for (const paper of PAPER_SIZES) {
    const err = Math.abs(ratio - paper.heightMm / paper.widthMm)
    if (err < bestErr) {
      bestErr = err
      best = paper.id
    }
  }
  return { sizeId: best, landscape: w > h }
}

/**
 * Perspective-rectify the paper region into a new canvas with a fixed
 * mm-to-pixel scale, so downstream contours are in real-world units.
 */
export function rectifyPaper(
  cv: CV,
  source: HTMLCanvasElement,
  corners: [Vec2, Vec2, Vec2, Vec2],
  paper: PaperSize,
  landscape: boolean,
  pxPerMm: number,
): HTMLCanvasElement {
  const wMm = landscape ? paper.heightMm : paper.widthMm
  const hMm = landscape ? paper.widthMm : paper.heightMm
  const outW = Math.round(wMm * pxPerMm)
  const outH = Math.round(hMm * pxPerMm)

  const src = cv.imread(source)
  const dst = new cv.Mat()
  const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, corners.flat())
  const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, outW, 0, outW, outH, 0, outH])
  const M = cv.getPerspectiveTransform(srcPts, dstPts)
  const out = document.createElement('canvas')
  try {
    cv.warpPerspective(src, dst, M, new cv.Size(outW, outH), cv.INTER_LINEAR, cv.BORDER_REPLICATE)
    cv.imshow(out, dst)
    return out
  } finally {
    src.delete()
    dst.delete()
    srcPts.delete()
    dstPts.delete()
    M.delete()
  }
}
