import type { Vec2 } from '../types'

const DEG = Math.PI / 180

/** Apply the tool placement transform: rotate by deg, then translate by (tx, ty). */
export function transformPoints(points: Vec2[], tx: number, ty: number, deg: number): Vec2[] {
  const c = Math.cos(deg * DEG)
  const s = Math.sin(deg * DEG)
  return points.map(([x, y]) => [tx + x * c - y * s, ty + x * s + y * c])
}

export function polygonArea(points: Vec2[]): number {
  let a = 0
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    a += (points[j][0] + points[i][0]) * (points[j][1] - points[i][1])
  }
  return a / 2
}

/** Area-weighted centroid. Falls back to vertex average for degenerate polygons. */
export function polygonCentroid(points: Vec2[]): Vec2 {
  const a = polygonArea(points)
  if (Math.abs(a) < 1e-9) {
    let sx = 0
    let sy = 0
    for (const [x, y] of points) {
      sx += x
      sy += y
    }
    return [sx / points.length, sy / points.length]
  }
  let cx = 0
  let cy = 0
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const cross = points[j][0] * points[i][1] - points[i][0] * points[j][1]
    cx += (points[j][0] + points[i][0]) * cross
    cy += (points[j][1] + points[i][1]) * cross
  }
  return [cx / (6 * a), cy / (6 * a)]
}

export interface BBox {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

export function polygonBBox(points: Vec2[]): BBox {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const [x, y] of points) {
    if (x < minX) minX = x
    if (y < minY) minY = y
    if (x > maxX) maxX = x
    if (y > maxY) maxY = y
  }
  return { minX, minY, maxX, maxY }
}

export function bboxesOverlap(a: BBox, b: BBox): boolean {
  return a.minX < b.maxX && b.minX < a.maxX && a.minY < b.maxY && b.minY < a.maxY
}

export function pointInPolygon(p: Vec2, points: Vec2[]): boolean {
  let inside = false
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i]
    const [xj, yj] = points[j]
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) {
      inside = !inside
    }
  }
  return inside
}

function segmentsIntersect(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2): boolean {
  const d = (a2[0] - a1[0]) * (b2[1] - b1[1]) - (a2[1] - a1[1]) * (b2[0] - b1[0])
  if (Math.abs(d) < 1e-12) return false
  const t = ((b1[0] - a1[0]) * (b2[1] - b1[1]) - (b1[1] - a1[1]) * (b2[0] - b1[0])) / d
  const u = ((b1[0] - a1[0]) * (a2[1] - a1[1]) - (b1[1] - a1[1]) * (a2[0] - a1[0])) / d
  return t >= 0 && t <= 1 && u >= 0 && u <= 1
}

/** True if the polygon boundaries cross or one polygon contains the other. */
export function polygonsIntersect(a: Vec2[], b: Vec2[]): boolean {
  if (!bboxesOverlap(polygonBBox(a), polygonBBox(b))) return false
  for (let i = 0, j = a.length - 1; i < a.length; j = i++) {
    for (let k = 0, l = b.length - 1; k < b.length; l = k++) {
      if (segmentsIntersect(a[j], a[i], b[l], b[k])) return true
    }
  }
  return pointInPolygon(a[0], b) || pointInPolygon(b[0], a)
}

/** Closest point on the polygon outline (treated as a closed polyline) to p. */
export function closestPointOnPolygon(points: Vec2[], p: Vec2): { point: Vec2; dist: number } {
  let best: Vec2 = points[0]
  let bestDist = Infinity
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [x1, y1] = points[j]
    const [x2, y2] = points[i]
    const dx = x2 - x1
    const dy = y2 - y1
    const lenSq = dx * dx + dy * dy
    let t = lenSq > 0 ? ((p[0] - x1) * dx + (p[1] - y1) * dy) / lenSq : 0
    t = Math.max(0, Math.min(1, t))
    const q: Vec2 = [x1 + t * dx, y1 + t * dy]
    const d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
    if (d < bestDist) {
      bestDist = d
      best = q
    }
  }
  return { point: best, dist: Math.sqrt(bestDist) }
}
