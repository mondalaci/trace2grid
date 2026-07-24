import type { Vec2 } from '../types'

export function polygonToSvgPoints(points: Vec2[]): string {
  return points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')
}

/** Convert client (screen) coordinates to SVG user-space. */
export function svgClientPoint(svg: SVGSVGElement, clientX: number, clientY: number): Vec2 {
  const ctm = svg.getScreenCTM()
  if (!ctm) return [0, 0]
  const p = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse())
  return [p.x, p.y]
}

/** Convert a pointer event position to SVG user-space coordinates. */
export function svgEventPoint(svg: SVGSVGElement, evt: MouseEvent): Vec2 {
  return svgClientPoint(svg, evt.clientX, evt.clientY)
}
