import type { PaperSize, Vec2 } from '../../types'

export interface PdfTool {
  name: string
  /** Pocket outline(s), placed in bin coordinates (mm, y down). */
  polygons: Vec2[][]
  notches: { x: number; y: number; radius: number }[]
}

export interface PdfLayout {
  binW: number
  binH: number
  gridPitch: number
  tools: PdfTool[]
  paper: PaperSize
}

const MARGIN = 12
const FOOTER = 26

/**
 * Render the bin layout at exactly 1:1 scale so the user can verify tool fit
 * on a paper printout before committing to a 3D print.
 */
export async function exportLayoutPdf(layout: PdfLayout): Promise<Blob> {
  const { jsPDF } = await import('jspdf')

  const landscape = layout.binW > layout.binH
  const pageW = landscape ? layout.paper.heightMm : layout.paper.widthMm
  const pageH = landscape ? layout.paper.widthMm : layout.paper.heightMm
  const doc = new jsPDF({
    unit: 'mm',
    format: [pageW, pageH],
    orientation: pageW > pageH ? 'landscape' : 'portrait',
  })

  const usableW = pageW - 2 * MARGIN
  const usableH = pageH - MARGIN - FOOTER
  const fits = layout.binW <= usableW && layout.binH <= usableH
  const ox = Math.max(MARGIN, (pageW - layout.binW) / 2)
  const oy = Math.max(MARGIN, (pageH - FOOTER + MARGIN - layout.binH) / 2)

  // Header
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.text('Trace2Grid — 1:1 layout check', MARGIN, 8)
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.setTextColor(90)
  doc.text('Print at 100% scale (no "fit to page"). Lay your tools on the outlines to verify fit.', MARGIN, 12)
  doc.setTextColor(0)

  // Bin outline + grid
  doc.setDrawColor(60)
  doc.setLineWidth(0.4)
  doc.roundedRect(ox + 0.25, oy + 0.25, layout.binW - 0.5, layout.binH - 0.5, 3.75, 3.75, 'S')
  doc.setDrawColor(190)
  doc.setLineWidth(0.15)
  for (let x = layout.gridPitch; x < layout.binW - 0.1; x += layout.gridPitch) {
    doc.line(ox + x, oy, ox + x, oy + layout.binH)
  }
  for (let y = layout.gridPitch; y < layout.binH - 0.1; y += layout.gridPitch) {
    doc.line(ox, oy + y, ox + layout.binW, oy + y)
  }

  // Tool pockets
  doc.setDrawColor(20, 90, 200)
  doc.setLineWidth(0.3)
  for (const tool of layout.tools) {
    for (const poly of tool.polygons) {
      if (poly.length < 3) continue
      drawPolygon(doc, poly, ox, oy)
    }
    for (const notch of tool.notches) {
      doc.circle(ox + notch.x, oy + notch.y, notch.radius, 'S')
    }
    if (tool.polygons[0] && tool.polygons[0].length > 0) {
      const [lx, ly] = polygonLabelPoint(tool.polygons[0])
      doc.setFontSize(7)
      doc.setTextColor(20, 90, 200)
      doc.text(tool.name, ox + lx, oy + ly, { align: 'center' })
      doc.setTextColor(0)
    }
  }

  // Calibration bars
  const cy = pageH - 14
  doc.setDrawColor(0)
  doc.setLineWidth(0.4)
  doc.line(MARGIN, cy, MARGIN + 100, cy)
  for (const tick of [0, 50, 100]) {
    doc.line(MARGIN + tick, cy - 1.5, MARGIN + tick, cy + 1.5)
  }
  doc.setFontSize(8)
  doc.text('100 mm — measure with a ruler; must be exactly 100 mm', MARGIN, cy + 5)

  const vx = pageW - MARGIN
  doc.line(vx, cy, vx, cy - 50)
  doc.line(vx - 1.5, cy, vx + 0, cy)
  doc.line(vx - 1.5, cy - 50, vx, cy - 50)
  doc.text('50 mm', vx - 1, cy - 52, { align: 'right' })

  if (!fits) {
    doc.setTextColor(200, 30, 30)
    doc.setFontSize(9)
    doc.text(
      `Warning: the ${layout.binW} × ${layout.binH} mm bin exceeds this page — outlines are clipped.`,
      MARGIN,
      pageH - 6,
    )
    doc.setTextColor(0)
  }

  return doc.output('blob')
}

function drawPolygon(doc: { lines: Function }, poly: Vec2[], ox: number, oy: number) {
  const segments: [number, number][] = []
  for (let i = 1; i < poly.length; i++) {
    segments.push([poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1]])
  }
  doc.lines(segments, ox + poly[0][0], oy + poly[0][1], [1, 1], 'S', true)
}

function polygonLabelPoint(poly: Vec2[]): Vec2 {
  let sx = 0
  let sy = 0
  for (const [x, y] of poly) {
    sx += x
    sy += y
  }
  return [sx / poly.length, sy / poly.length]
}
