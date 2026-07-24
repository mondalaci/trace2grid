export type Vec2 = [number, number]

/** A finger notch attached to a tool pocket, in tool-local mm coordinates. */
export interface Notch {
  id: string
  /** Center, tool-local mm (y down, same space as Tool.points). */
  x: number
  y: number
  radius: number
  /** Cut depth in mm from the bin top. */
  depth: number
}

/**
 * A scanned tool contour placed in the bin.
 * `points` are in mm, relative to the contour centroid, y pointing down
 * (screen/editor convention). Placement maps local coords into bin space:
 * rotate by `rotationDeg`, then translate by (x, y).
 */
export interface Tool {
  id: string
  name: string
  points: Vec2[]
  /** Placement of the local origin in bin mm coordinates (origin = bin top-left corner in the editor). */
  x: number
  y: number
  rotationDeg: number
  /** Extra clearance offset applied around the contour, mm. */
  clearance: number
  /** Pocket depth from the bin top, mm. */
  pocketDepth: number
  notches: Notch[]
  /** Clearance-offset outline(s), tool-local mm; computed asynchronously. */
  offsetPolygons: Vec2[][] | null
}

export interface BinConfig {
  /** Footprint in 42 mm Gridfinity units. */
  gridX: number
  gridY: number
  /** Height in 7 mm Gridfinity units (includes the 4.75 mm base, excludes the stacking lip). */
  heightUnits: number
  stackingLip: boolean
  magnetHoles: boolean
}

export type PaperSizeId = 'a4' | 'letter' | 'legal'

export interface PaperSize {
  id: PaperSizeId
  name: string
  /** Short edge, mm. */
  widthMm: number
  /** Long edge, mm. */
  heightMm: number
}

/** Triangle mesh returned by the geometry worker (mm units, z up). */
export interface MeshData {
  positions: Float32Array
  indices: Uint32Array
}

export type Step = 'capture' | 'edit' | 'preview'
