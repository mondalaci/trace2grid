/**
 * Gridfinity spec constants (Zack Freedman's original spec), all mm.
 * https://gridfinity.xyz/specification/
 */
export const GRID_PITCH = 42
/** Gap between bin outer wall and the 42 mm grid cell, per side. */
export const BIN_CLEARANCE = 0.25
/** One Gridfinity height unit. */
export const HEIGHT_UNIT = 7
/** Height of the base foot profile. */
export const FOOT_HEIGHT = 4.75
/** Outer corner radius of a bin / foot top. */
export const OUTER_RADIUS = 3.75

/** Foot profile, bottom to top: 45° chamfer, straight wall, 45° chamfer. */
export const FOOT_CHAMFER_BOTTOM = 0.8
export const FOOT_STRAIGHT = 1.8
export const FOOT_CHAMFER_TOP = 2.15
/** Foot widths at the profile breakpoints (per 42 mm cell). */
export const FOOT_WIDTH_BOTTOM = 35.6
export const FOOT_WIDTH_MID = 37.2
export const FOOT_WIDTH_TOP = GRID_PITCH - 2 * BIN_CLEARANCE // 41.5
export const FOOT_RADIUS_BOTTOM = 0.8
export const FOOT_RADIUS_MID = 1.6

/** Stacking lip profile, bottom to top: 45° chamfer, straight, 45° chamfer to the rim. */
export const LIP_CHAMFER_BOTTOM = 0.7
export const LIP_STRAIGHT = 1.8
export const LIP_CHAMFER_TOP = 1.9
export const LIP_HEIGHT = LIP_CHAMFER_BOTTOM + LIP_STRAIGHT + LIP_CHAMFER_TOP // 4.4
/** Inset of the lip cavity floor from the outer wall. */
export const LIP_INSET = LIP_CHAMFER_BOTTOM + LIP_CHAMFER_TOP // 2.6

export const MAGNET_DIAMETER = 6.5
export const MAGNET_DEPTH = 2.4
/** Magnet hole centers are 26 mm apart within a cell (±13 from cell center). */
export const MAGNET_OFFSET = 13

/** Practical limits used by the editor. */
export const MIN_WALL = 1.2
/** Minimum floor left above the foot profile under the deepest pocket. */
export const MIN_FLOOR_ABOVE_FOOT = 1.0

export function binOuterWidth(gridX: number): number {
  return gridX * GRID_PITCH - 2 * BIN_CLEARANCE
}

export function binOuterDepth(gridY: number): number {
  return gridY * GRID_PITCH - 2 * BIN_CLEARANCE
}

export function binTotalHeight(heightUnits: number): number {
  return heightUnits * HEIGHT_UNIT
}

export function maxPocketDepth(heightUnits: number): number {
  return binTotalHeight(heightUnits) - FOOT_HEIGHT - MIN_FLOOR_ABOVE_FOOT
}

export function defaultPocketDepth(heightUnits: number): number {
  // Leave the first height unit as floor when possible.
  return Math.max(
    HEIGHT_UNIT,
    Math.min(binTotalHeight(heightUnits) - HEIGHT_UNIT, maxPocketDepth(heightUnits)),
  )
}
