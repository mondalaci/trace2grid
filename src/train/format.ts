import type { PaperSizeId, Vec2 } from '../types'

/** One ground-truth tool silhouette: outer ring minus hole rings (even-odd). */
export interface TruthPolygon {
  /** Outer boundary in paper millimeters (y-down, origin at top-left corner). */
  outerMm: Vec2[]
  holesMm: Vec2[][]
}

/**
 * Ground-truth annotation for one photo in training/, persisted next to it
 * as training/<photo>.json. `truth` holds the final corrected tool
 * outlines: the labeling UI seeds them from the detector output and the user
 * edits them with add/remove lassos until they match reality.
 */
export interface TrainingAnnotation {
  photo: string
  paperSizeId: PaperSizeId
  landscape: boolean
  /** Paper corners normalized 0..1 to photo width/height (tl, tr, br, bl). */
  corners: [Vec2, Vec2, Vec2, Vec2]
  truth: TruthPolygon[]
}
