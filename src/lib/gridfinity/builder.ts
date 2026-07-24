import type { CrossSection, Manifold, ManifoldToplevel } from 'manifold-3d'
import type { MeshData, Vec2 } from '../../types'
import type { BuildRequest } from './api'
import {
  binOuterDepth,
  binOuterWidth,
  binTotalHeight,
  FOOT_CHAMFER_BOTTOM,
  FOOT_HEIGHT,
  FOOT_RADIUS_BOTTOM,
  FOOT_RADIUS_MID,
  FOOT_STRAIGHT,
  FOOT_WIDTH_BOTTOM,
  FOOT_WIDTH_MID,
  FOOT_WIDTH_TOP,
  GRID_PITCH,
  LIP_CHAMFER_BOTTOM,
  LIP_HEIGHT,
  LIP_INSET,
  LIP_STRAIGHT,
  MAGNET_DEPTH,
  MAGNET_DIAMETER,
  MAGNET_OFFSET,
  OUTER_RADIUS,
} from './spec'

type Disposable = { delete(): void }

/** Thin slab used as a hull endpoint (chamfer construction). */
const SLAB = 0.02
const ROUND_SEGMENTS = 32

/**
 * Offset a single polygon outline by delta mm (positive = outward).
 * Returns the resulting polygon(s) — offsetting can split/merge contours.
 */
export function offsetOutline(wasm: ManifoldToplevel, points: Vec2[], delta: number): Vec2[][] {
  if (delta === 0 || points.length < 3) return [points]
  const cs = new wasm.CrossSection([points], 'EvenOdd')
  const off = cs.offset(delta, 'Round', 2, ROUND_SEGMENTS)
  const simplified = off.simplify(0.02)
  const polygons = simplified.toPolygons() as Vec2[][]
  cs.delete()
  off.delete()
  simplified.delete()
  return polygons
}

class Scope {
  private items: Disposable[] = []

  track<T extends Disposable>(item: T): T {
    this.items.push(item)
    return item
  }

  dispose() {
    for (const item of this.items) {
      try {
        item.delete()
      } catch {
        // already deleted
      }
    }
    this.items = []
  }
}

export function buildBinMesh(wasm: ManifoldToplevel, request: BuildRequest): MeshData {
  const { CrossSection, Manifold } = wasm
  const scope = new Scope()
  const t = <T extends Disposable>(item: T): T => scope.track(item)

  const { bin, pockets } = request
  const width = binOuterWidth(bin.gridX)
  const depth = binOuterDepth(bin.gridY)
  const height = binTotalHeight(bin.heightUnits)

  const roundedRect = (w: number, h: number, r: number): CrossSection => {
    const inner = t(CrossSection.square([w - 2 * r, h - 2 * r], true))
    return t(inner.offset(r, 'Round', 2, ROUND_SEGMENTS))
  }

  const slab = (cs: CrossSection, z: number): Manifold => t(t(cs.extrude(SLAB)).translate(0, 0, z))

  try {
    // --- Base feet, one per 42 mm cell ---
    const footBottom = roundedRect(FOOT_WIDTH_BOTTOM, FOOT_WIDTH_BOTTOM, FOOT_RADIUS_BOTTOM)
    const footMid = roundedRect(FOOT_WIDTH_MID, FOOT_WIDTH_MID, FOOT_RADIUS_MID)
    const footTop = roundedRect(FOOT_WIDTH_TOP, FOOT_WIDTH_TOP, OUTER_RADIUS)

    const chamferBottom = t(
      Manifold.hull([slab(footBottom, 0), slab(footMid, FOOT_CHAMFER_BOTTOM - SLAB)]),
    )
    const straight = t(t(footMid.extrude(FOOT_STRAIGHT)).translate(0, 0, FOOT_CHAMFER_BOTTOM))
    const chamferTop = t(
      Manifold.hull([
        slab(footMid, FOOT_CHAMFER_BOTTOM + FOOT_STRAIGHT),
        slab(footTop, FOOT_HEIGHT - SLAB),
      ]),
    )
    let foot = t(Manifold.union([chamferBottom, straight, chamferTop]))

    if (bin.magnetHoles) {
      const holes: Manifold[] = []
      for (const sx of [-1, 1]) {
        for (const sy of [-1, 1]) {
          holes.push(
            t(
              t(
                Manifold.cylinder(MAGNET_DEPTH + 0.01, MAGNET_DIAMETER / 2, MAGNET_DIAMETER / 2, 48),
              ).translate(sx * MAGNET_OFFSET, sy * MAGNET_OFFSET, -0.005),
            ),
          )
        }
      }
      foot = t(foot.subtract(t(Manifold.union(holes))))
    }

    const parts: Manifold[] = []
    for (let i = 0; i < bin.gridX; i++) {
      for (let j = 0; j < bin.gridY; j++) {
        const cx = (i + 0.5) * GRID_PITCH - (bin.gridX * GRID_PITCH) / 2
        const cy = (j + 0.5) * GRID_PITCH - (bin.gridY * GRID_PITCH) / 2
        parts.push(t(foot.translate(cx, cy, 0)))
      }
    }

    // --- Solid body above the feet ---
    const bodyCS = roundedRect(width, depth, OUTER_RADIUS)
    parts.push(t(t(bodyCS.extrude(height - FOOT_HEIGHT)).translate(0, 0, FOOT_HEIGHT)))

    // --- Stacking lip: a rim whose cavity mirrors the foot profile ---
    if (bin.stackingLip) {
      const ring = t(t(bodyCS.extrude(LIP_HEIGHT)).translate(0, 0, height))
      const insetFloor = t(bodyCS.offset(-LIP_INSET, 'Round', 2, ROUND_SEGMENTS))
      const insetMid = t(bodyCS.offset(-(LIP_INSET - LIP_CHAMFER_BOTTOM), 'Round', 2, ROUND_SEGMENTS))
      // Overshoot outward/up so the top chamfer exits cleanly through the rim.
      const overshoot = 0.1
      const insetTop = t(bodyCS.offset(overshoot, 'Round', 2, ROUND_SEGMENTS))

      const cavity = t(
        Manifold.union([
          t(Manifold.hull([slab(insetFloor, height), slab(insetMid, height + LIP_CHAMFER_BOTTOM - SLAB)])),
          t(t(insetMid.extrude(LIP_STRAIGHT)).translate(0, 0, height + LIP_CHAMFER_BOTTOM)),
          t(
            Manifold.hull([
              slab(insetMid, height + LIP_CHAMFER_BOTTOM + LIP_STRAIGHT),
              slab(insetTop, height + LIP_HEIGHT + overshoot),
            ]),
          ),
        ]),
      )
      parts.push(t(ring.subtract(cavity)))
    }

    let result = t(Manifold.union(parts))

    // --- Tool pockets and finger notches ---
    const cutters: Manifold[] = []
    const topCut = height + (bin.stackingLip ? LIP_HEIGHT : 0) + 1
    for (const pocket of pockets) {
      const valid = pocket.polygons.filter((p) => p.length >= 3)
      if (valid.length > 0) {
        const cs = t(new CrossSection(valid, 'EvenOdd'))
        cutters.push(
          t(t(cs.extrude(pocket.depth + topCut - height)).translate(0, 0, height - pocket.depth)),
        )
      }
      for (const notch of pocket.notches) {
        cutters.push(
          t(
            t(
              Manifold.cylinder(notch.depth + topCut - height, notch.radius, notch.radius, 48),
            ).translate(notch.x, notch.y, height - notch.depth),
          ),
        )
      }
    }
    if (cutters.length > 0) {
      result = t(result.subtract(t(Manifold.union(cutters))))
    }

    const mesh = result.getMesh()
    const numProp = mesh.numProp
    const vertCount = mesh.vertProperties.length / numProp
    let positions: Float32Array
    if (numProp === 3) {
      positions = mesh.vertProperties.slice()
    } else {
      positions = new Float32Array(vertCount * 3)
      for (let v = 0; v < vertCount; v++) {
        positions[v * 3] = mesh.vertProperties[v * numProp]
        positions[v * 3 + 1] = mesh.vertProperties[v * numProp + 1]
        positions[v * 3 + 2] = mesh.vertProperties[v * numProp + 2]
      }
    }
    return { positions, indices: mesh.triVerts.slice() }
  } finally {
    scope.dispose()
  }
}
