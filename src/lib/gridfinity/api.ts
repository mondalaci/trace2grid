import type { BinConfig, MeshData, Vec2 } from '../../types'

/**
 * One tool pocket to cut into the bin, in *centered* bin coordinates:
 * origin at the bin center, x right, y toward the "back" (i.e. editor y
 * flipped), mm units — the worker's 3D space looking down the z axis.
 */
export interface PocketSpec {
  polygons: Vec2[][]
  /** Cut depth from the bin top, mm. */
  depth: number
  notches: { x: number; y: number; radius: number; depth: number }[]
}

export interface BuildRequest {
  bin: BinConfig
  pockets: PocketSpec[]
}

export type WorkerRequest =
  | { id: number; op: 'offset'; points: Vec2[]; delta: number }
  | { id: number; op: 'build'; request: BuildRequest }

export type WorkerResponse =
  | { id: number; ok: true; result: Vec2[][] | MeshData }
  | { id: number; ok: false; error: string }
