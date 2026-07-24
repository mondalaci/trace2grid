import type { MeshData, Vec2 } from '../../types'
import type { BuildRequest, WorkerResponse } from './api'

let worker: Worker | null = null
let nextId = 1
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>()

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data
      const entry = pending.get(msg.id)
      if (!entry) return
      pending.delete(msg.id)
      if (msg.ok) entry.resolve(msg.result)
      else entry.reject(new Error(msg.error))
    }
    worker.onerror = (event) => {
      const error = new Error(event.message || 'Geometry worker failed')
      for (const entry of pending.values()) entry.reject(error)
      pending.clear()
    }
  }
  return worker
}

function call<T>(msg: object): Promise<T> {
  const id = nextId++
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject })
    // JSON round-trip strips Vue reactivity proxies, which structured clone rejects.
    getWorker().postMessage(JSON.parse(JSON.stringify({ id, ...msg })))
  })
}

/** Offset a polygon outline by delta mm (positive = outward). */
export function offsetPolygon(points: Vec2[], delta: number): Promise<Vec2[][]> {
  return call<Vec2[][]>({ op: 'offset', points, delta })
}

/** Build the full bin mesh. Requests are serialized by the worker (one thread). */
export function buildBin(request: BuildRequest): Promise<MeshData> {
  return call<MeshData>({ op: 'build', request })
}
