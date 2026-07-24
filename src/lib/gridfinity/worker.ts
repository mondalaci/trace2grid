/// <reference lib="webworker" />
import Module from 'manifold-3d'
import wasmUrl from 'manifold-3d/manifold.wasm?url'
import type { MeshData } from '../../types'
import type { WorkerRequest, WorkerResponse } from './api'
import { buildBinMesh, offsetOutline } from './builder'

const wasmReady = (async () => {
  const wasm = await Module({ locateFile: () => wasmUrl })
  wasm.setup()
  return wasm
})()

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data
  try {
    const wasm = await wasmReady
    if (msg.op === 'offset') {
      const result = offsetOutline(wasm, msg.points, msg.delta)
      const response: WorkerResponse = { id: msg.id, ok: true, result }
      self.postMessage(response)
    } else {
      const result: MeshData = buildBinMesh(wasm, msg.request)
      const response: WorkerResponse = { id: msg.id, ok: true, result }
      self.postMessage(response, [result.positions.buffer, result.indices.buffer])
    }
  } catch (error) {
    const response: WorkerResponse = {
      id: msg.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }
    self.postMessage(response)
  }
}
