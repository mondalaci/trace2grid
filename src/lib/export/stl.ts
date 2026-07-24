import type { MeshData } from '../../types'

/** Serialize a triangle mesh (mm, z-up) to binary STL. */
export function meshToBinarySTL(mesh: MeshData): Blob {
  const triCount = mesh.indices.length / 3
  const buffer = new ArrayBuffer(84 + triCount * 50)
  const view = new DataView(buffer)

  const header = 'Trace2Grid gridfinity bin (units: mm)'
  for (let i = 0; i < Math.min(80, header.length); i++) {
    view.setUint8(i, header.charCodeAt(i))
  }
  view.setUint32(80, triCount, true)

  let offset = 84
  const p = mesh.positions
  for (let t = 0; t < triCount; t++) {
    const i0 = mesh.indices[t * 3] * 3
    const i1 = mesh.indices[t * 3 + 1] * 3
    const i2 = mesh.indices[t * 3 + 2] * 3

    const ux = p[i1] - p[i0]
    const uy = p[i1 + 1] - p[i0 + 1]
    const uz = p[i1 + 2] - p[i0 + 2]
    const vx = p[i2] - p[i0]
    const vy = p[i2 + 1] - p[i0 + 1]
    const vz = p[i2 + 2] - p[i0 + 2]
    let nx = uy * vz - uz * vy
    let ny = uz * vx - ux * vz
    let nz = ux * vy - uy * vx
    const len = Math.hypot(nx, ny, nz) || 1
    nx /= len
    ny /= len
    nz /= len

    view.setFloat32(offset, nx, true)
    view.setFloat32(offset + 4, ny, true)
    view.setFloat32(offset + 8, nz, true)
    offset += 12
    for (const idx of [i0, i1, i2]) {
      view.setFloat32(offset, p[idx], true)
      view.setFloat32(offset + 4, p[idx + 1], true)
      view.setFloat32(offset + 8, p[idx + 2], true)
      offset += 12
    }
    view.setUint16(offset, 0, true)
    offset += 2
  }
  return new Blob([buffer], { type: 'model/stl' })
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 5000)
}
