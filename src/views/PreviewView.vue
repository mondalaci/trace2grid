<script setup lang="ts">
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { downloadBlob, meshToBinarySTL } from '../lib/export/stl'
import { exportLayoutPdf } from '../lib/export/pdf'
import { transformPoints } from '../lib/geometry'
import { buildBin } from '../lib/gridfinity/client'
import { GRID_PITCH } from '../lib/gridfinity/spec'
import { PAPER_SIZES, paperById } from '../lib/scan/paper'
import { useProjectStore } from '../stores/project'
import type { MeshData, PaperSizeId } from '../types'

const store = useProjectStore()
const container = ref<HTMLDivElement | null>(null)
const building = ref(false)
const error = ref<string | null>(null)
const triangleCount = ref(0)
const pdfPaper = ref<PaperSizeId>('a4')

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let controls: OrbitControls | null = null
let binMesh: THREE.Mesh | null = null
let animationId = 0
let resizeObserver: ResizeObserver | null = null
let lastMesh: MeshData | null = null
let rebuildTimer: ReturnType<typeof setTimeout> | undefined
let rebuildQueued = false

const binHeightMm = computed(() => store.bin.heightUnits * 7)

function initScene() {
  const el = container.value!
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(el.clientWidth, el.clientHeight)
  el.appendChild(renderer.domElement)

  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x6b7280)

  camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 1, 5000)
  resetCamera()

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.1
  // Middle-drag pans (wheel already zooms); right-drag keeps panning too.
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.PAN,
    RIGHT: THREE.MOUSE.PAN,
  }

  scene.add(new THREE.HemisphereLight(0xdde4ee, 0x30363f, 1.1))
  const sun = new THREE.DirectionalLight(0xffffff, 1.6)
  sun.position.set(120, 220, 160)
  scene.add(sun)
  const fill = new THREE.DirectionalLight(0xd8dde5, 0.4)
  fill.position.set(-150, 80, -120)
  scene.add(fill)

  const grid = new THREE.GridHelper(GRID_PITCH * 12, 12, 0x5a626d, 0x4a515b)
  grid.position.y = -0.05
  scene.add(grid)

  resizeObserver = new ResizeObserver(() => {
    if (!renderer || !camera || !el.clientWidth || !el.clientHeight) return
    camera.aspect = el.clientWidth / el.clientHeight
    camera.updateProjectionMatrix()
    renderer.setSize(el.clientWidth, el.clientHeight)
  })
  resizeObserver.observe(el)

  const animate = () => {
    animationId = requestAnimationFrame(animate)
    controls?.update()
    if (renderer && scene && camera) renderer.render(scene, camera)
  }
  animate()
}

function resetCamera() {
  if (!camera || !controls) {
    if (!camera) return
  }
  const spread = Math.max(store.binWidthMm, store.binDepthMm, binHeightMm.value)
  camera!.position.set(spread * 1.1, spread * 1.2, spread * 1.5)
  camera!.lookAt(0, binHeightMm.value / 2, 0)
  if (controls) {
    controls.target.set(0, binHeightMm.value / 3, 0)
    controls.update()
  }
}

async function rebuild() {
  if (building.value) {
    rebuildQueued = true
    return
  }
  building.value = true
  error.value = null
  try {
    const mesh = await buildBin(store.toBuildRequest())
    lastMesh = mesh
    triangleCount.value = mesh.indices.length / 3

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3))
    geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1))
    const flat = geometry.toNonIndexed()
    geometry.dispose()
    flat.computeVertexNormals()

    if (binMesh) {
      scene?.remove(binMesh)
      binMesh.geometry.dispose()
      ;(binMesh.material as THREE.Material).dispose()
    }
    const material = new THREE.MeshStandardMaterial({
      color: 0x9aa1a9,
      roughness: 0.55,
      metalness: 0.05,
    })
    binMesh = new THREE.Mesh(flat, material)
    // Geometry is z-up (CAD convention); three.js scenes are y-up.
    binMesh.rotation.x = -Math.PI / 2
    scene?.add(binMesh)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    building.value = false
    if (rebuildQueued) {
      rebuildQueued = false
      rebuild()
    }
  }
}

watch(
  () => [store.bin, store.tools],
  () => {
    clearTimeout(rebuildTimer)
    rebuildTimer = setTimeout(rebuild, 500)
  },
  { deep: true },
)

onMounted(() => {
  initScene()
  rebuild()
})

onBeforeUnmount(() => {
  cancelAnimationFrame(animationId)
  clearTimeout(rebuildTimer)
  resizeObserver?.disconnect()
  controls?.dispose()
  if (binMesh) {
    binMesh.geometry.dispose()
    ;(binMesh.material as THREE.Material).dispose()
  }
  renderer?.dispose()
  renderer?.domElement.remove()
  renderer = null
  scene = null
})

function exportStl() {
  if (!lastMesh) return
  const name = `gridfinity-${store.bin.gridX}x${store.bin.gridY}x${store.bin.heightUnits}.stl`
  downloadBlob(meshToBinarySTL(lastMesh), name)
}

async function exportPdf() {
  const blob = await exportLayoutPdf({
    binW: store.binWidthMm,
    binH: store.binDepthMm,
    gridPitch: GRID_PITCH,
    paper: paperById(pdfPaper.value),
    tools: store.tools.map((tool) => ({
      name: tool.name,
      polygons: (tool.offsetPolygons ?? [tool.points]).map((poly) =>
        transformPoints(poly, tool.x, tool.y, tool.rotationDeg),
      ),
      notches: tool.notches.map((notch) => {
        const [x, y] = transformPoints([[notch.x, notch.y]], tool.x, tool.y, tool.rotationDeg)[0]
        return { x, y, radius: notch.radius }
      }),
    })),
  })
  downloadBlob(blob, `gridfinity-layout-${store.bin.gridX}x${store.bin.gridY}.pdf`)
}
</script>

<template>
  <div class="preview">
    <div ref="container" class="viewport">
      <div v-if="building" class="overlay">Building geometry…</div>
      <div v-if="error" class="overlay error">{{ error }}</div>
    </div>
    <aside class="sidebar">
      <div class="panel">
        <h3>Bin</h3>
        <p class="muted">
          {{ store.bin.gridX }} × {{ store.bin.gridY }} units ·
          {{ (store.binWidthMm - 0.5).toFixed(1) }} × {{ (store.binDepthMm - 0.5).toFixed(1) }} ×
          {{ binHeightMm }} mm · {{ store.tools.length }} tool(s) ·
          {{ triangleCount.toLocaleString() }} triangles
        </p>
        <div class="field">
          <label>Height (7 mm units)</label>
          <input
            :value="store.bin.heightUnits"
            type="number"
            min="2"
            max="15"
            @change="store.setHeightUnits(Number(($event.target as HTMLInputElement).value) || 2)"
          />
        </div>
        <label class="check"><input v-model="store.bin.stackingLip" type="checkbox" /> Stacking lip</label>
        <label class="check"><input v-model="store.bin.magnetHoles" type="checkbox" /> Magnet holes</label>
        <button @click="resetCamera">Reset camera</button>
      </div>

      <div class="panel">
        <h3>Export</h3>
        <button class="primary" :disabled="building || !!error" @click="exportStl">
          Download STL (3D print)
        </button>
        <div class="field">
          <label>PDF paper size</label>
          <select v-model="pdfPaper">
            <option v-for="p in PAPER_SIZES" :key="p.id" :value="p.id">{{ p.name }}</option>
          </select>
        </div>
        <button @click="exportPdf">Download 1:1 PDF (paper check)</button>
        <p class="muted note">
          Print the PDF at 100% scale and lay your tools on it to verify the fit before printing in 3D.
        </p>
      </div>

      <div class="panel nav">
        <button @click="store.step = 'edit'">← Back to arrange</button>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.preview {
  height: 100%;
  display: flex;
  gap: 14px;
  padding: 14px;
}

.viewport {
  flex: 1;
  min-width: 0;
  position: relative;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}

.viewport :deep(canvas) {
  display: block;
}

.overlay {
  position: absolute;
  top: 12px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 6px 16px;
  pointer-events: none;
}

.overlay.error {
  color: var(--danger);
  border-color: var(--danger);
}

.sidebar {
  width: 320px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
}

.panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.check {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.note {
  font-size: 12px;
  margin: 0;
}

@media (max-width: 760px) {
  .preview {
    flex-direction: column;
  }

  .viewport {
    min-height: 50vh;
  }

  .sidebar {
    width: 100%;
  }
}
</style>
