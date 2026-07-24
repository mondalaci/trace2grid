<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import {
  closestPointOnPolygon,
  polygonBBox,
  polygonsIntersect,
  transformPoints,
} from '../lib/geometry'
import { GRID_PITCH, MIN_WALL } from '../lib/gridfinity/spec'
import { polygonToSvgPoints, svgEventPoint } from '../lib/svg'
import { useProjectStore } from '../stores/project'
import type { Tool, Vec2 } from '../types'

const store = useProjectStore()
const svgEl = ref<SVGSVGElement | null>(null)
const notchMode = ref(false)

const MARGIN = 18

const viewBox = computed(
  () => `${-MARGIN} ${-MARGIN} ${store.binWidthMm + 2 * MARGIN} ${store.binDepthMm + 2 * MARGIN}`,
)

const gridLinesX = computed(() =>
  Array.from({ length: store.bin.gridX - 1 }, (_, i) => (i + 1) * GRID_PITCH),
)
const gridLinesY = computed(() =>
  Array.from({ length: store.bin.gridY - 1 }, (_, i) => (i + 1) * GRID_PITCH),
)

// --- dragging ---
type DragState =
  | { mode: 'move'; toolId: string; startPointer: Vec2; startPos: Vec2 }
  | { mode: 'rotate'; toolId: string }
  | { mode: 'notch'; toolId: string; notchId: string }
let drag: DragState | null = null

function toolById(id: string): Tool | undefined {
  return store.tools.find((t) => t.id === id)
}

function localPoint(tool: Tool, p: Vec2): Vec2 {
  // Inverse of the placement transform: un-translate, un-rotate.
  const rad = (-tool.rotationDeg * Math.PI) / 180
  const dx = p[0] - tool.x
  const dy = p[1] - tool.y
  return [dx * Math.cos(rad) - dy * Math.sin(rad), dx * Math.sin(rad) + dy * Math.cos(rad)]
}

function handleRadius(tool: Tool): number {
  const bb = polygonBBox(tool.points)
  return Math.max(bb.maxX - bb.minX, bb.maxY - bb.minY) / 2 + 10
}

function onToolDown(tool: Tool, event: PointerEvent) {
  store.selectedToolId = tool.id
  if (notchMode.value) {
    const p = localPoint(tool, svgEventPoint(svgEl.value!, event))
    const { point } = closestPointOnPolygon(tool.points, p)
    store.addNotch(tool.id, point)
    notchMode.value = false
    return
  }
  drag = {
    mode: 'move',
    toolId: tool.id,
    startPointer: svgEventPoint(svgEl.value!, event),
    startPos: [tool.x, tool.y],
  }
  ;(event.currentTarget as Element).setPointerCapture(event.pointerId)
}

function onRotateDown(tool: Tool, event: PointerEvent) {
  store.selectedToolId = tool.id
  drag = { mode: 'rotate', toolId: tool.id }
  ;(event.currentTarget as Element).setPointerCapture(event.pointerId)
}

function onNotchDown(tool: Tool, notchId: string, event: PointerEvent) {
  store.selectedToolId = tool.id
  drag = { mode: 'notch', toolId: tool.id, notchId }
  ;(event.currentTarget as Element).setPointerCapture(event.pointerId)
  event.stopPropagation()
}

function onPointerMove(event: PointerEvent) {
  if (!drag || !svgEl.value) return
  const tool = toolById(drag.toolId)
  if (!tool) return
  const p = svgEventPoint(svgEl.value, event)

  if (drag.mode === 'move') {
    tool.x = drag.startPos[0] + (p[0] - drag.startPointer[0])
    tool.y = drag.startPos[1] + (p[1] - drag.startPointer[1])
  } else if (drag.mode === 'rotate') {
    // Handle sits at local "up" (0, -r): pointer angle + 90° = tool rotation.
    const deg = (Math.atan2(p[1] - tool.y, p[0] - tool.x) * 180) / Math.PI + 90
    tool.rotationDeg = Math.round(((deg + 540) % 360) - 180)
  } else {
    const notchId = drag.notchId
    const notch = tool.notches.find((n) => n.id === notchId)
    if (notch) {
      const { point } = closestPointOnPolygon(tool.points, localPoint(tool, p))
      notch.x = point[0]
      notch.y = point[1]
    }
  }
}

function onPointerUp() {
  drag = null
}

function onKeyDown(event: KeyboardEvent) {
  if (event.key !== 'Delete' && event.key !== 'Backspace') return
  const target = event.target as HTMLElement
  if (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA') return
  if (store.selectedToolId) {
    store.removeTool(store.selectedToolId)
    event.preventDefault()
  }
}

onMounted(() => window.addEventListener('keydown', onKeyDown))
onBeforeUnmount(() => window.removeEventListener('keydown', onKeyDown))

// --- warnings ---
const warnings = computed<string[]>(() => {
  const result: string[] = []
  const placed = store.tools.map((tool) => ({
    tool,
    polys: (tool.offsetPolygons ?? [tool.points]).map((poly) =>
      transformPoints(poly, tool.x, tool.y, tool.rotationDeg),
    ),
  }))

  const minX = MIN_WALL
  const minY = MIN_WALL
  const maxX = store.binWidthMm - MIN_WALL
  const maxY = store.binDepthMm - MIN_WALL
  for (const { tool, polys } of placed) {
    const out = polys.some((poly) =>
      poly.some(([x, y]) => x < minX || y < minY || x > maxX || y > maxY),
    )
    if (out) result.push(`“${tool.name}” is too close to (or outside) the bin walls.`)
    for (const notch of tool.notches) {
      const [nx, ny] = transformPoints([[notch.x, notch.y]], tool.x, tool.y, tool.rotationDeg)[0]
      if (
        nx - notch.radius < minX ||
        ny - notch.radius < minY ||
        nx + notch.radius > maxX ||
        ny + notch.radius > maxY
      ) {
        result.push(`A notch on “${tool.name}” pokes through the bin walls.`)
      }
    }
  }
  for (let i = 0; i < placed.length; i++) {
    for (let j = i + 1; j < placed.length; j++) {
      const hit = placed[i].polys.some((a) => placed[j].polys.some((b) => polygonsIntersect(a, b)))
      if (hit) {
        result.push(`“${placed[i].tool.name}” and “${placed[j].tool.name}” overlap.`)
      }
    }
  }
  return result
})

const binHeightMm = computed(() => store.bin.heightUnits * 7)

function updateClearance(tool: Tool, event: Event) {
  const value = Number((event.target as HTMLInputElement).value)
  if (Number.isFinite(value)) store.setClearance(tool.id, Math.max(0, Math.min(5, value)))
}
</script>

<template>
  <div class="edit">
    <div class="canvas-wrap">
      <svg
        ref="svgEl"
        class="board"
        :viewBox="viewBox"
        @pointermove="onPointerMove"
        @pointerup="onPointerUp"
        @pointerdown.self="store.selectedToolId = null"
      >
        <!-- bin -->
        <rect
          x="0.25"
          y="0.25"
          :width="store.binWidthMm - 0.5"
          :height="store.binDepthMm - 0.5"
          rx="3.75"
          fill="#1a2029"
          stroke="#3a4250"
          stroke-width="0.6"
          @pointerdown="store.selectedToolId = null"
        />
        <g stroke="#2a3140" stroke-width="0.35">
          <line v-for="x in gridLinesX" :key="'x' + x" :x1="x" y1="0" :x2="x" :y2="store.binDepthMm" />
          <line v-for="y in gridLinesY" :key="'y' + y" x1="0" :y1="y" :x2="store.binWidthMm" :y2="y" />
        </g>

        <!-- tools -->
        <g
          v-for="tool in store.tools"
          :key="tool.id"
          :transform="`translate(${tool.x} ${tool.y}) rotate(${tool.rotationDeg})`"
          :class="{ selected: tool.id === store.selectedToolId }"
          style="touch-action: none"
        >
          <!-- clearance outline (actual pocket) -->
          <polygon
            v-for="(poly, pi) in tool.offsetPolygons ?? []"
            :key="pi"
            :points="polygonToSvgPoints(poly)"
            fill="rgba(76,141,255,0.10)"
            stroke="#38598f"
            stroke-width="0.35"
            stroke-dasharray="1.5 1.5"
            pointer-events="none"
          />
          <!-- notches -->
          <circle
            v-for="notch in tool.notches"
            :key="notch.id"
            :cx="notch.x"
            :cy="notch.y"
            :r="notch.radius"
            fill="rgba(255,180,84,0.25)"
            stroke="#ffb454"
            stroke-width="0.4"
            style="cursor: move; touch-action: none"
            @pointerdown="onNotchDown(tool, notch.id, $event)"
          />
          <!-- tool contour -->
          <polygon
            :points="polygonToSvgPoints(tool.points)"
            :fill="tool.id === store.selectedToolId ? 'rgba(76,141,255,0.45)' : 'rgba(122,139,166,0.35)'"
            :stroke="tool.id === store.selectedToolId ? '#4c8dff' : '#7a8ba6'"
            stroke-width="0.5"
            :style="{ cursor: notchMode && tool.id === store.selectedToolId ? 'crosshair' : 'move' }"
            @pointerdown="onToolDown(tool, $event)"
          />
          <!-- rotation handle -->
          <g v-if="tool.id === store.selectedToolId">
            <line
              x1="0"
              y1="0"
              x2="0"
              :y2="-handleRadius(tool)"
              stroke="#4c8dff"
              stroke-width="0.35"
              stroke-dasharray="1.2 1.2"
              pointer-events="none"
            />
            <circle
              cx="0"
              :cy="-handleRadius(tool)"
              r="3.2"
              fill="#4c8dff"
              stroke="#fff"
              stroke-width="0.5"
              style="cursor: grab; touch-action: none"
              @pointerdown="onRotateDown(tool, $event)"
            />
          </g>
        </g>
      </svg>
    </div>

    <aside class="sidebar">
      <div class="panel">
        <h3>Bin</h3>
        <div class="field-row">
          <div class="field">
            <label>Width (units)</label>
            <input v-model.number="store.bin.gridX" type="number" min="1" max="10" />
          </div>
          <div class="field">
            <label>Depth (units)</label>
            <input v-model.number="store.bin.gridY" type="number" min="1" max="10" />
          </div>
          <div class="field">
            <label>Height (units)</label>
            <input
              :value="store.bin.heightUnits"
              type="number"
              min="2"
              max="15"
              @change="store.setHeightUnits(Number(($event.target as HTMLInputElement).value) || 2)"
            />
          </div>
        </div>
        <p class="muted dims">
          {{ (store.binWidthMm - 0.5).toFixed(1) }} × {{ (store.binDepthMm - 0.5).toFixed(1) }} ×
          {{ binHeightMm }} mm{{ store.bin.stackingLip ? ' + lip' : '' }}
        </p>
        <label class="check"><input v-model="store.bin.stackingLip" type="checkbox" /> Stacking lip</label>
        <label class="check"><input v-model="store.bin.magnetHoles" type="checkbox" /> Magnet holes (6.5 × 2.4 mm)</label>
        <button @click="store.suggestBinSize()">Grow bin to fit tools</button>
      </div>

      <div v-if="store.selectedTool" class="panel">
        <h3>
          <input v-model="store.selectedTool.name" class="name-input" type="text" />
        </h3>
        <div class="field-row">
          <div class="field">
            <label>Rotation (°)</label>
            <input v-model.number="store.selectedTool.rotationDeg" type="number" step="5" />
          </div>
          <div class="field">
            <label>Clearance (mm)</label>
            <input
              :value="store.selectedTool.clearance"
              type="number"
              min="0"
              max="5"
              step="0.1"
              @change="updateClearance(store.selectedTool, $event)"
            />
          </div>
          <div class="field">
            <label>Depth (mm)</label>
            <input
              v-model.number="store.selectedTool.pocketDepth"
              type="number"
              min="1"
              :max="store.maxDepth"
              step="0.5"
            />
          </div>
        </div>
        <button :class="{ primary: notchMode }" @click="notchMode = !notchMode">
          {{ notchMode ? 'Click on the contour to place the notch…' : 'Add finger notch' }}
        </button>
        <ul v-if="store.selectedTool.notches.length" class="notch-list">
          <li v-for="(notch, i) in store.selectedTool.notches" :key="notch.id">
            <span>Notch {{ i + 1 }}</span>
            <label>r</label>
            <input v-model.number="notch.radius" type="number" min="3" max="25" step="1" />
            <label>depth</label>
            <input v-model.number="notch.depth" type="number" min="1" :max="store.maxDepth" step="0.5" />
            <button class="danger small" @click="store.removeNotch(store.selectedTool!.id, notch.id)">✕</button>
          </li>
        </ul>
        <div class="actions">
          <button @click="store.duplicateTool(store.selectedTool.id)">Duplicate</button>
          <button class="danger" @click="store.removeTool(store.selectedTool.id)">Delete</button>
        </div>
      </div>
      <div v-else class="panel muted">
        Select a tool to edit it. Drag to move, use the round handle to rotate, Delete key removes it.
      </div>

      <div v-if="warnings.length" class="panel">
        <h3 class="warning">Warnings</h3>
        <ul class="warn-list">
          <li v-for="(w, i) in warnings" :key="i" class="warning">{{ w }}</li>
        </ul>
      </div>

      <div class="panel nav">
        <button @click="store.step = 'capture'">← Scan more tools</button>
        <button class="primary" :disabled="store.tools.length === 0" @click="store.step = 'preview'">
          3D preview & export →
        </button>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.edit {
  height: 100%;
  display: flex;
  gap: 14px;
  padding: 14px;
}

.canvas-wrap {
  flex: 1;
  min-width: 0;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  display: flex;
}

.board {
  width: 100%;
  height: 100%;
  touch-action: none;
}

.sidebar {
  width: 340px;
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

.dims {
  margin: 0;
  font-size: 12.5px;
}

.check {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.name-input {
  background: transparent;
  border: none;
  border-bottom: 1px dashed var(--border);
  color: var(--text);
  font: inherit;
  font-weight: 600;
  width: 100%;
  padding: 2px 0;
}

.notch-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.notch-list li {
  display: flex;
  align-items: center;
  gap: 6px;
}

.notch-list input {
  width: 58px;
}

.small {
  padding: 3px 8px;
}

.warn-list {
  margin: 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.actions {
  display: flex;
  gap: 8px;
}

.nav {
  flex-direction: row;
  justify-content: space-between;
}

@media (max-width: 760px) {
  .edit {
    flex-direction: column;
  }

  .canvas-wrap {
    min-height: 45vh;
  }

  .sidebar {
    width: 100%;
  }
}
</style>
