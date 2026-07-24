<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { polygonCentroid } from '../lib/geometry'
import { loadOpenCV } from '../lib/scan/opencv'
import {
  classifyPaper,
  detectPaperQuad,
  PAPER_SIZES,
  paperById,
  rectifyPaper,
} from '../lib/scan/paper'
import { extractToolContours, type ExtractedContour } from '../lib/scan/contours'
import { polygonToSvgPoints, svgEventPoint } from '../lib/svg'
import { useProjectStore } from '../stores/project'
import type { PaperSizeId, Vec2 } from '../types'

const PX_PER_MM = 4
const store = useProjectStore()

type Phase = 'idle' | 'corners' | 'review'
const phase = ref<Phase>('idle')
const busy = ref<string | null>(null)
const error = ref<string | null>(null)

// --- corners phase state ---
const photoCanvas = ref<HTMLCanvasElement | null>(null)
const photoUrl = ref('')
const photoW = ref(0)
const photoH = ref(0)
const corners = ref<[Vec2, Vec2, Vec2, Vec2] | null>(null)
const autoDetected = ref(false)
const paperChoice = ref<'auto' | PaperSizeId>('auto')
const cornersSvg = ref<SVGSVGElement | null>(null)
let draggingCorner = -1

// --- review phase state ---
const rectifiedUrl = ref('')
const rectifiedW = ref(0)
const rectifiedH = ref(0)
const rectifiedCanvas = ref<HTMLCanvasElement | null>(null)
const detected = ref<ExtractedContour[]>([])
const kept = ref<boolean[]>([])
const sensitivity = ref(0)
let sensitivityTimer: ReturnType<typeof setTimeout> | undefined

const autoClass = computed(() => (corners.value ? classifyPaper(corners.value) : null))
const effectivePaperId = computed<PaperSizeId>(() =>
  paperChoice.value === 'auto' ? (autoClass.value?.sizeId ?? 'a4') : paperChoice.value,
)
const keptCount = computed(() => kept.value.filter(Boolean).length)

async function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return
  error.value = null
  busy.value = 'Loading photo…'
  try {
    const bitmap = await createImageBitmap(file)
    const maxDim = 1800
    const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height))
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(bitmap.width * scale)
    canvas.height = Math.round(bitmap.height * scale)
    canvas.getContext('2d')!.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    bitmap.close()

    photoCanvas.value = canvas
    photoUrl.value = canvas.toDataURL('image/jpeg', 0.85)
    photoW.value = canvas.width
    photoH.value = canvas.height

    busy.value = 'Loading OpenCV (first time only)…'
    const cv = await loadOpenCV()
    busy.value = 'Detecting paper…'
    await new Promise((r) => setTimeout(r)) // let the status paint
    const quad = detectPaperQuad(cv, canvas)
    if (quad) {
      corners.value = quad
      autoDetected.value = true
    } else {
      const inX = canvas.width * 0.1
      const inY = canvas.height * 0.1
      corners.value = [
        [inX, inY],
        [canvas.width - inX, inY],
        [canvas.width - inX, canvas.height - inY],
        [inX, canvas.height - inY],
      ]
      autoDetected.value = false
    }
    phase.value = 'corners'
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = null
  }
}

function onCornerDown(index: number, event: PointerEvent) {
  draggingCorner = index
  ;(event.target as Element).setPointerCapture(event.pointerId)
}

function onCornersMove(event: PointerEvent) {
  if (draggingCorner < 0 || !cornersSvg.value || !corners.value) return
  const [x, y] = svgEventPoint(cornersSvg.value, event)
  corners.value[draggingCorner] = [
    Math.max(0, Math.min(photoW.value, x)),
    Math.max(0, Math.min(photoH.value, y)),
  ]
}

function onCornersUp() {
  draggingCorner = -1
}

async function confirmCorners() {
  if (!photoCanvas.value || !corners.value) return
  error.value = null
  busy.value = 'Rectifying and finding tools…'
  try {
    const cv = await loadOpenCV()
    await new Promise((r) => setTimeout(r))
    const paper = paperById(effectivePaperId.value)
    const landscape = autoClass.value?.landscape ?? false
    const canvas = rectifyPaper(cv, photoCanvas.value, corners.value, paper, landscape, PX_PER_MM)
    rectifiedCanvas.value = canvas
    rectifiedUrl.value = canvas.toDataURL('image/jpeg', 0.85)
    rectifiedW.value = canvas.width
    rectifiedH.value = canvas.height
    sensitivity.value = 0
    runExtraction()
    phase.value = 'review'
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = null
  }
}

async function runExtraction() {
  if (!rectifiedCanvas.value) return
  const cv = await loadOpenCV()
  detected.value = extractToolContours(cv, rectifiedCanvas.value, {
    pxPerMm: PX_PER_MM,
    sensitivity: sensitivity.value,
  })
  kept.value = detected.value.map(() => true)
}

watch(sensitivity, () => {
  clearTimeout(sensitivityTimer)
  sensitivityTimer = setTimeout(runExtraction, 250)
})

function toggleKeep(index: number) {
  kept.value[index] = !kept.value[index]
}

function addToProject() {
  const contours: Vec2[][] = []
  detected.value.forEach((contour, i) => {
    if (!kept.value[i]) return
    const mm: Vec2[] = contour.pointsPx.map(([x, y]) => [x / PX_PER_MM, y / PX_PER_MM])
    const [cx, cy] = polygonCentroid(mm)
    contours.push(mm.map(([x, y]) => [x - cx, y - cy]))
  })
  store.addTools(contours)
  store.step = 'edit'
}

function restart() {
  phase.value = 'idle'
  photoUrl.value = ''
  rectifiedUrl.value = ''
  detected.value = []
  error.value = null
}
</script>

<template>
  <div class="capture">
    <!-- Phase: idle -->
    <div v-if="phase === 'idle'" class="idle">
      <div class="hero panel">
        <h2>Scan your tools</h2>
        <ol class="muted">
          <li>Lay your tools flat on a blank sheet of paper (A4 or US Letter). Colored or dark tools work best; pale gray tools on white paper are hardest to detect.</li>
          <li>Put the paper on a contrasting (darker) surface, with even light and no harsh shadows.</li>
          <li>Take the photo from straight above, with the whole sheet visible.</li>
          <li>
            Shoot from as far away as practical and zoom in — distance reduces perspective
            distortion from tool height. A telephoto lens (e.g. 2–3× on your phone) works best.
          </li>
        </ol>
        <label class="file-button primary">
          <input type="file" accept="image/*" capture="environment" @change="onFileChange" />
          Take / choose photo
        </label>
        <p v-if="store.tools.length" class="muted">
          {{ store.tools.length }} tool(s) already in the bin — scanning again adds more.
        </p>
      </div>
    </div>

    <!-- Phase: corner adjustment -->
    <div v-else-if="phase === 'corners'" class="stage">
      <div class="canvas-wrap">
        <svg
          ref="cornersSvg"
          class="fit"
          :viewBox="`0 0 ${photoW} ${photoH}`"
          @pointermove="onCornersMove"
          @pointerup="onCornersUp"
        >
          <image :href="photoUrl" :width="photoW" :height="photoH" />
          <polygon
            v-if="corners"
            :points="polygonToSvgPoints(corners)"
            fill="rgba(76,141,255,0.15)"
            stroke="#4c8dff"
            :stroke-width="photoW / 400"
          />
          <circle
            v-for="(c, i) in corners"
            :key="i"
            :cx="c[0]"
            :cy="c[1]"
            :r="photoW / 45"
            fill="rgba(76,141,255,0.35)"
            stroke="#fff"
            :stroke-width="photoW / 500"
            style="cursor: grab; touch-action: none"
            @pointerdown="onCornerDown(i, $event)"
          />
        </svg>
      </div>
      <aside class="sidebar">
        <div class="panel">
          <h3>Paper outline</h3>
          <p class="muted">
            {{ autoDetected ? 'Paper detected automatically — adjust the corners if needed.'
              : 'Could not auto-detect the sheet. Drag the four corners onto the paper edges.' }}
          </p>
          <div class="field">
            <label>Paper size</label>
            <select v-model="paperChoice">
              <option value="auto">
                Auto{{ autoClass ? ` — ${paperById(autoClass.sizeId).name}` : '' }}
              </option>
              <option v-for="p in PAPER_SIZES" :key="p.id" :value="p.id">{{ p.name }}</option>
            </select>
          </div>
          <div class="actions">
            <button @click="restart">Back</button>
            <button class="primary" :disabled="!!busy" @click="confirmCorners">
              Extract tools
            </button>
          </div>
        </div>
      </aside>
    </div>

    <!-- Phase: contour review -->
    <div v-else class="stage">
      <div class="canvas-wrap">
        <svg class="fit" :viewBox="`0 0 ${rectifiedW} ${rectifiedH}`">
          <image :href="rectifiedUrl" :width="rectifiedW" :height="rectifiedH" />
          <polygon
            v-for="(contour, i) in detected"
            :key="i"
            :points="polygonToSvgPoints(contour.pointsPx)"
            :fill="kept[i] ? 'rgba(63,185,112,0.30)' : 'rgba(255,107,107,0.15)'"
            :stroke="kept[i] ? '#3fb970' : '#ff6b6b'"
            :stroke-width="rectifiedW / 400"
            :stroke-dasharray="kept[i] ? 'none' : '8 6'"
            style="cursor: pointer"
            @click="toggleKeep(i)"
          />
        </svg>
      </div>
      <aside class="sidebar">
        <div class="panel">
          <h3>Detected tools</h3>
          <p class="muted">
            Click a contour to keep (green) or discard (red).
            {{ detected.length }} found, {{ keptCount }} kept.
          </p>
          <div class="field">
            <label>Detection sensitivity</label>
            <input v-model.number="sensitivity" type="range" min="-60" max="60" step="5" />
            <span class="muted">{{ sensitivity > 0 ? '+' : '' }}{{ sensitivity }} (raise if parts of tools are missed, lower if shadows are picked up)</span>
          </div>
          <ul class="contour-list">
            <li v-for="(contour, i) in detected" :key="i" :class="{ off: !kept[i] }">
              <input :id="`keep-${i}`" v-model="kept[i]" type="checkbox" />
              <label :for="`keep-${i}`">
                Contour {{ i + 1 }} — {{ contour.areaMm2.toFixed(0) }} mm²
              </label>
            </li>
          </ul>
          <div class="actions">
            <button @click="phase = 'corners'">Back</button>
            <button class="primary" :disabled="keptCount === 0" @click="addToProject">
              Add {{ keptCount }} tool(s) to bin
            </button>
          </div>
        </div>
      </aside>
    </div>

    <div v-if="busy" class="status">{{ busy }}</div>
    <div v-if="error" class="status error">{{ error }}</div>
  </div>
</template>

<style scoped>
.capture {
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
}

.idle {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.hero {
  max-width: 560px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 28px;
}

.hero ol {
  margin: 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.file-button {
  display: inline-block;
  text-align: center;
  background: var(--accent);
  border-radius: 8px;
  padding: 12px 18px;
  font-weight: 600;
  color: #fff;
  cursor: pointer;
}

.file-button input {
  display: none;
}

.stage {
  flex: 1;
  min-height: 0;
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
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.fit {
  max-width: 100%;
  max-height: 100%;
  width: 100%;
  height: 100%;
}

.sidebar {
  width: 320px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
}

.sidebar .panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.contour-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 220px;
  overflow-y: auto;
}

.contour-list li {
  display: flex;
  align-items: center;
  gap: 8px;
}

.contour-list li.off label {
  text-decoration: line-through;
  opacity: 0.6;
}

.actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.status {
  position: absolute;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 8px 18px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
}

.status.error {
  color: var(--danger);
  border-color: var(--danger);
}

@media (max-width: 760px) {
  .stage {
    flex-direction: column;
  }

  .sidebar {
    width: 100%;
  }
}
</style>
