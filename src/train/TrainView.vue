<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { extractToolContours, type ExtractedContour } from '../lib/scan/contours'
import { loadOpenCV } from '../lib/scan/opencv'
import {
  classifyPaper,
  detectPaperQuad,
  PAPER_SIZES,
  paperById,
  rectifyPaper,
} from '../lib/scan/paper'
import { polygonToSvgPoints, svgEventPoint } from '../lib/svg'
import type { PaperSizeId, Vec2 } from '../types'
import type { TrainingAnnotation, TruthPolygon } from './format'

const PX_PER_MM = 4
const MAX_UNDO = 25

const photos = ref<string[]>([])
const selectedPhoto = ref('')
const phase = ref<'corners' | 'regions'>('corners')
const busy = ref<string | null>(null)
const error = ref<string | null>(null)

// --- corners phase ---
const photoCanvas = ref<HTMLCanvasElement | null>(null)
const photoUrl = ref('')
const photoW = ref(0)
const photoH = ref(0)
const corners = ref<[Vec2, Vec2, Vec2, Vec2] | null>(null)
const paperChoice = ref<'auto' | PaperSizeId>('auto')
const cornersSvg = ref<SVGSVGElement | null>(null)
const cornersViewBox = reactive({ x: 0, y: 0, w: 100, h: 100 })
let draggingCorner = -1

const autoClass = computed(() => (corners.value ? classifyPaper(corners.value) : null))
const effectivePaperId = computed<PaperSizeId>(() =>
  paperChoice.value === 'auto' ? (autoClass.value?.sizeId ?? 'a4') : paperChoice.value,
)

// --- regions phase ---
const rectifiedCanvas = ref<HTMLCanvasElement | null>(null)
const rectifiedUrl = ref('')
const rectifiedW = ref(0)
const rectifiedH = ref(0)
const regionsSvg = ref<SVGSVGElement | null>(null)
const viewBox = reactive({ x: 0, y: 0, w: 100, h: 100 })
const lasso = ref<{ mode: 'add' | 'remove'; pointsPx: Vec2[] } | null>(null)
const hoverPoint = ref<Vec2 | null>(null)
let panLast: Vec2 | null = null
let lassoButtonDown = false

// Ground truth lives in a raster mask; the outline shown to the user (and
// persisted) is re-vectorized from it after every edit.
const truthCanvas = document.createElement('canvas')
let truthCtx: CanvasRenderingContext2D | null = null
let seededFor = ''
const truthPolysPx = ref<{ outer: Vec2[]; holes: Vec2[][] }[]>([])
const undoStack: ImageData[] = []
const undoDepth = ref(0)
const redoStack: ImageData[] = []
const redoDepth = ref(0)
let savedTruth: TruthPolygon[] | null = null
/** Bumped on photo switch and every outline rebuild so stale async work is ignored. */
let truthEpoch = 0

const sensitivity = ref(0)
const showDetection = ref(false)
const detected = ref<ExtractedContour[]>([])
const iou = ref<number | null>(null)
let extractTimer: ReturnType<typeof setTimeout> | undefined

// --- persistence ---
const saveState = ref<'saved' | 'dirty' | 'saving'>('saved')
let saveTimer: ReturnType<typeof setTimeout> | undefined
let loading = false

const viewBoxAttr = computed(() => `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`)
const strokePx = computed(() => viewBox.w / 300)
const cornersViewBoxAttr = computed(
  () => `${cornersViewBox.x} ${cornersViewBox.y} ${cornersViewBox.w} ${cornersViewBox.h}`,
)

type ViewBox = { x: number; y: number; w: number; h: number }

function zoomAt(vb: ViewBox, svg: SVGSVGElement, event: WheelEvent, fullW: number) {
  const factor = event.deltaY > 0 ? 1.2 : 1 / 1.2
  const newW = Math.max(fullW / 50, Math.min(fullW * 2, vb.w * factor))
  const scale = newW / vb.w
  const [ax, ay] = svgEventPoint(svg, event)
  vb.x = ax - (ax - vb.x) * scale
  vb.y = ay - (ay - vb.y) * scale
  vb.w = newW
  vb.h *= scale
}

function panBy(vb: ViewBox, svg: SVGSVGElement, event: PointerEvent, last: Vec2): Vec2 {
  const rect = svg.getBoundingClientRect()
  vb.x -= ((event.clientX - last[0]) * vb.w) / rect.width
  vb.y -= ((event.clientY - last[1]) * vb.h) / rect.height
  return [event.clientX, event.clientY]
}

function truthPathD(poly: { outer: Vec2[]; holes: Vec2[][] }): string {
  const ring = (pts: Vec2[]) =>
    pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join('') + 'Z'
  return ring(poly.outer) + poly.holes.map(ring).join('')
}

onMounted(async () => {
  window.addEventListener('keyup', onKeyUp)
  window.addEventListener('keydown', onKeyDown)
  const res = await fetch('/__training/photos')
  photos.value = (await res.json()).photos
  const requested = new URLSearchParams(location.search).get('photo')
  const first = requested && photos.value.includes(requested) ? requested : photos.value[0]
  if (first) await selectPhoto(first)
})

async function selectPhoto(name: string) {
  await flushSave()
  selectedPhoto.value = name
  error.value = null
  busy.value = 'Loading photo…'
  loading = true
  seededFor = ''
  savedTruth = null
  // Invalidate in-flight outline refreshes / extractions from the previous photo
  // so they can't paint or autosave under the new name.
  truthEpoch++
  truthPolysPx.value = []
  detected.value = []
  iou.value = null
  undoStack.length = 0
  undoDepth.value = 0
  redoStack.length = 0
  redoDepth.value = 0
  lasso.value = null
  hoverPoint.value = null
  truthCtx = null
  truthCanvas.width = 0
  truthCanvas.height = 0
  try {
    const blob = await (await fetch(`/__training/photo/${encodeURIComponent(name)}`)).blob()
    const bitmap = await createImageBitmap(blob)
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
    Object.assign(cornersViewBox, { x: 0, y: 0, w: canvas.width, h: canvas.height })

    const dataRes = await fetch(`/__training/data/${encodeURIComponent(name)}`)
    const saved: TrainingAnnotation | null = dataRes.ok ? await dataRes.json() : null

    if (saved?.corners) {
      corners.value = saved.corners.map(([x, y]) => [x * canvas.width, y * canvas.height]) as [
        Vec2, Vec2, Vec2, Vec2,
      ]
      paperChoice.value = saved.paperSizeId
      // An empty truth means "never labeled" — reseed from detection then.
      savedTruth = saved.truth?.length ? saved.truth : null
    } else {
      busy.value = 'Detecting paper…'
      const cv = await loadOpenCV()
      await new Promise((r) => setTimeout(r))
      const quad = detectPaperQuad(cv, canvas)
      corners.value = quad ?? [
        [canvas.width * 0.1, canvas.height * 0.1],
        [canvas.width * 0.9, canvas.height * 0.1],
        [canvas.width * 0.9, canvas.height * 0.9],
        [canvas.width * 0.1, canvas.height * 0.9],
      ]
      paperChoice.value = 'auto'
    }
    phase.value = 'corners'
    saveState.value = saved ? 'saved' : 'dirty'
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = null
    loading = false
  }
}

// --- corner dragging, with the same zoom/pan as the labeling phase ---
function onCornerDown(index: number, event: PointerEvent) {
  draggingCorner = index
  ;(event.target as Element).setPointerCapture(event.pointerId)
}

function onCornersWheel(event: WheelEvent) {
  if (cornersSvg.value) zoomAt(cornersViewBox, cornersSvg.value, event, photoW.value)
}

function onCornersPointerDown(event: PointerEvent) {
  if (event.button === 1 && cornersSvg.value) {
    event.preventDefault()
    cornersSvg.value.setPointerCapture(event.pointerId)
    panLast = [event.clientX, event.clientY]
  }
}

function onCornersMove(event: PointerEvent) {
  if (!cornersSvg.value) return
  if (panLast) {
    panLast = panBy(cornersViewBox, cornersSvg.value, event, panLast)
    return
  }
  if (draggingCorner < 0 || !corners.value) return
  const [x, y] = svgEventPoint(cornersSvg.value, event)
  corners.value[draggingCorner] = [
    Math.max(0, Math.min(photoW.value, x)),
    Math.max(0, Math.min(photoH.value, y)),
  ]
}

function onCornersUp() {
  draggingCorner = -1
  panLast = null
}

async function redetectCorners() {
  if (!photoCanvas.value) return
  busy.value = 'Detecting paper…'
  try {
    const cv = await loadOpenCV()
    await new Promise((r) => setTimeout(r))
    const quad = detectPaperQuad(cv, photoCanvas.value)
    if (quad) corners.value = quad
    else error.value = 'Auto-detection found no paper'
  } finally {
    busy.value = null
  }
}

async function toRegions() {
  if (!photoCanvas.value || !corners.value) return
  busy.value = 'Straightening the photo and detecting tools…'
  error.value = null
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
    Object.assign(viewBox, { x: 0, y: 0, w: canvas.width, h: canvas.height })
    await runExtraction()

    const seedKey = `${selectedPhoto.value}:${canvas.width}x${canvas.height}`
    if (seededFor !== seedKey) {
      await seedTruth(savedTruth)
      seededFor = seedKey
    }
    phase.value = 'regions'
    computeIoU()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = null
  }
}

// --- truth mask management ---
function fillRings(ctx: CanvasRenderingContext2D, rings: Vec2[][], erase = false) {
  const path = new Path2D()
  for (const ring of rings) {
    ring.forEach(([x, y], i) => (i ? path.lineTo(x, y) : path.moveTo(x, y)))
    path.closePath()
  }
  ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
  ctx.fillStyle = '#fff'
  ctx.fill(path, 'evenodd')
  ctx.globalCompositeOperation = 'source-over'
}

async function seedTruth(fromSaved: TruthPolygon[] | null) {
  const epoch = ++truthEpoch
  truthCanvas.width = rectifiedW.value
  truthCanvas.height = rectifiedH.value
  truthCtx = truthCanvas.getContext('2d', { willReadFrequently: true })!
  undoStack.length = 0
  undoDepth.value = 0
  redoStack.length = 0
  redoDepth.value = 0
  if (fromSaved) {
    for (const poly of fromSaved) {
      fillRings(truthCtx, [
        poly.outerMm.map(([x, y]) => [x * PX_PER_MM, y * PX_PER_MM] as Vec2),
        ...poly.holesMm.map((h) => h.map(([x, y]) => [x * PX_PER_MM, y * PX_PER_MM] as Vec2)),
      ])
    }
  } else {
    for (const contour of detected.value) fillRings(truthCtx, [contour.pointsPx])
  }
  await refreshTruthOutline(epoch)
}

async function refreshTruthOutline(existingEpoch?: number) {
  if (!truthCtx) return
  const epoch = existingEpoch ?? ++truthEpoch
  try {
    await refreshTruthOutlineInner(epoch)
  } catch (e) {
    if (epoch === truthEpoch) {
      error.value = `Outline update failed: ${e instanceof Error ? e.message : e}`
    }
  }
}

async function refreshTruthOutlineInner(epoch: number) {
  // Snapshot the mask up front so a later seedTruth resize of truthCanvas
  // can't make this job read the next photo's pixels.
  const w = truthCanvas.width
  const h = truthCanvas.height
  if (!truthCtx || w === 0 || h === 0) return
  const img = truthCtx.getImageData(0, 0, w, h)
  const maskData = new Uint8Array(w * h)
  for (let i = 0; i < w * h; i++) maskData[i] = img.data[i * 4 + 3] > 127 ? 255 : 0

  const cv = await loadOpenCV()
  if (epoch !== truthEpoch) return

  const mask = cv.matFromArray(h, w, cv.CV_8UC1, maskData as unknown as number[])
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  try {
    cv.findContours(mask, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    const rings: { pts: Vec2[]; parent: number }[] = []
    for (let i = 0; i < contours.size(); i++) {
      const approx = new cv.Mat()
      try {
        cv.approxPolyDP(contours.get(i), approx, 0.3 * PX_PER_MM, true)
        const pts: Vec2[] = []
        for (let p = 0; p < approx.rows; p++) {
          pts.push([approx.data32S[p * 2], approx.data32S[p * 2 + 1]])
        }
        rings.push({ pts, parent: hierarchy.data32S[i * 4 + 3] })
      } finally {
        approx.delete()
      }
    }
    const polys: { outer: Vec2[]; holes: Vec2[][] }[] = []
    const outerIndex = new Map<number, number>()
    rings.forEach((ring, i) => {
      if (ring.parent === -1 && ring.pts.length >= 3) {
        outerIndex.set(i, polys.length)
        polys.push({ outer: ring.pts, holes: [] })
      }
    })
    rings.forEach((ring) => {
      if (ring.parent !== -1 && ring.pts.length >= 3) {
        const target = outerIndex.get(ring.parent)
        if (target !== undefined) polys[target].holes.push(ring.pts)
      }
    })
    if (epoch !== truthEpoch) return
    truthPolysPx.value = polys
  } finally {
    mask.delete()
    contours.delete()
    hierarchy.delete()
  }
  if (epoch !== truthEpoch) return
  scheduleSave()
  computeIoU()
}

function pushUndo() {
  if (!truthCtx) return
  undoStack.push(truthCtx.getImageData(0, 0, truthCanvas.width, truthCanvas.height))
  if (undoStack.length > MAX_UNDO) undoStack.shift()
  undoDepth.value = undoStack.length
  // A new edit branches the history — drop any redo trail.
  redoStack.length = 0
  redoDepth.value = 0
}

function applyLasso(mode: 'add' | 'remove', pointsPx: Vec2[]) {
  if (!truthCtx) return
  pushUndo()
  fillRings(truthCtx, [pointsPx], mode === 'remove')
  refreshTruthOutline()
}

function undoEdit() {
  const snapshot = undoStack.pop()
  if (!snapshot || !truthCtx) return
  redoStack.push(truthCtx.getImageData(0, 0, truthCanvas.width, truthCanvas.height))
  redoDepth.value = redoStack.length
  undoDepth.value = undoStack.length
  truthCtx.putImageData(snapshot, 0, 0)
  refreshTruthOutline()
}

function redoEdit() {
  const snapshot = redoStack.pop()
  if (!snapshot || !truthCtx) return
  undoStack.push(truthCtx.getImageData(0, 0, truthCanvas.width, truthCanvas.height))
  if (undoStack.length > MAX_UNDO) undoStack.shift()
  undoDepth.value = undoStack.length
  redoDepth.value = redoStack.length
  truthCtx.putImageData(snapshot, 0, 0)
  refreshTruthOutline()
}

function resetToDetection() {
  if (!truthCtx) return
  pushUndo()
  truthCtx.clearRect(0, 0, truthCanvas.width, truthCanvas.height)
  for (const contour of detected.value) fillRings(truthCtx, [contour.pointsPx])
  refreshTruthOutline()
}

// --- zoom / pan / lasso ---
function onWheel(event: WheelEvent) {
  if (regionsSvg.value) zoomAt(viewBox, regionsSvg.value, event, rectifiedW.value)
}

function onRegionsDown(event: PointerEvent) {
  if (!regionsSvg.value) return
  event.preventDefault()
  regionsSvg.value.setPointerCapture(event.pointerId)
  if (event.button === 1) {
    panLast = [event.clientX, event.clientY]
  } else if (event.button === 0 || event.button === 2) {
    const p = svgEventPoint(regionsSvg.value, event)
    lassoButtonDown = true
    if (lasso.value) {
      // Selection kept open with Shift: a click appends a straight segment.
      lasso.value.pointsPx.push(p)
    } else {
      const mode = event.button === 0 ? 'add' : 'remove'
      lasso.value = { mode, pointsPx: [p] }
    }
  }
}

function onRegionsMove(event: PointerEvent) {
  if (!regionsSvg.value) return
  if (panLast) {
    panLast = panBy(viewBox, regionsSvg.value, event, panLast)
    return
  }
  if (lasso.value) {
    const p = svgEventPoint(regionsSvg.value, event)
    hoverPoint.value = p
    // Freehand only while a lasso button is actually held down; while the
    // Shift-held selection hovers, the dashed preview segment follows instead.
    if (event.buttons & 3) {
      const pts = lasso.value.pointsPx
      const [lx, ly] = pts[pts.length - 1]
      if (Math.hypot(p[0] - lx, p[1] - ly) > viewBox.w / 400) pts.push(p)
    }
  }
}

function onRegionsUp(event: PointerEvent) {
  panLast = null
  lassoButtonDown = false
  if (!lasso.value) return
  if (event.shiftKey) return // Shift keeps the selection open
  finishLasso()
}

function finishLasso() {
  if (!lasso.value) return
  if (lasso.value.pointsPx.length >= 3) {
    applyLasso(lasso.value.mode, lasso.value.pointsPx)
  }
  lasso.value = null
  hoverPoint.value = null
}

function cancelLasso() {
  lasso.value = null
  hoverPoint.value = null
}

function onKeyUp(event: KeyboardEvent) {
  // If a button is still down, the coming pointerup finalizes instead.
  if (event.key === 'Shift' && lasso.value && !lassoButtonDown) finishLasso()
}

function onKeyDown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    cancelLasso()
    return
  }
  if (!(event.ctrlKey || event.metaKey) || phase.value !== 'regions') return
  const key = event.key.toLowerCase()
  if (key === 'z' && !event.shiftKey) {
    if (!undoDepth.value) return
    event.preventDefault()
    undoEdit()
  } else if (key === 'y' || (key === 'z' && event.shiftKey)) {
    if (!redoDepth.value) return
    event.preventDefault()
    redoEdit()
  }
}

// --- detection overlay + IoU against the labeled truth ---
async function runExtraction() {
  if (!rectifiedCanvas.value) return
  const epoch = truthEpoch
  const canvas = rectifiedCanvas.value
  const cv = await loadOpenCV()
  if (epoch !== truthEpoch || canvas !== rectifiedCanvas.value) return
  detected.value = extractToolContours(cv, canvas, {
    pxPerMm: PX_PER_MM,
    sensitivity: sensitivity.value,
  })
  computeIoU()
}

function computeIoU() {
  if (!truthCtx || !rectifiedW.value) {
    iou.value = null
    return
  }
  const scale = 0.5 // score at 2 px/mm — plenty
  const w = Math.round(rectifiedW.value * scale)
  const h = Math.round(rectifiedH.value * scale)

  const truthSmall = document.createElement('canvas')
  truthSmall.width = w
  truthSmall.height = h
  const tc = truthSmall.getContext('2d', { willReadFrequently: true })!
  tc.drawImage(truthCanvas, 0, 0, w, h)

  const det = document.createElement('canvas')
  det.width = w
  det.height = h
  const dc = det.getContext('2d', { willReadFrequently: true })!
  dc.scale(scale, scale)
  for (const contour of detected.value) fillRings(dc, [contour.pointsPx])

  const ta = tc.getImageData(0, 0, w, h).data
  const da = dc.getImageData(0, 0, w, h).data
  let inter = 0
  let union = 0
  for (let i = 0; i < w * h; i++) {
    const t = ta[i * 4 + 3] > 127
    const d = da[i * 4 + 3] > 127
    if (t && d) inter++
    if (t || d) union++
  }
  iou.value = union === 0 ? null : inter / union
}

watch(sensitivity, () => {
  clearTimeout(extractTimer)
  extractTimer = setTimeout(runExtraction, 250)
})

// --- autosave ---
function buildAnnotation(): TrainingAnnotation | null {
  if (!corners.value || !photoW.value) return null
  return {
    photo: selectedPhoto.value,
    paperSizeId: effectivePaperId.value,
    landscape: autoClass.value?.landscape ?? false,
    corners: corners.value.map(([x, y]) => [x / photoW.value, y / photoH.value]) as [
      Vec2, Vec2, Vec2, Vec2,
    ],
    truth: truthPolysPx.value.map((poly) => ({
      outerMm: poly.outer.map(([x, y]) => [x / PX_PER_MM, y / PX_PER_MM] as Vec2),
      holesMm: poly.holes.map((h) => h.map(([x, y]) => [x / PX_PER_MM, y / PX_PER_MM] as Vec2)),
    })),
  }
}

function scheduleSave() {
  if (loading) return
  saveState.value = 'dirty'
  clearTimeout(saveTimer)
  saveTimer = setTimeout(doSave, 800)
}

async function doSave() {
  const annotation = buildAnnotation()
  if (!annotation) return
  const epoch = truthEpoch
  const photo = annotation.photo
  saveState.value = 'saving'
  try {
    await fetch(`/__training/data/${encodeURIComponent(photo)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(annotation),
    })
    // Only clear the dirty flag if nothing newer started while we were writing.
    if (epoch === truthEpoch && selectedPhoto.value === photo) saveState.value = 'saved'
  } catch (e) {
    if (selectedPhoto.value === photo) saveState.value = 'dirty'
    error.value = `Save failed: ${e instanceof Error ? e.message : e}`
  }
}

async function flushSave() {
  clearTimeout(saveTimer)
  if (saveState.value !== 'saved') await doSave()
}

watch([corners, paperChoice], () => scheduleSave(), { deep: true })
</script>

<template>
  <div class="train">
    <header>
      <h1>Detection training</h1>
      <select :value="selectedPhoto" @change="selectPhoto(($event.target as HTMLSelectElement).value)">
        <option v-for="p in photos" :key="p" :value="p">{{ p }}</option>
      </select>
      <span class="save-state" :class="saveState">{{
        saveState === 'saved' ? 'Saved' : saveState === 'saving' ? 'Saving…' : 'Unsaved'
      }}</span>
    </header>

    <!-- Phase: paper corners -->
    <div v-if="phase === 'corners'" class="stage">
      <div class="canvas-wrap">
        <svg
          ref="cornersSvg"
          class="fit draw"
          :viewBox="cornersViewBoxAttr"
          @wheel.prevent="onCornersWheel"
          @pointerdown="onCornersPointerDown"
          @pointermove="onCornersMove"
          @pointerup="onCornersUp"
        >
          <image :href="photoUrl" :width="photoW" :height="photoH" />
          <polygon
            v-if="corners"
            :points="polygonToSvgPoints(corners)"
            fill="rgba(76,141,255,0.15)"
            stroke="#4c8dff"
            :stroke-width="cornersViewBox.w / 400"
          />
          <circle
            v-for="(c, i) in corners"
            :key="i"
            :cx="c[0]"
            :cy="c[1]"
            :r="cornersViewBox.w / 45"
            fill="rgba(76,141,255,0.35)"
            stroke="#fff"
            :stroke-width="cornersViewBox.w / 500"
            style="cursor: grab; touch-action: none"
            @pointerdown="onCornerDown(i, $event)"
          />
        </svg>
      </div>
      <aside class="sidebar">
        <div class="panel">
          <h3>Expected paper outline</h3>
          <p class="muted">
            Drag the corners onto the exact paper edges — this is ground truth.
            <b>Wheel</b> zooms, <b>middle-drag</b> pans.
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
            <button @click="redetectCorners">Re-detect</button>
            <button class="primary" :disabled="!!busy" @click="toRegions">Label tools →</button>
          </div>
        </div>
      </aside>
    </div>

    <!-- Phase: tool outline correction -->
    <div v-else class="stage">
      <div class="canvas-wrap">
        <svg
          ref="regionsSvg"
          class="fit draw"
          :viewBox="viewBoxAttr"
          @wheel.prevent="onWheel"
          @pointerdown="onRegionsDown"
          @pointermove="onRegionsMove"
          @pointerup="onRegionsUp"
          @contextmenu.prevent
        >
          <image :href="rectifiedUrl" :width="rectifiedW" :height="rectifiedH" />
          <template v-if="showDetection">
            <polygon
              v-for="(contour, i) in detected"
              :key="`d${i}`"
              :points="polygonToSvgPoints(contour.pointsPx)"
              fill="none"
              stroke="#ff9f43"
              :stroke-width="strokePx * 1.5"
              stroke-dasharray="6 4"
            />
          </template>
          <path
            v-for="(poly, i) in truthPolysPx"
            :key="`t${i}`"
            :d="truthPathD(poly)"
            fill="rgba(76,141,255,0.18)"
            fill-rule="evenodd"
            stroke="#4c8dff"
            :stroke-width="strokePx"
          />
          <polyline
            v-if="lasso"
            :points="polygonToSvgPoints(lasso.pointsPx)"
            fill="none"
            :stroke="lasso.mode === 'add' ? '#3fb970' : '#ff5050'"
            :stroke-width="strokePx * 2"
          />
          <line
            v-if="lasso && hoverPoint"
            :x1="lasso.pointsPx[lasso.pointsPx.length - 1][0]"
            :y1="lasso.pointsPx[lasso.pointsPx.length - 1][1]"
            :x2="hoverPoint[0]"
            :y2="hoverPoint[1]"
            :stroke="lasso.mode === 'add' ? '#3fb970' : '#ff5050'"
            :stroke-width="strokePx"
            stroke-dasharray="5 5"
          />
        </svg>
      </div>
      <aside class="sidebar">
        <div class="panel">
          <h3>Correct the outline</h3>
          <p class="muted">The blue outline is the ground truth (seeded from detection).</p>
          <ul class="muted help-list">
            <li><b>Left-drag</b> — add the encircled area</li>
            <li><b>Right-drag</b> — carve it away</li>
            <li><b>Shift</b> — keep selection open; clicks add straight segments, drags add freehand; release Shift to apply</li>
            <li><b>Esc</b> — cancel the selection</li>
            <li><b>Ctrl+Z</b> / <b>Ctrl+Y</b> — undo / redo</li>
            <li><b>Wheel</b> — zoom</li>
            <li><b>Middle-drag</b> — pan</li>
          </ul>
          <div class="actions left">
            <button :disabled="!undoDepth" @click="undoEdit">Undo</button>
            <button :disabled="!redoDepth" @click="redoEdit">Redo</button>
            <button @click="resetToDetection">Reset to detection</button>
          </div>
        </div>
        <div class="panel">
          <h3>Current detection</h3>
          <div class="field">
            <label>
              <input v-model="showDetection" type="checkbox" />
              Show detected contours (orange dashed)
            </label>
          </div>
          <div class="field">
            <label>Sensitivity {{ sensitivity > 0 ? '+' : '' }}{{ sensitivity }}</label>
            <input v-model.number="sensitivity" type="range" min="-60" max="60" step="5" />
          </div>
          <p class="iou">
            Detection IoU vs truth:
            <b>{{ iou === null ? '—' : (iou * 100).toFixed(1) + '%' }}</b>
          </p>
        </div>
        <div class="panel">
          <div class="actions">
            <button @click="phase = 'corners'">← Corners</button>
            <button class="primary" @click="flushSave">Save now</button>
          </div>
        </div>
      </aside>
    </div>

    <div v-if="busy" class="busy-overlay">
      <span>{{ busy }}</span>
    </div>
    <div v-if="error" class="status error">{{ error }}</div>
  </div>
</template>

<style scoped>
.train {
  height: 100vh;
  display: flex;
  flex-direction: column;
  position: relative;
}

header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-raised);
}

header h1 {
  font-size: 16px;
  margin: 0;
}

header select {
  min-width: 260px;
}

.save-state {
  margin-left: auto;
  font-size: 13px;
  color: var(--muted);
}

.save-state.dirty {
  color: #e8b13f;
}

.save-state.saved {
  color: #3fb970;
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

.fit.draw {
  touch-action: none;
  cursor: crosshair;
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

.actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.actions.left {
  justify-content: flex-start;
}

.help-list {
  margin: 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
}

.iou {
  margin: 0;
  font-size: 14px;
}

.busy-overlay {
  position: absolute;
  inset: 0;
  z-index: 20;
  background: rgba(5, 8, 12, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
}

.busy-overlay span {
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 12px 26px;
  font-weight: 600;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
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
</style>
