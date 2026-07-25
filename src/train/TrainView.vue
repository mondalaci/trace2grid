<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref, triggerRef, watch } from 'vue'
import { type ExtractedContour } from '../lib/scan/contours'
import { extractToolContoursNn } from '../lib/scan/nnSeg'
import { loadOpenCV } from '../lib/scan/opencv'
import {
  classifyPaper,
  detectPaperQuadNn,
  PAPER_SIZES,
  paperById,
  rectifyPaper,
} from '../lib/scan/paper'
import { polygonToSvgPoints, svgClientPoint, svgEventPoint } from '../lib/svg'
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
let panLast: Vec2 | null = null

type TruthPoly = { outer: Vec2[]; holes: Vec2[][] }
/** hole = -1 → outer ring; otherwise index into holes. */
type RingLoc = { poly: number; hole: number }
type VertDrag = RingLoc & { idx: number }

const truthPolysPx = ref<TruthPoly[]>([])
const undoStack: TruthPoly[][] = []
const undoDepth = ref(0)
const redoStack: TruthPoly[][] = []
const redoDepth = ref(0)
let savedTruth: TruthPolygon[] | null = null
let seededFor = ''
/** Bumped on photo switch so stale async extraction is ignored. */
let truthEpoch = 0
let draggingVert: VertDrag | null = null

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
const vertexR = computed(() => viewBox.w / 280)
/** Invisible hit target — larger than the drawn handle for easier grabbing. */
const vertexHitR = computed(() => Math.max(vertexR.value * 4, viewBox.w / 70))
const cornerR = computed(() => cornersViewBox.w / 280)
const cornerHitR = computed(() => Math.max(cornerR.value * 4, cornersViewBox.w / 70))
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
  // Use the screen CTM so letterboxed `meet` scaling matches 1:1 with the cursor
  // (dividing by the element rect over-damps pan when the image doesn't fill the SVG).
  const [x0, y0] = svgClientPoint(svg, last[0], last[1])
  const [x1, y1] = svgClientPoint(svg, event.clientX, event.clientY)
  vb.x -= x1 - x0
  vb.y -= y1 - y0
  return [event.clientX, event.clientY]
}

function truthPathD(poly: TruthPoly): string {
  const ring = (pts: Vec2[]) =>
    pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join('') + 'Z'
  return ring(poly.outer) + poly.holes.map(ring).join('')
}

function clonePolys(polys: TruthPoly[]): TruthPoly[] {
  return polys.map((p) => ({
    outer: p.outer.map((v) => [v[0], v[1]] as Vec2),
    holes: p.holes.map((h) => h.map((v) => [v[0], v[1]] as Vec2)),
  }))
}

function getRing(loc: RingLoc): Vec2[] | null {
  const poly = truthPolysPx.value[loc.poly]
  if (!poly) return null
  return loc.hole < 0 ? poly.outer : (poly.holes[loc.hole] ?? null)
}

function fillRings(ctx: CanvasRenderingContext2D, rings: Vec2[][]) {
  const path = new Path2D()
  for (const ring of rings) {
    ring.forEach(([x, y], i) => (i ? path.lineTo(x, y) : path.moveTo(x, y)))
    path.closePath()
  }
  ctx.fillStyle = '#fff'
  ctx.fill(path, 'evenodd')
}

onMounted(async () => {
  window.addEventListener('keydown', onKeyDown)
  window.addEventListener('beforeunload', onBeforeUnload)
  const res = await fetch('/__training/photos')
  photos.value = (await res.json()).photos
  const requested = new URLSearchParams(location.search).get('photo')
  const first = requested && photos.value.includes(requested) ? requested : photos.value[0]
  if (first) await selectPhoto(first)
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDown)
  window.removeEventListener('beforeunload', onBeforeUnload)
})

function isDirty() {
  return saveState.value === 'dirty' || saveState.value === 'saving'
}

function onBeforeUnload(event: BeforeUnloadEvent) {
  if (!isDirty()) return
  event.preventDefault()
  // Required for Chromium to show the native leave dialog.
  event.returnValue = ''
}

async function onPhotoSelect(event: Event) {
  const sel = event.target as HTMLSelectElement
  const name = sel.value
  if (name === selectedPhoto.value) return
  if (isDirty()) {
    const ok = confirm(
      'You have unsaved changes. Save them and switch photos?\n\nCancel to stay on this photo.',
    )
    if (!ok) {
      sel.value = selectedPhoto.value
      return
    }
  }
  await selectPhoto(name)
}

async function selectPhoto(name: string) {
  if (isDirty()) await flushSave()
  selectedPhoto.value = name
  error.value = null
  busy.value = 'Loading photo…'
  loading = true
  seededFor = ''
  savedTruth = null
  undoStack.length = 0
  undoDepth.value = 0
  redoStack.length = 0
  redoDepth.value = 0
  draggingVert = null
  truthPolysPx.value = []
  detected.value = []
  iou.value = null
  truthEpoch++
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
      const quad = await detectPaperQuadNn(cv, canvas)
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
  if (event.button !== 0 || !cornersSvg.value) return
  event.preventDefault()
  event.stopPropagation()
  cornersSvg.value.setPointerCapture(event.pointerId)
  draggingCorner = index
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
    const quad = await detectPaperQuadNn(cv, photoCanvas.value)
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
      seedTruth(savedTruth)
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

// --- truth polygon editing ---
function seedTruth(fromSaved: TruthPolygon[] | null) {
  undoStack.length = 0
  undoDepth.value = 0
  redoStack.length = 0
  redoDepth.value = 0
  if (fromSaved?.length) {
    truthPolysPx.value = fromSaved.map((poly) => ({
      outer: poly.outerMm.map(([x, y]) => [x * PX_PER_MM, y * PX_PER_MM] as Vec2),
      holes: poly.holesMm.map((h) => h.map(([x, y]) => [x * PX_PER_MM, y * PX_PER_MM] as Vec2)),
    }))
  } else {
    truthPolysPx.value = detected.value.map((c) => ({
      outer: c.pointsPx.map(([x, y]) => [x, y] as Vec2),
      holes: [],
    }))
  }
  scheduleSave()
  computeIoU()
}

function pushUndo() {
  undoStack.push(clonePolys(truthPolysPx.value))
  if (undoStack.length > MAX_UNDO) undoStack.shift()
  undoDepth.value = undoStack.length
  redoStack.length = 0
  redoDepth.value = 0
}

function commitEdit() {
  scheduleSave()
  computeIoU()
}

function undoEdit() {
  const snapshot = undoStack.pop()
  if (!snapshot) return
  redoStack.push(clonePolys(truthPolysPx.value))
  redoDepth.value = redoStack.length
  undoDepth.value = undoStack.length
  truthPolysPx.value = snapshot
  commitEdit()
}

function redoEdit() {
  const snapshot = redoStack.pop()
  if (!snapshot) return
  undoStack.push(clonePolys(truthPolysPx.value))
  if (undoStack.length > MAX_UNDO) undoStack.shift()
  undoDepth.value = undoStack.length
  redoDepth.value = redoStack.length
  truthPolysPx.value = snapshot
  commitEdit()
}

function resetToDetection() {
  pushUndo()
  truthPolysPx.value = detected.value.map((c) => ({
    outer: c.pointsPx.map(([x, y]) => [x, y] as Vec2),
    holes: [],
  }))
  commitEdit()
}

function projectOnSegment(
  p: Vec2,
  a: Vec2,
  b: Vec2,
): { point: Vec2; t: number; dist: number } {
  const abx = b[0] - a[0]
  const aby = b[1] - a[1]
  const len2 = abx * abx + aby * aby
  if (len2 < 1e-9) return { point: [a[0], a[1]], t: 0, dist: Math.hypot(p[0] - a[0], p[1] - a[1]) }
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / len2))
  const point: Vec2 = [a[0] + abx * t, a[1] + aby * t]
  return { point, t, dist: Math.hypot(p[0] - point[0], p[1] - point[1]) }
}

function nearestEdge(
  p: Vec2,
): (RingLoc & { idx: number; point: Vec2 }) | null {
  const maxDist = viewBox.w / 40
  let bestPoly = -1
  let bestHole = -1
  let bestIdx = -1
  let bestPoint: Vec2 = [0, 0]
  let bestDist = maxDist
  for (let polyI = 0; polyI < truthPolysPx.value.length; polyI++) {
    const poly = truthPolysPx.value[polyI]
    const rings: { hole: number; pts: Vec2[] }[] = [
      { hole: -1, pts: poly.outer },
      ...poly.holes.map((pts, hole) => ({ hole, pts })),
    ]
    for (const { hole, pts } of rings) {
      for (let i = 0; i < pts.length; i++) {
        const a = pts[i]
        const b = pts[(i + 1) % pts.length]
        const hit = projectOnSegment(p, a, b)
        if (hit.t < 0.05 || hit.t > 0.95) continue
        if (hit.dist >= bestDist) continue
        bestDist = hit.dist
        bestPoly = polyI
        bestHole = hole
        bestIdx = i
        bestPoint = hit.point
      }
    }
  }
  if (bestPoly < 0) return null
  return { poly: bestPoly, hole: bestHole, idx: bestIdx, point: bestPoint }
}

function onVertDown(loc: VertDrag, event: PointerEvent) {
  if (event.button === 2) {
    event.preventDefault()
    event.stopPropagation()
    deleteVertex(loc)
    return
  }
  if (event.button !== 0 || !regionsSvg.value) return
  event.preventDefault()
  event.stopPropagation()
  regionsSvg.value.setPointerCapture(event.pointerId)
  pushUndo()
  draggingVert = loc
}

function onVertMove(event: PointerEvent) {
  if (!draggingVert || !regionsSvg.value) return
  const ring = getRing(draggingVert)
  if (!ring) return
  const [x, y] = svgEventPoint(regionsSvg.value, event)
  ring[draggingVert.idx] = [x, y]
  triggerRef(truthPolysPx)
}

function onVertUp() {
  if (!draggingVert) return
  draggingVert = null
  commitEdit()
}

function deleteVertex(loc: VertDrag) {
  const ring = getRing(loc)
  if (!ring || ring.length <= 3) return
  pushUndo()
  ring.splice(loc.idx, 1)
  triggerRef(truthPolysPx)
  commitEdit()
}

function onRegionsDblClick(event: MouseEvent) {
  if (!regionsSvg.value) return
  const p = svgEventPoint(regionsSvg.value, event)
  const edge = nearestEdge(p)
  if (!edge) return
  const ring = getRing(edge)
  if (!ring) return
  pushUndo()
  ring.splice(edge.idx + 1, 0, edge.point)
  triggerRef(truthPolysPx)
  commitEdit()
}

// --- zoom / pan ---
function onWheel(event: WheelEvent) {
  if (regionsSvg.value) zoomAt(viewBox, regionsSvg.value, event, rectifiedW.value)
}

function onRegionsDown(event: PointerEvent) {
  if (!regionsSvg.value) return
  if (event.button === 1) {
    event.preventDefault()
    regionsSvg.value.setPointerCapture(event.pointerId)
    panLast = [event.clientX, event.clientY]
  }
}

function onRegionsMove(event: PointerEvent) {
  if (!regionsSvg.value) return
  if (draggingVert) {
    onVertMove(event)
    return
  }
  if (panLast) panLast = panBy(viewBox, regionsSvg.value, event, panLast)
}

function onRegionsUp() {
  if (draggingVert) onVertUp()
  panLast = null
}

function onKeyDown(event: KeyboardEvent) {
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
  detected.value = await extractToolContoursNn(cv, canvas, {
    pxPerMm: PX_PER_MM,
    sensitivity: sensitivity.value,
  })
  computeIoU()
}

function computeIoU() {
  if (!rectifiedW.value || !truthPolysPx.value.length) {
    iou.value = null
    return
  }
  const scale = 0.5
  const w = Math.round(rectifiedW.value * scale)
  const h = Math.round(rectifiedH.value * scale)

  const truthSmall = document.createElement('canvas')
  truthSmall.width = w
  truthSmall.height = h
  const tc = truthSmall.getContext('2d', { willReadFrequently: true })!
  tc.scale(scale, scale)
  for (const poly of truthPolysPx.value) {
    fillRings(tc, [poly.outer, ...poly.holes])
  }

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
  // While still on the corners phase, truthPolysPx is empty — keep any
  // previously labeled outlines instead of wiping them on corner tweaks.
  const truth =
    truthPolysPx.value.length > 0
      ? truthPolysPx.value.map((poly) => ({
          outerMm: poly.outer.map(([x, y]) => [x / PX_PER_MM, y / PX_PER_MM] as Vec2),
          holesMm: poly.holes.map((h) => h.map(([x, y]) => [x / PX_PER_MM, y / PX_PER_MM] as Vec2)),
        }))
      : seededFor
        ? []
        : (savedTruth ?? [])
  return {
    photo: selectedPhoto.value,
    paperSizeId: effectivePaperId.value,
    landscape: autoClass.value?.landscape ?? false,
    corners: corners.value.map(([x, y]) => [x / photoW.value, y / photoH.value]) as [
      Vec2, Vec2, Vec2, Vec2,
    ],
    truth,
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
      <select :value="selectedPhoto" @change="onPhotoSelect">
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
          <g v-for="(c, i) in corners" :key="i" class="vertex">
            <circle
              class="vertex-hit"
              :cx="c[0]"
              :cy="c[1]"
              :r="cornerHitR"
              @pointerdown="onCornerDown(i, $event)"
            />
            <circle
              class="vertex-dot"
              :cx="c[0]"
              :cy="c[1]"
              :r="cornerR"
              fill="rgba(76,141,255,0.55)"
              stroke="#fff"
              :stroke-width="cornersViewBox.w / 500"
            />
          </g>
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
          @dblclick.prevent="onRegionsDblClick"
          @contextmenu.prevent
        >
          <image :href="rectifiedUrl" :width="rectifiedW" :height="rectifiedH" />
          <!-- Truth fill (no stroke) under the blend group -->
          <path
            v-for="(poly, pi) in truthPolysPx"
            :key="`tf${pi}`"
            :d="truthPathD(poly)"
            fill="rgba(76,141,255,0.18)"
            fill-rule="evenodd"
            stroke="none"
            style="pointer-events: none"
          />
          <!-- Strokes isolated so orange∩blue → purple, without blending into the photo -->
          <g class="contour-strokes">
            <path
              v-for="(poly, pi) in truthPolysPx"
              :key="`ts${pi}`"
              :d="truthPathD(poly)"
              fill="none"
              fill-rule="evenodd"
              stroke="#4c8dff"
              :stroke-width="strokePx"
              style="pointer-events: none"
            />
            <template v-if="showDetection">
              <polygon
                v-for="(contour, i) in detected"
                :key="`d${i}`"
                class="detection-stroke"
                :points="polygonToSvgPoints(contour.pointsPx)"
                fill="none"
                stroke="#ff9f43"
                :stroke-width="strokePx"
              />
            </template>
          </g>
          <g v-for="(poly, pi) in truthPolysPx" :key="`t${pi}`">
            <template v-for="(c, vi) in poly.outer" :key="`o${vi}`">
              <g class="vertex">
                <circle
                  class="vertex-hit"
                  :cx="c[0]"
                  :cy="c[1]"
                  :r="vertexHitR"
                  @pointerdown="onVertDown({ poly: pi, hole: -1, idx: vi }, $event)"
                />
                <circle
                  class="vertex-dot"
                  :cx="c[0]"
                  :cy="c[1]"
                  :r="vertexR"
                  fill="rgba(76,141,255,0.55)"
                  stroke="#fff"
                  :stroke-width="strokePx * 0.4"
                />
              </g>
            </template>
            <template v-for="(hole, hi) in poly.holes" :key="`h${hi}`">
              <template v-for="(c, vi) in hole" :key="`hv${vi}`">
                <g class="vertex">
                  <circle
                    class="vertex-hit"
                    :cx="c[0]"
                    :cy="c[1]"
                    :r="vertexHitR"
                    @pointerdown="onVertDown({ poly: pi, hole: hi, idx: vi }, $event)"
                  />
                  <circle
                    class="vertex-dot"
                    :cx="c[0]"
                    :cy="c[1]"
                    :r="vertexR"
                    fill="rgba(255,120,80,0.55)"
                    stroke="#fff"
                    :stroke-width="strokePx * 0.4"
                  />
                </g>
              </template>
            </template>
          </g>
        </svg>
      </div>
      <aside class="sidebar">
        <div class="panel">
          <h3>Correct the outline</h3>
          <p class="muted">
            Blue = truth, orange = detection, purple = where they overlap.
          </p>
          <ul class="muted help-list">
            <li><b>Left-drag</b> a vertex — move it</li>
            <li><b>Double-click</b> an edge — insert a vertex</li>
            <li><b>Right-click</b> a vertex — delete it (rings keep ≥3 points)</li>
            <li><b>Ctrl+Z</b> / <b>Ctrl+Y</b> — undo / redo</li>
            <li><b>Wheel</b> — zoom</li>
            <li><b>Middle-drag</b> — pan</li>
          </ul>
          <div class="actions left">
            <button :disabled="!undoDepth" @click="undoEdit">Undo</button>
            <button :disabled="!redoDepth" @click="redoEdit">Redo</button>
          </div>
        </div>
        <div class="panel">
          <h3>Current detection</h3>
          <div class="field">
            <label>
              <input v-model="showDetection" type="checkbox" />
              Show detected contours (orange; purple where they match truth)
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
          <div class="actions left">
            <button :disabled="!detected.length" @click="resetToDetection">
              Override truth with detection
            </button>
          </div>
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

.contour-strokes {
  isolation: isolate;
  pointer-events: none;
}

.detection-stroke {
  /* Orange vs blue → magenta/purple where strokes overlap; orange alone elsewhere. */
  mix-blend-mode: difference;
}

.vertex {
  cursor: crosshair;
}

.vertex-hit {
  fill: transparent;
  touch-action: none;
}

.vertex-dot {
  pointer-events: none;
  transition: fill 80ms ease, stroke 80ms ease;
}

.vertex:hover .vertex-dot {
  fill: #ffe566;
  stroke: #1a1a1a;
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
