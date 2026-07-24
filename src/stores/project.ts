import { defineStore } from 'pinia'
import { polygonBBox, transformPoints } from '../lib/geometry'
import { offsetPolygon } from '../lib/gridfinity/client'
import type { BuildRequest, PocketSpec } from '../lib/gridfinity/api'
import {
  defaultPocketDepth,
  GRID_PITCH,
  maxPocketDepth,
  MIN_WALL,
} from '../lib/gridfinity/spec'
import type { BinConfig, Notch, Step, Tool, Vec2 } from '../types'

let idCounter = 0
function uid(prefix: string): string {
  return `${prefix}-${++idCounter}-${Date.now().toString(36)}`
}

export const useProjectStore = defineStore('project', {
  state: () => ({
    step: 'capture' as Step,
    tools: [] as Tool[],
    bin: {
      gridX: 3,
      gridY: 2,
      heightUnits: 4,
      stackingLip: true,
      magnetHoles: false,
    } as BinConfig,
    selectedToolId: null as string | null,
  }),

  getters: {
    selectedTool(state): Tool | null {
      return state.tools.find((t) => t.id === state.selectedToolId) ?? null
    },
    binWidthMm(state): number {
      return state.bin.gridX * GRID_PITCH
    },
    binDepthMm(state): number {
      return state.bin.gridY * GRID_PITCH
    },
    maxDepth(state): number {
      return maxPocketDepth(state.bin.heightUnits)
    },
  },

  actions: {
    /** Add scanned contours (mm, centroid-centered, y down) as tools and lay them out. */
    addTools(contours: Vec2[][]) {
      const added: Tool[] = contours.map((points, i) => ({
        id: uid('tool'),
        name: `Tool ${this.tools.length + i + 1}`,
        points,
        x: 0,
        y: 0,
        rotationDeg: 0,
        clearance: 0.4,
        pocketDepth: defaultPocketDepth(this.bin.heightUnits),
        notches: [],
        offsetPolygons: null,
      }))
      this.tools.push(...added)
      this.autoLayout(added)
      this.suggestBinSize()
      for (const tool of added) this.refreshOffset(tool.id)
      if (added.length > 0) this.selectedToolId = added[added.length - 1].id
    },

    /** Simple shelf layout for freshly added tools, in bin mm space. */
    autoLayout(tools: Tool[]) {
      const gap = 6
      let x = gap
      let y = gap
      let rowH = 0
      for (const tool of tools) {
        const bb = polygonBBox(tool.points)
        const w = bb.maxX - bb.minX
        const h = bb.maxY - bb.minY
        if (x + w + gap > this.binWidthMm && x > gap) {
          x = gap
          y += rowH + gap
          rowH = 0
        }
        tool.x = x - bb.minX
        tool.y = y - bb.minY
        x += w + gap
        rowH = Math.max(rowH, h)
      }
    },

    /** Grow the bin (never shrink) so all tools fit with wall margin. */
    suggestBinSize() {
      let maxX = 0
      let maxY = 0
      for (const tool of this.tools) {
        const bb = polygonBBox(transformPoints(tool.points, tool.x, tool.y, tool.rotationDeg))
        maxX = Math.max(maxX, bb.maxX + tool.clearance)
        maxY = Math.max(maxY, bb.maxY + tool.clearance)
      }
      const margin = MIN_WALL + 2
      this.bin.gridX = Math.max(this.bin.gridX, Math.ceil((maxX + margin) / GRID_PITCH))
      this.bin.gridY = Math.max(this.bin.gridY, Math.ceil((maxY + margin) / GRID_PITCH))
    },

    removeTool(id: string) {
      this.tools = this.tools.filter((t) => t.id !== id)
      if (this.selectedToolId === id) this.selectedToolId = null
    },

    duplicateTool(id: string) {
      const src = this.tools.find((t) => t.id === id)
      if (!src) return
      const copy: Tool = JSON.parse(JSON.stringify(src))
      copy.id = uid('tool')
      copy.name = `${src.name} copy`
      copy.x += 10
      copy.y += 10
      copy.notches = copy.notches.map((n) => ({ ...n, id: uid('notch') }))
      this.tools.push(copy)
      this.selectedToolId = copy.id
    },

    async refreshOffset(toolId: string) {
      const tool = this.tools.find((t) => t.id === toolId)
      if (!tool) return
      const clearance = tool.clearance
      try {
        const polygons = await offsetPolygon(tool.points, clearance)
        // Guard against stale results after rapid slider changes.
        const current = this.tools.find((t) => t.id === toolId)
        if (current && current.clearance === clearance) current.offsetPolygons = polygons
      } catch (error) {
        console.error('offset failed', error)
      }
    },

    setClearance(toolId: string, clearance: number) {
      const tool = this.tools.find((t) => t.id === toolId)
      if (!tool) return
      tool.clearance = clearance
      this.refreshOffset(toolId)
    },

    setHeightUnits(units: number) {
      this.bin.heightUnits = units
      const max = maxPocketDepth(units)
      for (const tool of this.tools) {
        if (tool.pocketDepth > max) tool.pocketDepth = max
        for (const notch of tool.notches) {
          if (notch.depth > max) notch.depth = max
        }
      }
    },

    addNotch(toolId: string, local: Vec2, radius = 8) {
      const tool = this.tools.find((t) => t.id === toolId)
      if (!tool) return
      const notch: Notch = {
        id: uid('notch'),
        x: local[0],
        y: local[1],
        radius,
        depth: tool.pocketDepth,
      }
      tool.notches.push(notch)
    },

    removeNotch(toolId: string, notchId: string) {
      const tool = this.tools.find((t) => t.id === toolId)
      if (!tool) return
      tool.notches = tool.notches.filter((n) => n.id !== notchId)
    },

    /** Assemble the geometry worker request: transform everything into centered, y-up bin space. */
    toBuildRequest(): BuildRequest {
      const halfW = this.binWidthMm / 2
      const halfD = this.binDepthMm / 2
      const flip = ([x, y]: Vec2): Vec2 => [x - halfW, halfD - y]

      const pockets: PocketSpec[] = this.tools.map((tool) => {
        const outlines = tool.offsetPolygons ?? [tool.points]
        return {
          polygons: outlines.map((poly) =>
            transformPoints(poly, tool.x, tool.y, tool.rotationDeg).map(flip),
          ),
          depth: Math.min(tool.pocketDepth, this.maxDepth),
          notches: tool.notches.map((notch) => {
            const [nx, ny] = transformPoints([[notch.x, notch.y]], tool.x, tool.y, tool.rotationDeg)[0]
            const [fx, fy] = flip([nx, ny])
            return { x: fx, y: fy, radius: notch.radius, depth: Math.min(notch.depth, this.maxDepth) }
          }),
        }
      })

      return { bin: { ...this.bin }, pockets }
    },
  },
})
