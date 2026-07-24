/** Tunable tool-contour detection parameters. */

export interface ContourParams {
  darkRatio0: number
  darkRatioPerSens: number
  chroma0: number
  chromaPerSens: number
  highlightDelta: number
  marginMm: number
  minAreaMm2: number
  openKernel: number
  closeKernel: number
  bgKernelMm: number
  approxEpsMm: number
}

/**
 * Defaults used by the production detector.
 * Tuned on training/; see training/accuracy-log.csv.
 */
export const DEFAULT_CONTOUR_PARAMS: ContourParams = {
  darkRatio0: 0.4,
  darkRatioPerSens: 1 / 150,
  chroma0: 33,
  chromaPerSens: 1 / 5,
  highlightDelta: 25,
  marginMm: 5,
  minAreaMm2: 80,
  openKernel: 3,
  closeKernel: 7,
  bgKernelMm: 45,
  approxEpsMm: 0.3,
}

export function darkRatioAt(params: ContourParams, sensitivity: number): number {
  return Math.min(0.92, Math.max(0.15, params.darkRatio0 + sensitivity * params.darkRatioPerSens))
}

export function chromaThreshAt(params: ContourParams, sensitivity: number): number {
  return Math.min(45, Math.max(15, params.chroma0 - sensitivity * params.chromaPerSens))
}
