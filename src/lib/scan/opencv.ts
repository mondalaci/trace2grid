// Load the raw UMD file as a classic script instead of importing the module:
// opencv.js sets `module.exports` to a *Promise*, which breaks bundler CJS
// interop (the generated ESM namespace becomes a broken thenable). As a
// classic script it simply sets `window.cv`.
import cvScriptUrl from '@techstark/opencv-js/dist/opencv.js?url'

export type CV = typeof import('@techstark/opencv-js')

let cvPromise: Promise<CV> | null = null

/**
 * Lazily load the ~10 MB OpenCV.js bundle (separate hashed asset) and wait
 * for the WASM runtime to initialize.
 */
export function loadOpenCV(): Promise<CV> {
  if (!cvPromise) {
    cvPromise = (async () => {
      await new Promise<void>((resolve, reject) => {
        const script = document.createElement('script')
        script.src = cvScriptUrl
        script.onload = () => resolve()
        script.onerror = () => reject(new Error('Failed to load OpenCV.js'))
        document.head.appendChild(script)
      })
      let cv = (window as unknown as { cv: unknown }).cv
      // Emscripten builds expose a promise (or thenable) resolving to the module.
      if (cv && typeof (cv as { then?: unknown }).then === 'function') {
        cv = await cv
      }
      const candidate = cv as CV & { onRuntimeInitialized?: () => void; Mat?: unknown }
      if (!candidate.Mat) {
        await new Promise<void>((resolve) => {
          candidate.onRuntimeInitialized = () => resolve()
        })
      }
      return candidate as CV
    })()
  }
  return cvPromise
}
