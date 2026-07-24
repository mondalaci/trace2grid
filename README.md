# Trace2Grid

Turn a photo of your tools into a 3D-printable [Gridfinity](https://gridfinity.xyz/) bin — entirely in the browser. No server, no uploads: everything (computer vision, solid modeling, exports) runs client-side, so the built app can be hosted on any static CDN.

## Usage

1. **Scan** — Lay your tools on a blank sheet of paper (A4, US Letter, and US Legal are recognized automatically) on a darker surface and photograph it from straight above. The app finds the sheet, lets you fine-tune the corners, perspective-corrects it to real-world millimeters, and extracts the tool silhouettes. Adjust the sensitivity slider and click contours to keep/discard them.
2. **Arrange** — Drag and rotate each contour inside the bin, set the bin footprint (42 mm grid units) and height (7 mm units), tweak per-tool clearance and pocket depth, and click-place finger notches so tools are easy to lift out. Overlap and wall-clearance warnings appear live.
3. **Print** — Inspect the bin in a pannable/rotatable 3D view, then export:
   - **STL** (binary, mm) for slicing and 3D printing
   - **1:1 PDF** with a 100 mm calibration bar — print it at 100% scale and lay your tools on the outlines to verify fit before wasting filament

## Tech

| Concern | Library |
| --- | --- |
| UI | Vue 3 + TypeScript + Vite + Pinia |
| Paper detection, rectification, contours | OpenCV.js (WASM, lazy-loaded) |
| Solid modeling (bin, pockets, offsetting) | manifold-3d (WASM CSG, in a Web Worker) |
| 3D preview | three.js |
| PDF export | jsPDF |

The Gridfinity base profile, stacking lip, and optional magnet holes (6.5 × 2.4 mm) follow the [original spec](https://gridfinity.xyz/specification/): 42 mm pitch, 41.5 mm bins, 7 mm height units, 4.75 mm foot profile.

## Development

```bash
npm install
npm run dev      # dev server
npm run build    # type-check + static bundle in dist/
npm run preview  # serve the production build locally
```

Deploy by uploading `dist/` to any static host — the bundle uses relative paths, so it works from a subdirectory too.

## Training data

Detection quality is tuned against labeled ground truth in `training/`: each photo sits next to a `<photo>.json` annotation with the exact paper corners and the corrected tool outlines (in paper millimeters).

To label photos, drop them into `training/`, run `npm run dev`, and open `http://localhost:5173/train.html`:

1. Pick the photo, drag the paper corners exactly onto the sheet edges (wheel zooms, middle-drag pans).
2. "Label tools →" seeds the blue outline from the current detector; left-drag adds area, right-drag carves away (shadows, gaps).
3. Everything autosaves to `training/<photo>.json`.

Contributions of tricky photos (shiny tools, harsh shadows, unusual paper/backgrounds) with corrected labels are welcome — submit both files in a PR. Please strip EXIF GPS data first, e.g. `exiftool -gps:all= training/*.jpg`.

## Tips for good scans

- Strong, even light; avoid hard shadows next to tools (or lower the sensitivity slider).
- Detection compares each pixel's color and brightness against the paper, so colored or dark tools work best; pale gray tools on white paper may need the sensitivity slider raised.
- Soft shadows are ignored automatically, but harsh directional light casts near-black contact shadows that can't be told apart from the tool — prefer diffuse, even light.
- Shoot from directly overhead. Tall tools "grow" slightly in the photo due to parallax — that's what the per-tool clearance and the 1:1 PDF check are for.
- Distance helps: the farther the camera, the smaller the perspective error from tool height. Step back and zoom in, ideally with a telephoto lens (2–3× on most phones).
