import vue from '@vitejs/plugin-vue'
import { existsSync } from 'node:fs'
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { defineConfig, type Plugin } from 'vite'

const PHOTO_EXTENSIONS = /\.(jpe?g|png|webp)$/i
const PHOTO_MIME: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
}

/**
 * Dev-only endpoints for the training UI (train.html). Photos and their
 * <photo>.json annotations live side by side in training/, which is tracked
 * in git so contributors can submit new labeled photos.
 *   GET  /__training/photos       -> { photos: string[] }
 *   GET  /__training/photo/<name> -> image bytes
 *   GET  /__training/data/<name>  -> saved annotation JSON (404 if none)
 *   POST /__training/data/<name>  -> persist annotation JSON
 */
function trainingPlugin(): Plugin {
  let root = ''
  return {
    name: 'trace2grid-training',
    apply: 'serve',
    configResolved(config) {
      root = config.root
    },
    configureServer(server) {
      server.middlewares.use('/__training', (req, res) => {
        void (async () => {
          const url = new URL(req.url ?? '/', 'http://localhost')
          const [action, rawName] = url.pathname.split('/').filter(Boolean)
          const name = rawName ? path.basename(decodeURIComponent(rawName)) : ''
          const photosDir = path.join(root, 'training')
          const dataDir = path.join(root, 'training')

          if (action === 'photos') {
            const entries = existsSync(photosDir) ? await readdir(photosDir) : []
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ photos: entries.filter((f) => PHOTO_EXTENSIONS.test(f)).sort() }))
            return
          }

          if (action === 'photo' && name) {
            const file = path.join(photosDir, name)
            if (!existsSync(file)) {
              res.statusCode = 404
              res.end('not found')
              return
            }
            res.setHeader('Content-Type', PHOTO_MIME[path.extname(name).toLowerCase()] ?? 'application/octet-stream')
            res.end(await readFile(file))
            return
          }

          if (action === 'data' && name) {
            const file = path.join(dataDir, `${name}.json`)
            if (req.method === 'POST') {
              const chunks: Buffer[] = []
              for await (const chunk of req) chunks.push(chunk as Buffer)
              const body = JSON.parse(Buffer.concat(chunks).toString('utf8'))
              await mkdir(dataDir, { recursive: true })
              await writeFile(file, JSON.stringify(body, null, 2))
              res.setHeader('Content-Type', 'application/json')
              res.end('{"ok":true}')
              return
            }
            if (!existsSync(file)) {
              res.statusCode = 404
              res.end('not found')
              return
            }
            res.setHeader('Content-Type', 'application/json')
            res.end(await readFile(file))
            return
          }

          res.statusCode = 400
          res.end('bad request')
        })().catch((err) => {
          res.statusCode = 500
          res.end(String(err))
        })
      })
    },
  }
}

export default defineConfig({
  // Relative base so the built bundle works from any CDN path.
  base: './',
  plugins: [trainingPlugin(), vue()],
  optimizeDeps: {
    // Pre-bundling breaks manifold's import.meta.url based wasm lookup.
    // opencv-js must NOT be excluded: it's CJS and needs the interop shim.
    exclude: ['manifold-3d'],
  },
  build: {
    chunkSizeWarningLimit: 15000,
    target: 'es2022',
    rollupOptions: {
      // train.html is a dev-only labeling tool but building it keeps types honest.
      input: {
        main: path.resolve(import.meta.dirname, 'index.html'),
        train: path.resolve(import.meta.dirname, 'train.html'),
      },
    },
  },
  worker: {
    format: 'es',
  },
})
