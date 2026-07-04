import { defineConfig } from 'vite'

/**
 * SharedArrayBuffer is only exposed to a cross-origin isolated document, and the whole
 * pipeline is built on it — so without these headers the app does not start at all rather
 * than running slowly.
 *
 * These cover the dev server and `vite preview` only. Production hosting has to send them
 * itself; see docs/deployment.md.
 */
const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

export default defineConfig({
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },

  worker: {
    // Workers default to an immediately-invoked bundle in the production build even though
    // the dev server serves real modules. That difference breaks top-level await and,
    // since Vite 8, leaves `import.meta.url` as undefined — which is exactly what the
    // WebAssembly loader uses to find its own binary. The failure only appears in a built
    // artefact, never in dev, so always verify with `vite build && vite preview`.
    format: 'es',
  },

  build: {
    target: 'es2022',
    // The WebAssembly binary is fetched by URL rather than inlined, so it stays cacheable
    // independently of the JavaScript around it.
    assetsInlineLimit: 0,
  },
})
