/**
 * Reads the host page's palette out of CSS custom properties, so the GPU surfaces are drawn
 * in the same colours as the surrounding UI and follow it across a light/dark toggle.
 *
 * The renderers cannot read a stylesheet, and hard-coding colours would leave the canvases
 * looking pasted onto the page. Resolving the variables once — and again whenever the theme
 * changes — keeps the trace, grid and background in step with everything drawn in the DOM.
 *
 * Two value forms have to be understood. A normal computed colour comes back as
 * `rgb(18 22 30)` (modern space-separated syntax) or `rgb(18, 22, 30)`. But a Tailwind-style
 * variable often stores only the bare channel triple `18 22 30`, because the framework wraps
 * it in `rgb(var(--x) / <alpha>)` at the point of use — so the raw property value has no
 * `rgb(...)` around it at all. Both are handled, along with hex, for robustness.
 */

/** Palette keys the renderers ask for. Anything unresolved falls back to a dark default. */
export type PaletteKey = 'background' | 'surface' | 'foreground' | 'grid' | 'trace' | 'accent'

type Rgb = [number, number, number]

/**
 * Candidate variable names per key, most specific first. Pages vary in what they expose, so
 * several conventional names are tried rather than betting on one.
 */
const CANDIDATES: Record<PaletteKey, readonly string[]> = {
  background: ['--sdr-bg', '--background', '--bg', '--color-background'],
  surface: ['--sdr-surface', '--surface', '--card', '--color-surface'],
  foreground: ['--sdr-fg', '--foreground', '--fg', '--text', '--color-foreground'],
  grid: ['--sdr-grid', '--grid', '--border', '--color-border'],
  trace: ['--sdr-trace', '--trace', '--primary', '--color-primary'],
  accent: ['--sdr-accent', '--accent', '--secondary', '--color-accent'],
}

/** Used when a variable is absent or unparseable, and off the main thread where there is no DOM. */
const DEFAULTS: Record<PaletteKey, Rgb> = {
  background: [0.04, 0.05, 0.07],
  surface: [0.08, 0.09, 0.12],
  foreground: [0.9, 0.93, 0.96],
  grid: [0.2, 0.22, 0.26],
  trace: [0.3, 0.8, 0.9],
  accent: [0.95, 0.6, 0.2],
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

function parseHex(text: string): Rgb | null {
  let hex = text.slice(1)
  if (hex.length === 3) {
    // Expand shorthand #abc to #aabbcc before slicing pairs.
    hex = hex.replace(/./g, (c) => c + c)
  }
  if (hex.length < 6) return null
  const r = parseInt(hex.slice(0, 2), 16)
  const g = parseInt(hex.slice(2, 4), 16)
  const b = parseInt(hex.slice(4, 6), 16)
  if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return null
  return [r / 255, g / 255, b / 255]
}

/** Parses any of the accepted colour forms into normalised floats, or null if it cannot. */
function parseColour(raw: string): Rgb | null {
  const text = raw.trim()
  if (text.length === 0) return null
  if (text.charCodeAt(0) === 0x23) return parseHex(text)

  // Drop an rgb()/rgba() wrapper if present; a bare channel triple survives unchanged.
  const inner = text.replace(/^rgba?\(/i, '').replace(/\)$/, '')
  // Channels may be separated by commas, whitespace, or the `/` that precedes an alpha.
  const parts = inner.split(/[\s,/]+/).filter((p) => p.length > 0)
  if (parts.length < 3) return null

  const r = Number(parts[0])
  const g = Number(parts[1])
  const b = Number(parts[2])
  if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) return null

  // The accepted non-hex forms all carry 0-255 channels; there is no 0-1 CSS colour syntax
  // that reaches here, so a single divide is unambiguous.
  return [clamp01(r / 255), clamp01(g / 255), clamp01(b / 255)]
}

/**
 * Resolves every palette key against the document root, falling back to the dark defaults.
 *
 * Returns a plain record rather than a typed-array so callers can feed individual colours
 * straight into uniform writes without reshaping.
 */
export function readPalette(): Record<PaletteKey, Rgb> {
  const palette: Record<PaletteKey, Rgb> = {
    background: [...DEFAULTS.background],
    surface: [...DEFAULTS.surface],
    foreground: [...DEFAULTS.foreground],
    grid: [...DEFAULTS.grid],
    trace: [...DEFAULTS.trace],
    accent: [...DEFAULTS.accent],
  }

  // Worklets and workers have no document; the defaults are the whole answer there.
  if (typeof document === 'undefined') return palette

  const style = getComputedStyle(document.documentElement)
  for (const key of Object.keys(CANDIDATES) as PaletteKey[]) {
    for (const variable of CANDIDATES[key]) {
      const parsed = parseColour(style.getPropertyValue(variable))
      if (parsed) {
        palette[key] = parsed
        break
      }
    }
  }
  return palette
}
