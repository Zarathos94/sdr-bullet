/**
 * A time-domain oscilloscope over the baseband I/Q — the "scope" every SDR and audio tool
 * has, and the one view the spectrum and waterfall cannot give you: it shows the signal as a
 * wave in time rather than a distribution in frequency. Two traces, I and Q, drawn across
 * the width; a strong carrier is a clean pair of sinusoids, noise is a fuzzy band.
 *
 * Canvas2D on purpose, for the same reasons as the demo display: it must draw where WebGPU
 * is unavailable, needs no device, and reads the same latest-frame slot the constellation
 * does (interleaved I/Q), so it costs nothing extra in the pipeline.
 */

/** Reads an `r g b` design token off the document and returns a CSS colour. */
function token(name: string, alpha = 1): string {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  const rgb = raw || '130 180 255'
  return alpha >= 1 ? `rgb(${rgb})` : `rgba(${rgb.split(/\s+/).join(',')},${alpha})`
}

export class ScopeDisplay {
  private readonly ctx: CanvasRenderingContext2D
  private raf = 0
  private running = false
  private width = 0
  private height = 0
  private phase = 0
  // Interleaved I/Q, newest frame; null before a device is connected.
  private source: (() => Float32Array | null) | null = null

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d', { alpha: false })
    if (!ctx) throw new Error('2D canvas context unavailable')
    this.ctx = ctx
    this.resize()
  }

  resize(): void {
    const rect = this.canvas.getBoundingClientRect()
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    this.width = Math.max(1, Math.round(rect.width * dpr))
    this.height = Math.max(1, Math.round(rect.height * dpr))
    this.canvas.width = this.width
    this.canvas.height = this.height
  }

  start(): void {
    if (this.running) return
    this.running = true
    const loop = () => {
      if (!this.running) return
      this.frame()
      this.raf = requestAnimationFrame(loop)
    }
    this.raf = requestAnimationFrame(loop)
  }

  stop(): void {
    this.running = false
    cancelAnimationFrame(this.raf)
  }

  setSource(source: (() => Float32Array | null) | null): void {
    this.source = source
  }

  private frame(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const wantW = Math.max(1, Math.round(this.canvas.clientWidth * dpr))
    if (wantW !== this.width && this.canvas.clientWidth > 0) this.resize()

    const ctx = this.ctx
    const { width: w, height: h } = this
    ctx.fillStyle = token('--surface')
    ctx.fillRect(0, 0, w, h)

    // Graticule.
    ctx.strokeStyle = token('--outline-variant', 0.7)
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let i = 1; i < 8; i++) {
      const x = Math.round((w * i) / 8) + 0.5
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
    }
    for (let i = 1; i < 4; i++) {
      const y = Math.round((h * i) / 4) + 0.5
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
    }
    ctx.stroke()
    // Zero line.
    ctx.strokeStyle = token('--outline', 0.9)
    ctx.beginPath()
    ctx.moveTo(0, Math.round(h / 2) + 0.5)
    ctx.lineTo(w, Math.round(h / 2) + 0.5)
    ctx.stroke()

    const iq = this.source ? this.source() : null
    if (iq && iq.length >= 4) {
      // I on the primary colour, Q on the muted variant, so the two are distinguishable.
      this.trace(iq, 0, token('--primary'), 1.9)
      this.trace(iq, 1, token('--on-surface-variant', 0.85), 1.4)
    } else {
      // Idle: a slow sweep so the scope reads as alive-but-waiting rather than dead.
      this.phase += 0.03
      ctx.strokeStyle = token('--on-surface-variant', 0.5)
      ctx.lineWidth = 1.6
      ctx.beginPath()
      for (let x = 0; x < w; x++) {
        const y = h / 2 + Math.sin(x * 0.02 + this.phase) * h * 0.12
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.stroke()
    }
  }

  /** Draws one interleaved channel (offset 0 = I, 1 = Q) as a trace across the full width. */
  private trace(iq: Float32Array, offset: number, colour: string, lineWidth: number): void {
    const { width: w, height: h } = this
    const samples = iq.length >> 1
    // A window wide enough to show a few cycles of a strong carrier without aliasing the line.
    const span = Math.min(samples, 1024)
    const mid = h / 2
    // Fixed ±1.0 full-scale; the baseband floats sit well inside that, so gain that up a touch.
    const scale = h * 0.42
    this.ctx.strokeStyle = colour
    this.ctx.lineWidth = lineWidth
    this.ctx.lineJoin = 'round'
    this.ctx.beginPath()
    for (let x = 0; x < w; x++) {
      const idx = Math.min(span - 1, Math.floor((x / w) * span))
      const v = iq[idx * 2 + offset] ?? 0
      const y = mid - Math.max(-1, Math.min(1, v * 2)) * scale
      x === 0 ? this.ctx.moveTo(x, y) : this.ctx.lineTo(x, y)
    }
    this.ctx.stroke()
  }
}
