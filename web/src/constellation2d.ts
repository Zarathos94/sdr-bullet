/**
 * A Canvas2D constellation — the I/Q density plot, drawn without WebGPU.
 *
 * The GPU constellation is the nicer of the two when it works, but WebGPU is unreliable on
 * some machines (a hybrid-graphics laptop can lose the Vulkan device mid-render), and on those
 * the GPU plot is simply blank. This is the fallback: it reads the same interleaved baseband
 * the GPU path does and scatters it with a little persistence so the shape builds up. A
 * constant-envelope signal like FM draws a ring; noise fills a disc.
 */

interface Palette {
  surface: string
  grid: string
  primary: string
  muted: string
}

function token(name: string, alpha = 1): string {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  const rgb = raw || '130 180 255'
  return alpha >= 1 ? `rgb(${rgb})` : `rgba(${rgb.split(/\s+/).join(',')},${alpha})`
}

export class Constellation2D {
  private readonly ctx: CanvasRenderingContext2D
  private raf = 0
  private running = false
  private width = 0
  private height = 0
  private phase = 0
  private palette: Palette
  private paletteAge = 0
  // Interleaved I/Q, newest frame; null before a device is connected.
  private source: (() => Float32Array | null) | null = null

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d', { alpha: false })
    if (!ctx) throw new Error('2D canvas context unavailable')
    this.ctx = ctx
    this.palette = this.readPalette()
    this.resize()
  }

  private readPalette(): Palette {
    return {
      surface: token('--surface'),
      grid: token('--outline-variant', 0.7),
      primary: token('--primary'),
      muted: token('--on-surface-variant', 0.7),
    }
  }

  resize(): void {
    const rect = this.canvas.getBoundingClientRect()
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    this.width = Math.max(1, Math.round(rect.width * dpr))
    this.height = Math.max(1, Math.round(rect.height * dpr))
    this.canvas.width = this.width
    this.canvas.height = this.height
    this.ctx.fillStyle = this.palette.surface
    this.ctx.fillRect(0, 0, this.width, this.height)
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
    const wantH = Math.max(1, Math.round(this.canvas.clientHeight * dpr))
    if ((wantW !== this.width || wantH !== this.height) && this.canvas.clientWidth > 0) this.resize()
    if (++this.paletteAge >= 60) {
      this.paletteAge = 0
      this.palette = this.readPalette()
    }

    const ctx = this.ctx
    const { width: w, height: h } = this
    const pal = this.palette
    const cx = w / 2
    const cy = h / 2
    const radius = Math.min(w, h) * 0.44

    // Persistence: fade the previous frame toward the background rather than clearing, so the
    // points leave short trails and the shape accumulates.
    ctx.fillStyle = pal.surface
    ctx.globalAlpha = 0.18
    ctx.fillRect(0, 0, w, h)
    ctx.globalAlpha = 1

    // Axes + unit circle.
    ctx.strokeStyle = pal.grid
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(cx, 0)
    ctx.lineTo(cx, h)
    ctx.moveTo(0, cy)
    ctx.lineTo(w, cy)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(cx, cy, radius, 0, Math.PI * 2)
    ctx.stroke()

    const iq = this.source ? this.source() : null
    if (iq && iq.length >= 4) {
      const samples = iq.length >> 1
      // Auto-scale to the signal's own magnitude so the cloud fills the plot regardless of gain.
      let peak = 1e-6
      for (let k = 0; k < samples; k++) {
        const i = iq[k * 2]!
        const q = iq[k * 2 + 1]!
        const m = i * i + q * q
        if (m > peak) peak = m
      }
      const scale = radius / Math.sqrt(peak)
      ctx.fillStyle = pal.primary
      ctx.globalAlpha = 0.5
      // Cap the number of plotted points so a large frame does not stall the paint.
      const step = Math.max(1, Math.floor(samples / 2000))
      for (let k = 0; k < samples; k += step) {
        const x = cx + iq[k * 2]! * scale
        const y = cy - iq[k * 2 + 1]! * scale
        ctx.fillRect(x, y, 1.6, 1.6)
      }
      ctx.globalAlpha = 1
    } else {
      // Idle: a point orbiting the unit circle, so the plot reads as alive-but-waiting.
      this.phase += 0.04
      ctx.fillStyle = pal.muted
      const x = cx + Math.cos(this.phase) * radius
      const y = cy - Math.sin(this.phase) * radius
      ctx.beginPath()
      ctx.arc(x, y, 2.5, 0, Math.PI * 2)
      ctx.fill()
    }
  }
}
