/**
 * The demo display shown before a receiver is connected.
 *
 * An SDR app with no device is a blank canvas, which reads as broken rather than idle. This
 * draws a synthetic spectrum and scrolling waterfall — a few drifting carriers over a noise
 * floor — so the displays are alive from the first paint and it is obvious the app has
 * initialised. It is deliberately Canvas2D rather than the WebGPU renderers: it must show
 * something even where WebGPU is unavailable, and it needs no device, no workers, and no
 * WebAssembly.
 *
 * The synthetic spectrum is generated the same way the real one is read — dB values across
 * a band — so the demo looks like what the live waterfall will.
 */

interface Carrier {
  centre: number // fractional position across the band, 0..1
  drift: number // cycles per second of slow lateral movement
  width: number // fraction of the band
  strength: number // dB above the floor
  modulation: number // how much its level breathes
}

/** A handful of stations that drift and breathe, so the display is never static. */
const CARRIERS: Carrier[] = [
  { centre: 0.18, drift: 0.013, width: 0.02, strength: 42, modulation: 6 },
  { centre: 0.34, drift: -0.009, width: 0.015, strength: 30, modulation: 10 },
  { centre: 0.52, drift: 0.006, width: 0.03, strength: 48, modulation: 4 },
  { centre: 0.67, drift: -0.017, width: 0.012, strength: 24, modulation: 12 },
  { centre: 0.83, drift: 0.011, width: 0.022, strength: 36, modulation: 8 },
]

const FLOOR_DB = -95
const CEIL_DB = -25
const BINS = 512

export class DemoDisplay {
  private readonly ctx: CanvasRenderingContext2D
  private readonly spectrum = new Float32Array(BINS)
  private raf = 0
  private running = false
  private startTime = 0
  private width = 0
  private height = 0
  private waterfallTop = 0

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
    // Split: spectrum trace on top third, waterfall below.
    this.waterfallTop = Math.round(this.height * 0.34)
    this.ctx.fillStyle = '#0a0c12'
    this.ctx.fillRect(0, 0, this.width, this.height)
  }

  start(): void {
    if (this.running) return
    this.running = true
    this.startTime = performance.now()
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

  private frame(): void {
    // Self-heal the backing-store size: the canvas may have been constructed before layout
    // gave it a width, and not every environment fires a ResizeObserver reliably. Cheap to
    // check, and only actually resizes on a real change.
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const wantW = Math.max(1, Math.round(this.canvas.clientWidth * dpr))
    if (wantW !== this.width && this.canvas.clientWidth > 0) this.resize()

    const t = (performance.now() - this.startTime) / 1000
    this.generate(t)
    this.drawWaterfallRow()
    this.drawSpectrum()
  }

  /** Fills `spectrum` with a synthetic band: noise floor plus the drifting carriers. */
  private generate(t: number): void {
    for (let k = 0; k < BINS; k++) {
      // A gently undulating noise floor.
      const x = k / BINS
      let db = FLOOR_DB + 6 * Math.sin(x * 40 + t * 0.7) * Math.sin(x * 7 - t * 0.3)
      db += (pseudoNoise(k, Math.floor(t * 30)) - 0.5) * 7

      for (const c of CARRIERS) {
        const centre = c.centre + Math.sin(t * c.drift * Math.PI * 2) * 0.03
        const d = (x - centre) / c.width
        const envelope = Math.exp(-d * d)
        const breath = 1 + Math.sin(t * (0.6 + c.drift * 20)) * (c.modulation / c.strength)
        db += c.strength * envelope * breath
        // A little sideband spread so carriers look like signals, not spikes.
        db += c.strength * 0.25 * Math.exp(-(d * d) / 9)
      }
      this.spectrum[k] = db
    }
  }

  private drawWaterfallRow(): void {
    const { ctx, width } = this
    const wfHeight = this.height - this.waterfallTop
    // Scroll everything down one pixel by drawing the region onto itself, then paint the
    // fresh row at the top of the waterfall.
    ctx.drawImage(
      this.canvas,
      0,
      this.waterfallTop,
      width,
      wfHeight - 1,
      0,
      this.waterfallTop + 1,
      width,
      wfHeight - 1,
    )
    const row = ctx.createImageData(width, 1)
    for (let px = 0; px < width; px++) {
      const db = this.spectrum[Math.floor((px / width) * BINS)]!
      const [r, g, b] = colormap((db - FLOOR_DB) / (CEIL_DB - FLOOR_DB))
      const o = px * 4
      row.data[o] = r
      row.data[o + 1] = g
      row.data[o + 2] = b
      row.data[o + 3] = 255
    }
    ctx.putImageData(row, 0, this.waterfallTop)
  }

  private drawSpectrum(): void {
    const { ctx, width } = this
    const h = this.waterfallTop
    ctx.fillStyle = '#0a0c12'
    ctx.fillRect(0, 0, width, h)

    // Faint reference gridlines.
    ctx.strokeStyle = 'rgba(120,140,180,0.10)'
    ctx.lineWidth = 1
    for (let i = 1; i < 4; i++) {
      const y = (h * i) / 4
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    ctx.beginPath()
    for (let px = 0; px < width; px++) {
      const db = this.spectrum[Math.floor((px / width) * BINS)]!
      const norm = clamp01((db - FLOOR_DB) / (CEIL_DB - FLOOR_DB))
      const y = h - norm * (h - 4) - 2
      if (px === 0) ctx.moveTo(px, y)
      else ctx.lineTo(px, y)
    }
    ctx.strokeStyle = 'rgba(90,200,160,0.95)'
    ctx.lineWidth = Math.max(1, Math.round((window.devicePixelRatio || 1)))
    ctx.stroke()
  }
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

/** Deterministic hash noise in [0,1), so the floor shimmers without a real RNG. */
function pseudoNoise(a: number, b: number): number {
  const s = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453
  return s - Math.floor(s)
}

/** A compact viridis-like ramp: dark navy → teal → green → yellow. */
function colormap(t: number): [number, number, number] {
  const x = clamp01(t)
  const stops: Array<[number, [number, number, number]]> = [
    [0.0, [8, 10, 20]],
    [0.25, [33, 60, 110]],
    [0.5, [30, 130, 130]],
    [0.75, [90, 200, 110]],
    [1.0, [245, 235, 120]],
  ]
  for (let i = 1; i < stops.length; i++) {
    const [p1, c1] = stops[i]!
    const [p0, c0] = stops[i - 1]!
    if (x <= p1) {
      const f = (x - p0) / (p1 - p0)
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ]
    }
  }
  return stops[stops.length - 1]![1]
}
