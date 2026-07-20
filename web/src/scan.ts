/**
 * Station scanning.
 *
 * Rather than dwelling on every channel across a band — minutes of tuning — the scanner
 * uses the receiver the way an SDR actually can: each capture is 2.4 MHz wide, so one dwell
 * sees a dozen FM channels at once. It steps the centre across the band in overlapping
 * windows, reads the spectrum at each, and picks out the peaks that stand above the noise
 * floor. Every peak maps back to an absolute frequency and is catalogued.
 *
 * The same peak-finding drives `seek`, which the radio's up/down buttons use to jump to the
 * next station.
 */

import type { Pipeline } from './pipeline.js'
import type { DemodModeName } from './workers/protocol.js'

export interface FoundStation {
  frequencyHz: number
  /** How far the carrier stands above the band's noise floor, in dB. */
  strengthDb: number
}

export interface ScanBand {
  label: string
  startHz: number
  endHz: number
  /** Channel grid the region uses; found frequencies snap to it. */
  channelSpacingHz: number
  mode: DemodModeName
}

/**
 * Broadcast bands. FM is on the 100 kHz grid used across most of the world; AM medium wave
 * is on the 9 kHz grid used outside the Americas. Both are starting points a user can tune
 * away from.
 */
export const SCAN_BANDS: Record<'fm' | 'am', ScanBand> = {
  fm: {
    label: 'FM',
    startHz: 87_500_000,
    endHz: 108_000_000,
    channelSpacingHz: 100_000,
    mode: 'wfm',
  },
  am: {
    label: 'AM',
    startHz: 522_000,
    endHz: 1_710_000,
    channelSpacingHz: 9_000,
    mode: 'am',
  },
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/** dB a peak must stand above the noise floor to count as a station. */
const DEFAULT_THRESHOLD_DB = 8

/** How long to let the front end settle after retuning before reading the spectrum. */
const DEFAULT_DWELL_MS = 320

export interface ScanOptions {
  thresholdDb?: number
  dwellMs?: number
  onProgress?: (fraction: number) => void
  onStation?: (station: FoundStation) => void
}

export class Scanner {
  private cancelled = false
  private scanning = false

  constructor(private readonly pipeline: Pipeline) {}

  get isScanning(): boolean {
    return this.scanning
  }

  cancel(): void {
    this.cancelled = true
  }

  /**
   * Sweeps a whole band and returns the stations found, sorted by frequency. Also reports
   * each one through `onStation` as it is discovered, so the catalogue fills in live.
   */
  async scan(band: ScanBand, options: ScanOptions = {}): Promise<FoundStation[]> {
    const threshold = options.thresholdDb ?? DEFAULT_THRESHOLD_DB
    const dwell = options.dwellMs ?? DEFAULT_DWELL_MS
    const found: FoundStation[] = []

    this.cancelled = false
    this.scanning = true
    try {
      const span = this.pipeline.sampleRate
      // Overlap the windows so a station sitting near a window edge is not missed.
      const stepHz = Math.floor(span * 0.8)
      const half = span / 2

      for (let centre = band.startHz + half; centre <= band.endHz + half; centre += stepHz) {
        if (this.cancelled) break
        this.pipeline.tune(Math.round(centre))
        options.onProgress?.(
          Math.min(1, (centre - band.startHz) / (band.endHz - band.startHz)),
        )
        await sleep(dwell)
        if (this.cancelled) break

        const bins = this.pipeline.peekSpectrum()
        if (!bins) continue

        for (const peak of detectPeaks(bins, threshold)) {
          const freq = snap(
            centre + binToOffset(peak.bin, bins.length, span),
            band.channelSpacingHz,
          )
          if (freq < band.startHz || freq > band.endHz) continue

          const near = found.find(
            (s) => Math.abs(s.frequencyHz - freq) < band.channelSpacingHz / 2,
          )
          if (near) {
            near.strengthDb = Math.max(near.strengthDb, peak.snrDb)
            continue
          }
          const station: FoundStation = { frequencyHz: freq, strengthDb: peak.snrDb }
          found.push(station)
          options.onStation?.(station)
        }
      }
    } finally {
      this.scanning = false
    }

    found.sort((a, b) => a.frequencyHz - b.frequencyHz)
    return found
  }

  /**
   * Tunes to the next station above or below the current frequency. Looks within the
   * current 2.4 MHz window first (instant), and only retunes the front end if there is
   * nothing there. Returns the frequency landed on, or null at the band edge.
   */
  async seek(
    direction: 1 | -1,
    band: ScanBand,
    thresholdDb = DEFAULT_THRESHOLD_DB,
  ): Promise<number | null> {
    const span = this.pipeline.sampleRate
    // A guard so seek does not immediately re-find the station it is sitting on.
    const guard = band.channelSpacingHz * 1.5
    let from = this.pipeline.currentFrequency

    for (let attempt = 0; attempt < 24; attempt++) {
      const bins = this.pipeline.peekSpectrum()
      if (bins) {
        const centre = this.pipeline.currentFrequency
        const candidates = detectPeaks(bins, thresholdDb)
          .map((p) => ({
            freq: snap(centre + binToOffset(p.bin, bins.length, span), band.channelSpacingHz),
            snr: p.snrDb,
          }))
          .filter(
            (c) =>
              c.freq >= band.startHz &&
              c.freq <= band.endHz &&
              direction * (c.freq - from) > guard,
          )
          .sort((a, b) => direction * (a.freq - b.freq))

        const next = candidates[0]
        if (next) {
          this.pipeline.tune(next.freq)
          return next.freq
        }
      }

      // Nothing in this window; move the front end most of a window onward and try again.
      const nextCentre = this.pipeline.currentFrequency + direction * Math.floor(span * 0.8)
      if (nextCentre < band.startHz || nextCentre > band.endHz) return null
      from = this.pipeline.currentFrequency
      this.pipeline.tune(nextCentre)
      await sleep(DEFAULT_DWELL_MS)
    }
    return null
  }
}

interface Peak {
  bin: number
  snrDb: number
}

/**
 * Finds spectral peaks that stand at least `thresholdDb` above the noise floor.
 *
 * The floor is the 30th-percentile bin — robust against the handful of strong carriers that
 * would drag a mean upward. A peak is a local maximum above the floor plus threshold, and
 * peaks are kept apart by a minimum bin separation so one broad carrier is not counted as
 * several.
 */
function detectPeaks(bins: Float32Array, thresholdDb: number): Peak[] {
  const n = bins.length
  const sorted = Float32Array.from(bins).sort()
  const floor = sorted[Math.floor(n * 0.3)] ?? -120
  const cutoff = floor + thresholdDb

  // Ignore the bins nearest DC: the tuner's own residual carrier sits there and is not a
  // station.
  const dcGuard = Math.max(2, Math.floor(n * 0.01))
  const centre = n / 2
  const minSeparation = Math.max(3, Math.floor(n * 0.01))

  const peaks: Peak[] = []
  for (let k = 1; k < n - 1; k++) {
    if (Math.abs(k - centre) < dcGuard) continue
    const v = bins[k]!
    if (v < cutoff) continue
    if (v < bins[k - 1]! || v < bins[k + 1]!) continue

    const last = peaks[peaks.length - 1]
    if (last && k - last.bin < minSeparation) {
      if (v > bins[last.bin]!) {
        last.bin = k
        last.snrDb = v - floor
      }
      continue
    }
    peaks.push({ bin: k, snrDb: v - floor })
  }
  return peaks
}

/** Absolute frequency offset, in hertz, of bin `k` from the window centre. */
function binToOffset(k: number, n: number, span: number): number {
  return (k - n / 2) * (span / n)
}

function snap(hz: number, grid: number): number {
  return Math.round(hz / grid) * grid
}
