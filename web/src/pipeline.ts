/**
 * Wires the workers, rings and audio graph together and exposes one control surface.
 *
 * Everything the UI touches goes through here, so the ordering constraints live in one
 * place rather than being rediscovered at each call site.
 */

import { AudioOutput, type AudioHealth } from './audio/output.js'
import { LatestFrame, allocateRing } from './ipc/ring.js'
import { allocateIqRing } from './ipc/iq-ring.js'
import { requestDevice, isSupported, type DeviceIdentity } from './usb/webusb.js'
import type {
  CaptureStatus,
  DemodModeName,
  DspStatus,
  FromWorker,
  ToCapture,
  ToDsp,
} from './workers/protocol.js'
import type { ToSpectrum } from './workers/spectrum.worker.js'

/**
 * Capture rate.
 *
 * 2.4 million samples a second divides by ten to a 240 kHz multiplex — which has room for
 * the stereo pilot at 19 kHz and the data subcarrier at 57 kHz — and then by five to 48 kHz
 * of audio. Integer ratios throughout, so no stage needs fractional resampling.
 */
export const CAPTURE_RATE = 2_400_000

/**
 * Ring capacity, in samples.
 *
 * About 350 ms at the capture rate. Long enough to ride out a scheduler hiccup or a
 * garbage collection pause, short enough that the delay between tuning and hearing the
 * result stays imperceptible.
 */
const IQ_RING_CAPACITY = 1 << 20

/** Audio ring, interleaved stereo. Roughly a second at 48 kHz. */
const AUDIO_RING_CAPACITY = 1 << 17

const FFT_SIZE = 2048
const SPECTRUM_ROWS_PER_SECOND = 60

export interface PipelineStatus {
  running: boolean
  device: DeviceIdentity | undefined
  simdBackend: string
  capture: CaptureStatus | undefined
  dsp: DspStatus | undefined
  audio: AudioHealth
  audioLatency: number
}

export type StatusListener = (status: PipelineStatus) => void
export type ErrorListener = (message: string, fatal: boolean) => void

export class Pipeline {
  private capture: Worker | undefined
  private dsp: Worker | undefined
  private spectrumWorker: Worker | undefined
  private readonly audio = new AudioOutput()

  private spectrumSink: LatestFrame | undefined
  private constellationSource: LatestFrame | undefined
  private readonly bins = new Float32Array(FFT_SIZE)
  private readonly constellationFrame = new Float32Array(FFT_SIZE * 2)

  private device: DeviceIdentity | undefined
  private simdBackend = 'unknown'
  private captureStatus: CaptureStatus | undefined
  private dspStatus: DspStatus | undefined
  private running = false

  private statusListener: StatusListener | undefined
  private errorListener: ErrorListener | undefined

  onStatus(listener: StatusListener): void {
    this.statusListener = listener
  }

  onError(listener: ErrorListener): void {
    this.errorListener = listener
  }

  /** Whether this browser can run the pipeline at all, with the reason if not. */
  static capabilities(): { ok: boolean; reason?: string } {
    if (!crossOriginIsolated) {
      return {
        ok: false,
        reason:
          'This page is not cross-origin isolated, so SharedArrayBuffer is unavailable and ' +
          'the workers cannot share memory. It must be served with ' +
          'Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: require-corp.',
      }
    }
    if (!isSupported()) {
      return {
        ok: false,
        reason:
          'This browser has no WebUSB, so it cannot reach the receiver. Chromium-based ' +
          'browsers support it; Mozilla and WebKit have both declined to implement it.',
      }
    }
    return { ok: true }
  }

  /** Opens the device chooser. Must be called from a user gesture, on the page. */
  async requestDevice(): Promise<DeviceIdentity> {
    this.device = await requestDevice()
    return this.device
  }

  /**
   * Starts capture, demodulation and audio.
   *
   * Must follow a successful `requestDevice`, and must itself run inside a user gesture —
   * an `AudioContext` created outside one starts suspended.
   */
  async start(centreHz: number, mode: DemodModeName, gainTenths = -1): Promise<void> {
    if (this.running) await this.stop()

    const iq = allocateIqRing(IQ_RING_CAPACITY)
    const audioRing = allocateRing(AUDIO_RING_CAPACITY)
    const spectrumSource = LatestFrame.allocate(FFT_SIZE * 2)
    const spectrumSink = LatestFrame.allocate(FFT_SIZE)
    const constellationSource = LatestFrame.allocate(FFT_SIZE * 2)
    this.spectrumSink = new LatestFrame(spectrumSink, FFT_SIZE)
    this.constellationSource = new LatestFrame(constellationSource, FFT_SIZE * 2)

    // Audio first: if the graph cannot be built there is no point opening the device, and
    // the failure is far easier to attribute this way round.
    await this.audio.start(audioRing, 48_000)

    this.spectrumWorker = new Worker(
      new URL('./workers/spectrum.worker.ts', import.meta.url),
      { type: 'module' },
    )
    this.post<ToSpectrum>(this.spectrumWorker, {
      type: 'start',
      config: {
        source: spectrumSource,
        sink: spectrumSink,
        fftSize: FFT_SIZE,
        rowsPerSecond: SPECTRUM_ROWS_PER_SECOND,
      },
    })

    this.dsp = new Worker(new URL('./workers/dsp.worker.ts', import.meta.url), {
      type: 'module',
    })
    this.dsp.onmessage = (event) => this.handle(event.data as FromWorker, 'dsp')
    this.post<ToDsp>(this.dsp, {
      type: 'start',
      config: {
        iq,
        audio: audioRing,
        sampleRate: CAPTURE_RATE,
        mode,
        // A block the consumer waits for. Large enough that per-call overhead is
        // negligible, small enough that it is a few milliseconds of latency, not tens.
        blockSize: 65536,
      },
    })

