/**
 * Wires the workers, rings and audio graph together and exposes one control surface.
 *
 * Everything the UI touches goes through here, so the ordering constraints live in one
 * place rather than being rediscovered at each call site.
 */

import { AudioOutput, type AudioHealth } from './audio/output.js'
import { LatestFrame, allocateRing } from './ipc/ring.js'
import { allocateIqRing } from './ipc/iq-ring.js'
import {
  requestDevice,
  grantedDevices,
  identify,
  isSupported,
  type DeviceIdentity,
} from './usb/webusb.js'
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
  private readonly peekBins = new Float32Array(FFT_SIZE)
  private readonly peekConstellationFrame = new Float32Array(FFT_SIZE * 2)

  private device: DeviceIdentity | undefined
  private simdBackend = 'unknown'
  private captureStatus: CaptureStatus | undefined
  private dspStatus: DspStatus | undefined
  private running = false
  private tunedHz = 0
  private actualSampleRate = CAPTURE_RATE

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

  /** Whether a compatible RTL-SDR has already been permitted, so connecting needs no chooser. */
  async hasPermittedDevice(): Promise<boolean> {
    return (await grantedDevices()).length > 0
  }

  /**
   * Resolves the device to open. Prefers one the user has already permitted — `getDevices()`
   * returns only granted devices and shows no UI, so a return visit connects with no chooser
   * popping up. The chooser (`requestDevice`) is only opened to grant a *new* device, which
   * the browser requires happen inside a user gesture on the page.
   */
  async requestDevice(): Promise<DeviceIdentity> {
    const permitted = await grantedDevices()
    if (permitted[0]) {
      this.device = identify(permitted[0])
      return this.device
    }
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
    this.tunedHz = centreHz

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

    this.capture = new Worker(new URL('./workers/capture.worker.ts', import.meta.url), {
      type: 'module',
    })
    this.capture.onmessage = (event) => this.handle(event.data as FromWorker, 'capture')
    this.post<ToCapture>(this.capture, {
      type: 'start',
      config: {
        iq,
        spectrum: spectrumSource,
        constellation: constellationSource,
        spectrumFrameLength: FFT_SIZE * 2,
        sampleRate: CAPTURE_RATE,
        centreHz,
        gainTenths,
        isV4: this.device?.isV4 ?? false,
      },
    })

    this.running = true
  }

  private post<T>(worker: Worker, message: T): void {
    worker.postMessage(message)
  }

  private handle(message: FromWorker, source: 'capture' | 'dsp'): void {
    switch (message.type) {
      case 'ready':
        this.simdBackend = message.simdBackend
        break
      case 'error':
        this.errorListener?.(message.message, message.fatal)
        if (message.fatal) void this.stop()
        break
      case 'status':
        if (source === 'capture') {
          const capture = message as CaptureStatus
          this.captureStatus = capture
          this.tunedHz = capture.tunedHz
          this.actualSampleRate = capture.sampleRate
        } else {
          this.dspStatus = message as DspStatus
        }
        break
    }
    this.statusListener?.(this.status())
  }

  /** Copies the newest spectrum row, or returns null if none has arrived since last call. */
  latestSpectrum(): Float32Array | null {
    if (!this.spectrumSink) return null
    return this.spectrumSink.consume(this.bins) ? this.bins : null
  }

  /**
   * The newest spectrum row without consuming it — dB bins in display order, lowest
   * frequency first. The scanner reads this while the render loop keeps consuming normally.
   */
  peekSpectrum(): Float32Array | null {
    if (!this.spectrumSink) return null
    return this.spectrumSink.peek(this.peekBins) ? this.peekBins : null
  }

  /** Number of spectrum bins, i.e. the span of a peeked row across the sample rate. */
  get spectrumBins(): number {
    return FFT_SIZE
  }

  /** The rate actually achieved, so a bin index maps to an absolute frequency. */
  get sampleRate(): number {
    return this.actualSampleRate
  }

  get currentFrequency(): number {
    return this.tunedHz
  }

  /**
   * The newest baseband frame as interleaved I/Q, or null if none is new.
   *
   * Feeds the constellation. This is the wideband capture rather than the demodulated
   * channel — for a constant-envelope mode like FM it reads as a ring, which is the
   * expected shape, and it avoids threading a second frame out of the DSP worker.
   */
  latestConstellation(): Float32Array | null {
    if (!this.constellationSource) return null
    return this.constellationSource.consume(this.constellationFrame)
      ? this.constellationFrame
      : null
  }

  /**
   * The newest baseband frame without consuming it, so the scope can read the same I/Q the
   * render loop's constellation is already draining — the peek/consume split the scanner
   * uses for the spectrum.
   */
  peekConstellation(): Float32Array | null {
    if (!this.constellationSource) return null
    return this.constellationSource.peek(this.peekConstellationFrame)
      ? this.peekConstellationFrame
      : null
  }

  tune(hz: number): void {
    this.tunedHz = hz
    if (this.capture) this.post<ToCapture>(this.capture, { type: 'tune', hz })
  }

  /**
   * Moves the wanted channel within the captured bandwidth, without retuning the hardware.
   *
   * Cheaper than a retune, and it leaves the display still while the channel marker moves.
   */
  setChannelOffset(hz: number): void {
    if (this.dsp) this.post<ToDsp>(this.dsp, { type: 'offset', hz })
  }

  setGain(tenths: number): void {
    if (this.capture) this.post<ToCapture>(this.capture, { type: 'gain', tenths })
  }

  setSquelch(enabled: boolean, threshold: number): void {
    if (this.dsp) this.post<ToDsp>(this.dsp, { type: 'squelch', enabled, threshold })
  }

  setDeemphasis(microseconds: number): void {
    if (this.dsp) this.post<ToDsp>(this.dsp, { type: 'deemphasis', microseconds })
  }

  setForcedMono(forced: boolean): void {
    if (this.dsp) this.post<ToDsp>(this.dsp, { type: 'mono', forced })
  }

  setAgc(enabled: boolean): void {
    if (this.dsp) this.post<ToDsp>(this.dsp, { type: 'agc', enabled })
  }

  setBiasTee(enabled: boolean): void {
    if (this.capture) this.post<ToCapture>(this.capture, { type: 'biasTee', enabled })
  }

  setFrequencyCorrection(ppm: number): void {
    if (this.capture) this.post<ToCapture>(this.capture, { type: 'correction', ppm })
  }

  setVolume(value: number): void {
    this.audio.setVolume(value)
  }

  /**
   * Unlocks the audio context. Must be called synchronously from the connect gesture, before
   * the device chooser spends the user activation — otherwise the context is born suspended
   * and stays silent. See {@link AudioOutput.unlock}.
   */
  async unlockAudio(): Promise<void> {
    await this.audio.unlock(48_000)
  }

  setSpectrumSmoothing(alpha: number): void {
    if (this.spectrumWorker) {
      this.post<ToSpectrum>(this.spectrumWorker, { type: 'smoothing', alpha })
    }
  }

  status(): PipelineStatus {
    return {
      running: this.running,
      device: this.device,
      simdBackend: this.simdBackend,
      capture: this.captureStatus,
      dsp: this.dspStatus,
      audio: this.audio.status(),
      audioLatency: this.audio.latency(),
    }
  }

  async stop(): Promise<void> {
    this.running = false

    // Capture first, so the stages downstream drain rather than being cut off mid-block.
    if (this.capture) this.post<ToCapture>(this.capture, { type: 'stop' })
    if (this.dsp) this.post<ToDsp>(this.dsp, { type: 'stop' })
    if (this.spectrumWorker) this.post<ToSpectrum>(this.spectrumWorker, { type: 'stop' })

    await this.audio.stop()

    // Give the workers a moment to close their devices before terminating them; a
    // terminate mid-transfer leaves the interface claimed until the page is reloaded.
    await new Promise((resolve) => setTimeout(resolve, 100))

    this.capture?.terminate()
    this.dsp?.terminate()
    this.spectrumWorker?.terminate()
    this.capture = undefined
    this.dsp = undefined
    this.spectrumWorker = undefined
    this.spectrumSink = undefined
    this.constellationSource = undefined
    this.captureStatus = undefined
    this.dspStatus = undefined
  }
}
