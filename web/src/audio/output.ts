/**
 * Sets up the audio graph and reports its health.
 *
 * The measurement side matters more than it looks. `AudioRenderCapacity` — which most
 * material still points at — was removed from the specification in early 2025 and never
 * shipped anywhere, and its replacement, `AudioContext.playbackStats`, exists only in
 * recent Chrome and is capped at one update a second as an anti-fingerprinting measure.
 * So the underrun count coming back from the worklet is the only portable way to know
 * whether the pipeline is keeping up, which is why the worklet bothers to keep it.
 */

import sinkUrl from './sink.worklet.ts?worker&url'

export interface AudioHealth {
  underruns: number
  framesDelivered: number
  /** How full the audio ring is, from 0 to 1. Persistently near zero means starvation. */
  fill: number
}

export class AudioOutput {
  private context: AudioContext | undefined
  private node: AudioWorkletNode | undefined
  private gain: GainNode | undefined
  private health: AudioHealth = { underruns: 0, framesDelivered: 0, fill: 0 }

  /**
   * Starts playback from `ring`, which carries interleaved stereo.
   *
   * Must be called from a user gesture — an `AudioContext` created outside one starts
   * suspended.
   */
  /**
   * Creates and resumes the AudioContext. Call this synchronously from the connect click,
   * before anything awaits: opening the WebUSB device chooser spends the click's user
   * activation, so a context created afterwards starts suspended and never makes a sound —
   * which is exactly the "connected, no errors, no audio" symptom. Creating it here, while
   * the gesture is still live, gets a running context.
   */
  async unlock(sampleRate: number): Promise<void> {
    if (!this.context) {
      this.context = new AudioContext({ sampleRate, latencyHint: 'interactive' })
    }
    if (this.context.state === 'suspended') {
      await this.context.resume().catch(() => {})
    }
  }

  async start(ring: SharedArrayBuffer, sampleRate: number): Promise<void> {
    // Asking for the pipeline's own rate avoids a resampling stage inside the browser,
    // which would add both latency and a filter nobody chose. Reuse the context unlock()
    // already created in the gesture if there is one.
    if (!this.context) {
      this.context = new AudioContext({ sampleRate, latencyHint: 'interactive' })
    }

    await this.context.audioWorklet.addModule(sinkUrl)

    this.node = new AudioWorkletNode(this.context, 'sdr-sink', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      // Delivered through the constructor rather than the port, because this arrives
      // synchronously and because the worklet-scope `port` is Firefox-only.
      processorOptions: { ring },
    })

    this.node.onprocessorerror = () => {
      // The specification permanently disables a processor that throws, so there is no
      // recovering this node — only rebuilding it.
      this.node = undefined
    }
    this.node.port.onmessage = (event) => {
      this.health = event.data as AudioHealth
    }

    this.gain = this.context.createGain()
    this.node.connect(this.gain)
    this.gain.connect(this.context.destination)

    // Autoplay policy leaves a context suspended until a gesture resumes it.
    if (this.context.state === 'suspended') {
      await this.context.resume().catch(() => {})
    }
    // If it is still suspended — the gesture was spent before we got here — resume on the
    // next user interaction anywhere in the page, then stop listening once it takes.
    if (this.context.state === 'suspended') {
      const events = ['pointerdown', 'keydown', 'touchend'] as const
      const resume = () => {
        void this.context?.resume().catch(() => {})
        if (this.context?.state === 'running') {
          events.forEach((event) => window.removeEventListener(event, resume))
        }
      }
      events.forEach((event) => window.addEventListener(event, resume))
    }
  }

  setVolume(value: number): void {
    if (!this.gain || !this.context) return
    // Ramp rather than assign: a step change in gain is a click.
    this.gain.gain.setTargetAtTime(
      Math.max(0, Math.min(1, value)),
      this.context.currentTime,
      0.02,
    )
  }

  /** Round-trip latency in seconds, as far as the browser will report it. */
  latency(): number {
    if (!this.context) return 0
    // `outputLatency` is the honest figure but reached Safari only recently, so fall back
    // to the part every browser reports.
    return this.context.outputLatency || this.context.baseLatency || 0
  }

  status(): AudioHealth {
    return this.health
  }

  get sampleRate(): number {
    return this.context?.sampleRate ?? 0
  }

  async stop(): Promise<void> {
    this.node?.port.postMessage('stop')
    this.node?.disconnect()
    this.gain?.disconnect()
    await this.context?.close().catch(() => {})
    this.context = undefined
    this.node = undefined
    this.gain = undefined
  }
}
