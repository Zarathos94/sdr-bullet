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
  async start(ring: SharedArrayBuffer, sampleRate: number): Promise<void> {
    // Asking for the pipeline's own rate avoids a resampling stage inside the browser,
    // which would add both latency and a filter nobody chose.
    this.context = new AudioContext({ sampleRate, latencyHint: 'interactive' })

    await this.context.audioWorklet.addModule(sinkUrl)

    this.node = new AudioWorkletNode(this.context, 'sdr-sink', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      // Delivered through the constructor rather than the port, because this arrives
      // synchronously and because the worklet-scope `port` is Firefox-only.
      processorOptions: { ring },
    })

