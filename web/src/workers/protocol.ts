/**
 * Messages exchanged with the pipeline workers.
 *
 * Only setup and control travel this way. Sample data never does — it goes through the
 * shared ring buffers, because a structured clone both copies and schedules, and a stage
 * that only wakes when the event loop reaches it introduces jitter the audio can hear.
 */

import type { IqRingBuffers } from '../ipc/iq-ring.js'

export type DemodModeName = 'nfm' | 'wfm' | 'am' | 'usb' | 'lsb' | 'cw'

/** Numeric values must match the `DemodMode` enum exported from the WebAssembly module. */
export const MODE_VALUES: Record<DemodModeName, number> = {
  nfm: 0,
  wfm: 1,
  am: 2,
  usb: 3,
  lsb: 4,
  cw: 5,
}

export interface CaptureConfig {
  iq: IqRingBuffers
  /** Latest-frame slot feeding the spectrum display. */
  spectrum: SharedArrayBuffer
  /**
   * Latest-frame slot feeding the constellation display, read on the main thread.
   *
   * Separate from the spectrum slot because a latest-frame slot has a single consumer —
   * the spectrum worker drains one, the render loop drains the other. Both carry the same
   * interleaved baseband; publishing it twice is one extra memcpy of a few thousand floats.
   */
  constellation: SharedArrayBuffer
  spectrumFrameLength: number
  sampleRate: number
  centreHz: number
  /** Tenths of a decibel, or negative for automatic. */
  gainTenths: number
  isV4: boolean
}

export interface DspConfig {
  iq: IqRingBuffers
  audio: SharedArrayBuffer
  sampleRate: number
  mode: DemodModeName
  blockSize: number
}

export type ToCapture =
  | { type: 'start'; config: CaptureConfig }
  | { type: 'tune'; hz: number }
  | { type: 'gain'; tenths: number }
  | { type: 'biasTee'; enabled: boolean }
  | { type: 'correction'; ppm: number }
  | { type: 'stop' }

export type ToDsp =
  | { type: 'start'; config: DspConfig }
  | { type: 'mode'; mode: DemodModeName }
  | { type: 'offset'; hz: number }
  | { type: 'squelch'; enabled: boolean; threshold: number }
  | { type: 'deemphasis'; microseconds: number }
  | { type: 'mono'; forced: boolean }
  | { type: 'agc'; enabled: boolean }
  | { type: 'stop' }

/** Periodic health and decode state, for the display. */
export interface CaptureStatus {
  type: 'status'
  tunedHz: number
  sampleRate: number
  band: string
  tunerFrequencyHz: number
  locked: boolean
  /** Samples discarded because a downstream stage could not keep up. */
  dropped: number
  bytesPerSecond: number
}

export interface DspStatus {
  type: 'status'
  stereo: boolean
  squelchOpen: boolean
  pilotLevel: number
  ringFill: number
  rds: {
    synchronised: boolean
    stationName: string
    radioText: string
    programId: number
    blockErrorRate: number
  }
}

export type FromWorker =
  | { type: 'ready'; simdBackend: string }
  | { type: 'error'; message: string; fatal: boolean }
  | CaptureStatus
  | DspStatus
