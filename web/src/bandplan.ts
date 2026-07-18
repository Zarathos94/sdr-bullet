/**
 * Band plan and sensible defaults.
 *
 * Not a complete allocation table — just enough that the app opens on something a user can
 * actually hear, and that each mode starts with settings suited to it rather than to
 * whatever the previous mode used. The presets exist because "tune to 100 MHz in USB with a
 * 500 Hz filter" is a frustrating first experience, and the fix is knowing that 100 MHz is
 * broadcast FM.
 */

import type { DemodModeName } from './workers/protocol.js'

export interface BandPreset {
  label: string
  frequencyHz: number
  mode: DemodModeName
  /** De-emphasis in microseconds; only meaningful for wideband FM. */
  deemphasisUs?: number
}

/**
 * Regional de-emphasis default.
 *
 * 50 microseconds across Europe, Africa, Asia and Australia; 75 in the Americas. Getting it
 * wrong only makes the treble sound dull or harsh, never broken, so a single default is
 * fine with an override in the controls.
 */
export const DEFAULT_DEEMPHASIS_US = 50

export const PRESETS: BandPreset[] = [
  { label: 'FM broadcast', frequencyHz: 98_000_000, mode: 'wfm', deemphasisUs: DEFAULT_DEEMPHASIS_US },
  { label: 'Air band', frequencyHz: 124_000_000, mode: 'am' },
  { label: '2 m amateur', frequencyHz: 145_000_000, mode: 'nfm' },
  { label: '70 cm amateur', frequencyHz: 433_500_000, mode: 'nfm' },
  { label: 'Marine VHF', frequencyHz: 156_800_000, mode: 'nfm' },
  { label: '40 m amateur (LSB)', frequencyHz: 7_100_000, mode: 'lsb' },
  { label: '20 m amateur (USB)', frequencyHz: 14_200_000, mode: 'usb' },
  { label: 'HF WWV time', frequencyHz: 10_000_000, mode: 'am' },
]

/** Reasonable starting settings for a mode, so switching mode is not also a chore. */
export interface ModeDefaults {
  /** Whether the channel filter should squelch on silence by default. */
  squelch: boolean
  squelchThreshold: number
  /** Whether the audio automatic gain control starts on. */
  agc: boolean
}

export function defaultsFor(mode: DemodModeName): ModeDefaults {
  switch (mode) {
    case 'nfm':
      // Repeater and simplex voice sits idle most of the time; squelch keeps the hiss out.
      return { squelch: true, squelchThreshold: 0.08, agc: true }
    case 'wfm':
      // Broadcast is continuous, so squelch would only ever cut the quiet passages.
      return { squelch: false, squelchThreshold: 0.08, agc: false }
    case 'am':
      return { squelch: false, squelchThreshold: 0.05, agc: true }
    case 'usb':
    case 'lsb':
    case 'cw':
      // Single sideband and Morse are weak-signal modes where the gain control earns its
      // keep and a squelch would clip the start of every transmission.
      return { squelch: false, squelchThreshold: 0.05, agc: true }
  }
}

/** Formats a frequency for display, choosing units by magnitude. */
export function formatFrequency(hz: number): { value: string; unit: string } {
  if (hz >= 1_000_000_000) {
    return { value: (hz / 1e9).toFixed(6), unit: 'GHz' }
  }
  if (hz >= 1_000_000) {
    return { value: (hz / 1e6).toFixed(4), unit: 'MHz' }
  }
  if (hz >= 1_000) {
    return { value: (hz / 1e3).toFixed(3), unit: 'kHz' }
  }
  return { value: hz.toFixed(0), unit: 'Hz' }
}

/** Parses a frequency, accepting a unit suffix or plain hertz. */
export function parseFrequency(text: string): number | null {
  const trimmed = text.trim()
  const match = trimmed.match(/^([\d.]+)\s*([kKmMgG]?)(?:Hz)?$/)
  if (!match) return null
  const value = Number.parseFloat(match[1]!)
  if (!Number.isFinite(value)) return null
  const scale = { k: 1e3, m: 1e6, g: 1e9, '': 1 }[match[2]!.toLowerCase()] ?? 1
  return Math.round(value * scale)
}

/** Lowest and highest the R828D can reach, upconverter included. */
export const MIN_FREQUENCY_HZ = 100_000
export const MAX_FREQUENCY_HZ = 1_766_000_000
