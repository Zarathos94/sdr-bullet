/**
 * Paired ring buffers carrying deinterleaved baseband.
 *
 * The pipeline works on separate in-phase and quadrature arrays throughout, because that
 * turns complex arithmetic into parallel real arithmetic and removes lane shuffles from
 * every vector kernel. Two independent rings would preserve that, but they can drift: if
 * one accepts a block and the other rejects it for want of space, every subsequent sample
 * is paired with the wrong partner and the signal is quietly destroyed rather than
 * interrupted.
 *
 * This wrapper makes that impossible by checking both before committing to either.
 */

import { RingConsumer, RingProducer, allocateRing } from './ring.js'

export interface IqRingBuffers {
  i: SharedArrayBuffer
  q: SharedArrayBuffer
}

export function allocateIqRing(capacity: number): IqRingBuffers {
  return { i: allocateRing(capacity), q: allocateRing(capacity) }
}

export class IqProducer {
  private readonly i: RingProducer
  private readonly q: RingProducer

  constructor(buffers: IqRingBuffers) {
    this.i = new RingProducer(buffers.i)
    this.q = new RingProducer(buffers.q)
  }

  /**
   * Writes a matched pair, or drops both.
   *
   * Returns whether the pair was accepted.
   */
  write(i: Float32Array, q: Float32Array): boolean {
    const count = Math.min(i.length, q.length)
    // Both must have room before either is committed, or the two fall out of step.
    if (this.i.available() < count || this.q.available() < count) {
      return false
    }
    this.i.write(i.subarray(0, count))
    this.q.write(q.subarray(0, count))
    return true
  }

  dropped(): number {
    return this.i.dropped()
  }

  close(): void {
    this.i.close()
    this.q.close()
  }
}

export class IqConsumer {
  private readonly i: RingConsumer
  private readonly q: RingConsumer

  constructor(buffers: IqRingBuffers) {
    this.i = new RingConsumer(buffers.i)
    this.q = new RingConsumer(buffers.q)
  }

  available(): number {
    return Math.min(this.i.available(), this.q.available())
  }

  fill(): number {
    return this.i.fill()
  }

  /** Reads a matched pair, returning how many samples were taken from each. */
  read(i: Float32Array, q: Float32Array): number {
    // Bound by the shorter side so the two indices advance by the same amount even if a
    // producer somehow got ahead on one.
    const count = Math.min(this.available(), i.length, q.length)
    if (count === 0) return 0
    this.i.read(i.subarray(0, count))
    this.q.read(q.subarray(0, count))
    return count
  }

  /** Blocks until `wanted` samples are ready. Dedicated workers only. */
  waitFor(wanted: number, timeoutMs = 100): boolean {
    return this.i.waitFor(wanted, timeoutMs) && this.available() >= wanted
  }

  clear(): void {
    this.i.clear()
    this.q.clear()
  }
}
