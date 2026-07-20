/**
 * Lock-free single-producer, single-consumer ring buffers over SharedArrayBuffer.
 *
 * This is the join between pipeline stages, each of which lives in its own worker with its
 * own WebAssembly instance. Passing samples by `postMessage` would work, but it copies and
 * it schedules — a structured clone lands as a task, so a stage only wakes when the event
 * loop gets round to it. At 2.4 million samples a second that jitter is audible.
 *
 * Correctness rests on one rule: the payload is written with ordinary stores, and only
 * then is the index published with an atomic store. A reader that observes the new index
 * is therefore guaranteed to observe the payload too, because JavaScript's `Atomics` are
 * sequentially consistent and an ordinary write cannot be reordered past one. Publishing
 * the index first — or using an ordinary write for it — reads as working code and
 * produces torn samples under load.
 */

/** Header slots, in `Int32Array` units. */
const enum Slot {
  Write = 0,
  Read = 1,
  Capacity = 2,
  /** Samples the producer had to discard because the consumer fell behind. */
  Dropped = 3,
  /** Set by the producer to release a blocked consumer at shutdown. */
  Closed = 4,
}

/**
 * Header length in `Int32Array` units.
 *
 * Larger than the five slots in use so the producer's and consumer's indices land in
 * different cache lines. Two cores writing adjacent words otherwise pass the line back and
 * forth on every update, which costs far more than the padding.
 */
const HEADER_SLOTS = 32
const HEADER_BYTES = HEADER_SLOTS * 4

/**
 * Allocates a ring for `capacity` float samples.
 *
 * Capacity must be a power of two so wrapping is a mask rather than a division.
 */
export function allocateRing(capacity: number): SharedArrayBuffer {
  if (!Number.isInteger(Math.log2(capacity))) {
    throw new Error(`ring capacity must be a power of two, got ${capacity}`)
  }
  if (typeof SharedArrayBuffer === 'undefined') {
    throw new Error(
      'SharedArrayBuffer is unavailable. The page must be cross-origin isolated: ' +
        'serve it with Cross-Origin-Opener-Policy: same-origin and ' +
        'Cross-Origin-Embedder-Policy: require-corp.',
    )
  }

  const sab = new SharedArrayBuffer(HEADER_BYTES + capacity * 4)
  const header = new Int32Array(sab, 0, HEADER_SLOTS)
  Atomics.store(header, Slot.Capacity, capacity)
  return sab
}

/** Shared view construction, so the two ends cannot disagree about the layout. */
function views(sab: SharedArrayBuffer) {
  const header = new Int32Array(sab, 0, HEADER_SLOTS)
  const data = new Float32Array(sab, HEADER_BYTES)
  const capacity = Atomics.load(header, Slot.Capacity)
  if (capacity <= 0) {
    throw new Error('ring buffer header is uninitialised')
  }
  return { header, data, capacity, mask: capacity - 1 }
}

/**
 * Distance between two indices.
 *
 * Indices count monotonically and are allowed to wrap through 32 bits, so the difference
 * has to be taken modulo 2^32 rather than by plain subtraction. `>>> 0` does exactly that.
 * Without it, the moment the producer passes two billion samples — about fifteen minutes
 * at this capture rate — the buffer reports a negative fill and stops.
 */
function distance(a: number, b: number): number {
  return (a - b) >>> 0
}

/** The writing end of a ring. Exactly one worker may hold this. */
export class RingProducer {
  private readonly header: Int32Array
  private readonly data: Float32Array
  private readonly capacity: number
  private readonly mask: number

  constructor(private readonly sab: SharedArrayBuffer) {
    const v = views(sab)
    this.header = v.header
    this.data = v.data
    this.capacity = v.capacity
    this.mask = v.mask
  }

  /** Samples that would fit right now. */
  available(): number {
    const write = Atomics.load(this.header, Slot.Write)
    const read = Atomics.load(this.header, Slot.Read)
    return this.capacity - distance(write, read)
  }

  /**
   * Writes a block, or drops it if it will not fit.
   *
   * Dropping whole blocks rather than partial ones keeps the consumer's samples
   * contiguous: a half-written block would put a discontinuity mid-buffer, which in audio
   * is a click and in a spectrum is a smear across every bin.
   *
   * Returns whether the block was accepted.
   */
  write(block: Float32Array): boolean {
    const write = Atomics.load(this.header, Slot.Write)
    // Acquire: everything the consumer did before publishing this index is visible.
    const read = Atomics.load(this.header, Slot.Read)

    if (this.capacity - distance(write, read) < block.length) {
      Atomics.add(this.header, Slot.Dropped, block.length)
      return false
    }

    const start = write & this.mask
    const firstRun = Math.min(block.length, this.capacity - start)
    this.data.set(block.subarray(0, firstRun), start)
    if (firstRun < block.length) {
      this.data.set(block.subarray(firstRun), 0)
    }

    // Release: publishing the index last is what makes the payload above visible to a
    // consumer that observes it.
    Atomics.store(this.header, Slot.Write, (write + block.length) >>> 0)
    Atomics.notify(this.header, Slot.Write)
    return true
  }

  /** Total samples discarded because the consumer could not keep up. */
  dropped(): number {
    return Atomics.load(this.header, Slot.Dropped)
  }

  /** Releases a consumer blocked in `waitFor`. */
  close(): void {
    Atomics.store(this.header, Slot.Closed, 1)
    Atomics.notify(this.header, Slot.Write)
  }

  get buffer(): SharedArrayBuffer {
    return this.sab
  }
}

/** The reading end of a ring. Exactly one worker may hold this. */
export class RingConsumer {
  private readonly header: Int32Array
  private readonly data: Float32Array
  private readonly capacity: number
  private readonly mask: number

  constructor(sab: SharedArrayBuffer) {
    const v = views(sab)
    this.header = v.header
    this.data = v.data
    this.capacity = v.capacity
    this.mask = v.mask
  }

  /** Samples ready to read. */
  available(): number {
    // Acquire: pairs with the producer's release store.
    const write = Atomics.load(this.header, Slot.Write)
    const read = Atomics.load(this.header, Slot.Read)
    return distance(write, read)
  }

  /**
   * Reads up to `out.length` samples, returning how many were taken.
   *
   * Never blocks. A consumer that must not stall — an audio callback, most of all — calls
   * this and accepts a short read.
   */
  read(out: Float32Array): number {
    const read = Atomics.load(this.header, Slot.Read)
    const write = Atomics.load(this.header, Slot.Write)
    const count = Math.min(distance(write, read), out.length)
    if (count === 0) return 0

    const start = read & this.mask
    const firstRun = Math.min(count, this.capacity - start)
    out.set(this.data.subarray(start, start + firstRun), 0)
    if (firstRun < count) {
      out.set(this.data.subarray(0, count - firstRun), firstRun)
    }

    Atomics.store(this.header, Slot.Read, (read + count) >>> 0)
    return count
  }

  /**
   * Blocks until at least `wanted` samples are ready, the ring closes, or the timeout
   * expires. Returns whether the samples arrived.
   *
   * **Only callable from a dedicated worker.** Worklet and window agents are specified
   * with `[[CanBlock]] = false`, so `Atomics.wait` throws a `TypeError` there rather than
   * waiting — which is correct, since blocking an audio render thread is exactly the
   * failure this whole design exists to avoid.
   */
  waitFor(wanted: number, timeoutMs = 100): boolean {
    for (;;) {
      if (this.available() >= wanted) return true
      if (Atomics.load(this.header, Slot.Closed) === 1) return false

      // Re-read the index inside the wait so a value published between the check above
      // and the wait below is not missed — `Atomics.wait` returns immediately when the
      // observed value has already moved on.
      const observed = Atomics.load(this.header, Slot.Write)
      if (distance(observed, Atomics.load(this.header, Slot.Read)) >= wanted) return true

      const result = Atomics.wait(this.header, Slot.Write, observed, timeoutMs)
      if (result === 'timed-out') return this.available() >= wanted
    }
  }

  /** Discards everything buffered. Used after retuning, when the old samples are stale. */
  clear(): void {
    Atomics.store(this.header, Slot.Read, Atomics.load(this.header, Slot.Write))
  }

  /** How full the ring is, from 0 to 1. Surfaced as a pipeline health indicator. */
  fill(): number {
    return this.available() / this.capacity
  }
}

/**
 * A single slot holding only the most recent frame.
 *
 * The right structure when a late frame is worthless rather than merely delayed. A
 * spectrum display wants whatever arrived most recently; queueing frames it will never
 * draw only adds latency, and back-pressure from a slow display must not reach the capture
 * stage. Audio uses a real queue for the opposite reason — there, every sample matters.
 *
 * Double-buffered with a sequence counter, so a reader can tell whether the frame it just
 * copied was overwritten while it was copying.
 */
export class LatestFrame {
  private readonly header: Int32Array
  private readonly slots: [Float32Array, Float32Array]

  constructor(
    private readonly sab: SharedArrayBuffer,
    private readonly frameLength: number,
  ) {
    this.header = new Int32Array(sab, 0, HEADER_SLOTS)
    const data = new Float32Array(sab, HEADER_BYTES)
    this.slots = [
      data.subarray(0, frameLength),
      data.subarray(frameLength, frameLength * 2),
    ]
  }

  static allocate(frameLength: number): SharedArrayBuffer {
    return new SharedArrayBuffer(HEADER_BYTES + frameLength * 2 * 4)
  }

  /** Overwrites the pending frame. Never blocks and never fails. */
  publish(frame: Float32Array): void {
    const sequence = Atomics.load(this.header, Slot.Write)
    // Write into whichever slot the reader is not currently looking at.
    const target = this.slots[(sequence + 1) & 1]!
    target.set(frame.subarray(0, this.frameLength))
    Atomics.store(this.header, Slot.Write, (sequence + 1) >>> 0)
  }

  /**
   * Copies the newest frame into `out`, returning false if none has been published since
   * the last call.
   */
  consume(out: Float32Array): boolean {
    const sequence = Atomics.load(this.header, Slot.Write)
    if (sequence === Atomics.load(this.header, Slot.Read)) return false

    out.set(this.slots[sequence & 1]!.subarray(0, Math.min(out.length, this.frameLength)))
    Atomics.store(this.header, Slot.Read, sequence)
    return true
  }

  /**
   * Copies the newest frame without advancing the read cursor, so a second reader (the
   * scanner reading the spectrum the render loop is already consuming) can look at it too.
   * Returns false only before the first frame is ever published.
   *
   * Safe against a concurrent `publish`: the writer always targets the other of the two
   * slots, so the slot read here is not the one being written.
   */
  peek(out: Float32Array): boolean {
    const sequence = Atomics.load(this.header, Slot.Write)
    if (sequence === 0) return false
    out.set(this.slots[sequence & 1]!.subarray(0, Math.min(out.length, this.frameLength)))
    return true
  }

  get buffer(): SharedArrayBuffer {
    return this.sab
  }
}
