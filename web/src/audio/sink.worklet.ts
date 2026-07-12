/**
 * Audio output. Pulls from a ring buffer and never blocks.
 *
 * This runs on the audio render thread, which is the one place in the pipeline where
 * waiting is not merely unwise but forbidden: worklet agents are specified with
 * `[[CanBlock]] = false`, so `Atomics.wait` throws a `TypeError` rather than waiting.
 * `Atomics.waitAsync` is technically permitted but no better an idea — the render thread
 * must return within its deadline whatever the rest of the pipeline is doing.
 *
 * So the contract is: take what is there, output silence for whatever is not, and count
 * the shortfall. An underrun becomes a number on a diagnostics panel instead of a stall
 * that takes the whole graph down.
 *
 * Three details here are load-bearing and each has cost someone an afternoon:
 *
 * - `process` must return `true`. The return value feeds the node's active-source flag,
 *   which is what keeps a node with no inputs alive. Returning `false` — or falling off
 *   the end, which yields `undefined` — makes the node collectable, and the audio stops
 *   for no visible reason.
 * - Nothing is allocated per call. Every buffer is created in the constructor.
 * - The whole body is wrapped. A throw inside `process` permanently disables the
 *   processor: the specification sets its callable flag false, fires `processorerror`, and
 *   outputs silence for the node's remaining lifetime.
 */

/** Header layout, matching `ipc/ring.ts`. */
const enum Slot {
  Write = 0,
  Read = 1,
  Capacity = 2,
}
const HEADER_SLOTS = 32
const HEADER_BYTES = HEADER_SLOTS * 4

/** Frames in one render quantum. Fixed at 128 in every shipping browser. */
const QUANTUM = 128

/**
 * The processor options this node is constructed with.
 *
 * Declared locally because the worklet-scope type definitions do not carry
 * `AudioWorkletNodeOptions` — its `processorOptions` field is untyped there anyway, so
 * naming exactly the shape passed from the page is both what we need and more precise.
 */
interface SinkConstructionOptions {
  processorOptions: { ring: SharedArrayBuffer }
}

class SdrSink extends AudioWorkletProcessor implements AudioWorkletProcessorImpl {
  private readonly header: Int32Array
  private readonly data: Float32Array
  private readonly capacity: number
  private readonly mask: number
  private readonly scratch: Float32Array

  private underruns = 0
  private framesDelivered = 0
  private reportCountdown = 0
  private stopped = false

  constructor(options: SinkConstructionOptions) {
    super()

    // `processorOptions` is readable synchronously in the constructor, unlike anything
    // arriving by `postMessage`. It is also the only route that works everywhere: the
    // worklet-scope `port` is implemented in Firefox alone.
    const { ring } = options.processorOptions
    this.header = new Int32Array(ring, 0, HEADER_SLOTS)
    this.data = new Float32Array(ring, HEADER_BYTES)
    this.capacity = Atomics.load(this.header, Slot.Capacity)
    this.mask = this.capacity - 1
    // Interleaved stereo, so two samples per frame.
    this.scratch = new Float32Array(QUANTUM * 2)

    this.port.onmessage = (event) => {
      if (event.data === 'stop') this.stopped = true
    }
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    try {
      const output = outputs[0]
      const left = output?.[0]
      const right = output?.[1] ?? output?.[0]
      if (!left || !right) return !this.stopped

      const frames = left.length
      const wanted = frames * 2

      const read = Atomics.load(this.header, Slot.Read)
      // Acquire: pairs with the producer publishing its index after writing the payload.
      const write = Atomics.load(this.header, Slot.Write)
      // Unsigned difference, so the indices may wrap through 32 bits without going
      // negative — which they do after about fifteen minutes at this rate.
      const ready = Math.min((write - read) >>> 0, wanted)

      if (ready > 0) {
        const start = read & this.mask
        const firstRun = Math.min(ready, this.capacity - start)
        this.scratch.set(this.data.subarray(start, start + firstRun), 0)
        if (firstRun < ready) {
          this.scratch.set(this.data.subarray(0, ready - firstRun), firstRun)
        }
        Atomics.store(this.header, Slot.Read, (read + ready) >>> 0)
      }

      const framesReady = ready >> 1
      for (let k = 0; k < framesReady; k++) {
        left[k] = this.scratch[k * 2]!
        right[k] = this.scratch[k * 2 + 1]!
      }
      // Silence rather than stale content for anything that did not arrive. Repeating the
      // previous quantum would be louder but would also mask the fault.
      for (let k = framesReady; k < frames; k++) {
        left[k] = 0
        right[k] = 0
      }

      if (framesReady < frames) this.underruns++
      this.framesDelivered += framesReady

      // Report occasionally rather than every quantum; at 128 frames that would be
      // hundreds of messages a second for a number nobody reads that fast.
      if (--this.reportCountdown <= 0) {
        this.reportCountdown = 200
        this.port.postMessage({
          underruns: this.underruns,
          framesDelivered: this.framesDelivered,
          fill: ((write - read) >>> 0) / this.capacity,
        })
      }

      // Keeping the active-source flag set is what stops this node being collected.
      return !this.stopped
    } catch {
      // A throw here would disable the processor for good, so swallow it and emit silence.
      // The underrun counter is the signal that something is wrong.
      return !this.stopped
    }
  }
}

registerProcessor('sdr-sink', SdrSink)
