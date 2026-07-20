# Architecture

## The shape of the problem

A software radio is a pipeline. Bytes arrive from the dongle at 2.4 million samples a
second; each stage does a little less work on a little less data until what comes out is
48 kHz of audio and a few spectrum rows a second. The stages are independent — each one
consumes what the previous produced and nothing else — which is what makes the pipeline the
natural unit of parallelism.

```
USB ─▶ capture ─▶ IQ ring ─▶ demodulate ─▶ audio ring ─▶ worklet ─▶ speakers
                     │
                     └─────▶ spectrum ─▶ latest-frame ─▶ WebGPU
```

## Why workers, not threads

The obvious way to parallelise WebAssembly is shared linear memory across threads. It is
not available here. Shared memory needs the standard library compiled with the `atomics`
and `bulk-memory` target features, which needs `-Z build-std`, which needs a nightly
toolchain. Forcing it on stable produces one of two failures: either a module whose memory
cannot back a `SharedArrayBuffer` at all, or — with `--no-check-features` — a shared memory
over a standard library that still assumes it is single-threaded, which corrupts silently.

So each stage runs in its own worker with its own WebAssembly instance and its own
non-shared memory. The stages communicate through `SharedArrayBuffer` ring buffers managed
from JavaScript. This is pipeline parallelism rather than data parallelism, and for a
streaming signal chain that is the right decomposition regardless of the toolchain
constraint — the stages were always going to be a chain.

The cost is a copy at each ring boundary. At these block sizes that is memory bandwidth and
is dwarfed by the signal processing on either side of it.

## The two kinds of ring

Not every consumer wants the same delivery guarantee.

**Audio must not drop a sample.** A gap in the audio ring is an audible click. So the IQ and
audio paths use a strict single-producer, single-consumer queue: the producer writes only
what fits and drops whole blocks when the consumer falls behind, keeping the delivered
samples contiguous.

**The display should drop stale frames.** A spectrum frame that arrives late is worth
nothing — the next one is already better. Queueing frames the display will never draw only
adds latency, and back-pressure from a slow display must never reach the capture stage. So
the spectrum path uses a single latest-wins slot: the producer overwrites, the consumer
takes whatever is newest.

The correctness of both rests on one rule, which is the whole reason to hand-write them
rather than pass messages: the payload is written with ordinary stores and the index is
published afterwards with an atomic store. A reader that sees the new index is guaranteed to
see the payload, because JavaScript atomics are sequentially consistent and an ordinary
write cannot be reordered past one.

## Where blocking is allowed

`Atomics.wait` blocks a thread until an index changes, which is exactly what a demodulation
worker wants — it should sleep until samples arrive, not spin on a timer. It is permitted in
a dedicated worker, whose agent is specified with `[[CanBlock]] = true`.

It is forbidden on the audio render thread. A worklet's agent is specified with
`[[CanBlock]] = false`, so `Atomics.wait` there throws a `TypeError` rather than waiting.
That is correct: the render thread has a deadline and must never block. So the worklet
polls its ring, takes what is there, and outputs silence for whatever is not — turning an
underrun into a counter on the diagnostics panel instead of a stall that takes the graph
down.

## Memory views and the growth hazard

The stages expose pointers into WebAssembly memory so JavaScript can read and write their
buffers without copying. Those views are valid only while the heap does not grow — growing
it detaches every view over `memory.buffer`, silently turning them into zero-length arrays
with no error raised.

The stages therefore allocate every buffer in their constructors, from sizes fixed then,
and nothing on the hot path allocates. The workers build their views after constructing the
stages, and never rebuild them. A `SharedArrayBuffer`-backed view never detaches, which is
a further reason the ring buffers live in shared memory rather than instance memory.

## The driver split

The RTL2832U and R828D registers are driven by pure Rust that knows nothing about USB. It
is generic over a `Transport` trait with three methods, and there are three implementations:
a native one over libusb, a browser one over WebUSB, and a recording mock for the tests.

This is not tidiness for its own sake. The hard part of a USB driver is the register
arithmetic — the synthesiser dividers, the sample-rate ratio, the intermediate-frequency
offset — and those are exactly the parts you cannot debug from a bus trace. Keeping them
transport-independent means they run first against real hardware from a terminal, where a
failure is a stack trace, before they ever run behind three layers of browser sandbox where
a failure is a silent stream of zeroes.

Two behaviours in the driver were found only by driving the real device, and both contradict
the published reference implementations: this hardware's tuner reads are not bit-reversed,
and its I2C writes must be split into eight-byte messages. See
[register-protocol.md](register-protocol.md).
