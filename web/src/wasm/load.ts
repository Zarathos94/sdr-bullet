/**
 * Loads the WebAssembly module and hands back its memory.
 *
 * Each worker instantiates the module separately, with its own linear memory. That is the
 * architecture rather than an oversight: shared-memory threading on the Rust side needs an
 * atomics-enabled standard library, which needs a nightly toolchain and `build-std`, and
 * produces a module whose failures are invisible. Independent instances joined by ring
 * buffers give real parallelism on a stable toolchain, and a signal chain splits naturally
 * by stage anyway.
 *
 * # Views into WebAssembly memory
 *
 * The stages expose pointers so JavaScript can read and write their buffers without
 * copying. Those views are only valid while the heap does not grow — **growing it detaches
 * every existing view**, silently turning them into zero-length arrays rather than raising
 * anything. So the order matters: construct every stage first, then build the views. The
 * stages allocate all their buffers in their constructors for exactly this reason, and
 * nothing on the hot path allocates.
 */

import init from './sdr_wasm.js'
// The ?url import makes Vite emit the binary as a fingerprinted asset and hand back its
// final URL. Passing that URL to init() explicitly means the loader never depends on the
// glue resolving its own path through import.meta.url — which Vite 8 leaves undefined in a
// production worker, a failure that appears only in the built artefact and never in dev.
import wasmUrl from './sdr_wasm_bg.wasm?url'

export interface WasmModule {
  memory: WebAssembly.Memory
}

let loaded: Promise<WasmModule> | undefined

/** Loads the module once per worker, returning the same promise on repeat calls. */
export function loadWasm(): Promise<WasmModule> {
  const initialise = init as unknown as (options: {
    module_or_path: string
  }) => Promise<{ memory: WebAssembly.Memory }>
  loaded ??= initialise({ module_or_path: wasmUrl }).then((exports) => ({
    memory: exports.memory,
  }))
  return loaded
}

/**
 * A float view over WebAssembly memory.
 *
 * Wrapped rather than used inline so every call site is a reminder that the result has a
 * lifetime tied to the heap not growing.
 */
export function floatView(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
): Float32Array {
  return new Float32Array(memory.buffer, pointer, length)
}

export function byteView(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
): Uint8Array {
  return new Uint8Array(memory.buffer, pointer, length)
}
