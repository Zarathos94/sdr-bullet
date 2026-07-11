/// <reference lib="webworker" />
/**
 * Transforms baseband into spectrum rows for the display.
 *
 * Deliberately decoupled from the rest of the pipeline: it reads from a single overwritten
 * slot rather than a queue, so falling behind costs dropped frames and nothing else. A
 * display that stutters is a nuisance; a display that applies back pressure to the capture
 * loop would break the audio, which is not.
 *
 * It also paces itself to the display rather than to the capture rate. At 2.4 million
 * samples a second the capture stage offers far more frames than any screen can show, and
 * transforming them all would spend a core to produce rows nobody sees.
 */

import { SpectrumStage } from '../wasm/sdr_wasm.js'
import { floatView, loadWasm } from '../wasm/load.js'
import { LatestFrame } from '../ipc/ring.js'

export interface SpectrumConfig {
  /** Slot the capture worker publishes interleaved baseband into. */
  source: SharedArrayBuffer
  /** Slot this worker publishes finished rows into. */
  sink: SharedArrayBuffer
  fftSize: number
  /** Target rows per second. Beyond the display's refresh rate this is wasted work. */
  rowsPerSecond: number
}

export type ToSpectrum =
  | { type: 'start'; config: SpectrumConfig }
  | { type: 'smoothing'; alpha: number }
  | { type: 'stop' }

let stage: SpectrumStage | undefined
let running = false

async function start(config: SpectrumConfig) {
  const { memory } = await loadWasm()

  stage = new SpectrumStage(config.fftSize)
  const re = floatView(memory, stage.i_ptr(), config.fftSize)
  const im = floatView(memory, stage.q_ptr(), config.fftSize)
  const bins = floatView(memory, stage.bins_ptr(), config.fftSize)

  const source = new LatestFrame(config.source, config.fftSize * 2)
  const sink = new LatestFrame(config.sink, config.fftSize)

  const interleaved = new Float32Array(config.fftSize * 2)
  const interval = 1000 / config.rowsPerSecond
  running = true

  while (running) {
    if (source.consume(interleaved)) {
      for (let k = 0; k < config.fftSize; k++) {
        re[k] = interleaved[k * 2]!
        im[k] = interleaved[k * 2 + 1]!
      }
      stage.process()
      sink.publish(bins)
    }

    // A plain sleep, since there is nothing to block on — the source slot has no notify,
    // by design, because waiting on it would be waiting for work that can be skipped.
    await new Promise((resolve) => setTimeout(resolve, interval))
  }
}

self.onmessage = async (event: MessageEvent<ToSpectrum>) => {
  const message = event.data
  switch (message.type) {
    case 'start':
      await start(message.config)
      break
    case 'smoothing':
      stage?.set_smoothing(message.alpha)
      break
    case 'stop':
      running = false
      break
  }
}
