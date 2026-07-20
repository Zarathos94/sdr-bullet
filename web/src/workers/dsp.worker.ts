/// <reference lib="webworker" />
/**
 * Channel selection and demodulation.
 *
 * Blocks on the incoming ring rather than polling. `Atomics.wait` is permitted here — a
 * dedicated worker is specified with `[[CanBlock]] = true`, unlike a window or a worklet —
 * and blocking beats a timer because it wakes on the producer's notify rather than on the
 * next tick, which is the difference between microseconds and milliseconds of added
 * latency.
 */

import { ChannelStage } from '../wasm/sdr_wasm.js'
import { floatView, loadWasm } from '../wasm/load.js'
import { IqConsumer } from '../ipc/iq-ring.js'
import { RingProducer } from '../ipc/ring.js'
import { MODE_VALUES, type DspConfig, type FromWorker, type ToDsp } from './protocol.js'

let stage: ChannelStage | undefined
let running = false

function post(message: FromWorker) {
  self.postMessage(message)
}

async function start(config: DspConfig) {
  const { memory } = await loadWasm()

  stage = new ChannelStage(config.sampleRate, config.blockSize, MODE_VALUES[config.mode])

  const iq = new IqConsumer(config.iq)
  const audio = new RingProducer(config.audio)

  // Interleaved stereo, because that is what the audio callback wants and interleaving
  // 128 frames there would be work on the render thread.
  const interleaved = new Float32Array(config.blockSize * 2)

  running = true
  let lastReport = performance.now()

  while (running) {
    if (!iq.waitFor(config.blockSize, 100)) continue

    // Re-view the wasm input buffers every pass. A typed array over the wasm memory detaches
    // the instant that memory grows — and a lazy allocation inside the demod chain grows it
    // on the first process() — after which a cached view has length 0, so the read below
    // silently takes nothing and the audio stops dead after one block. Re-creating the view
    // over the current buffer each iteration is cheap and immune to that.
    const inputI = floatView(memory, stage.i_ptr(), stage.input_capacity())
    const inputQ = floatView(memory, stage.q_ptr(), stage.input_capacity())
    const samples = iq.read(inputI, inputQ)
    if (samples === 0) continue

    const frames = stage.process(samples)
    if (frames > 0) {
      const left = floatView(memory, stage.audio_left_ptr(), frames)
      const right = floatView(memory, stage.audio_right_ptr(), frames)
      for (let k = 0; k < frames; k++) {
        interleaved[k * 2] = left[k]!
        interleaved[k * 2 + 1] = right[k]!
      }
      audio.write(interleaved.subarray(0, frames * 2))
    }

    const now = performance.now()
    if (now - lastReport > 500) {
      post({
        type: 'status',
        stereo: stage.stereo(),
        squelchOpen: stage.squelch_open(),
        pilotLevel: stage.pilot_level(),
        ringFill: iq.fill(),
        rds: {
          synchronised: stage.rds_synchronised(),
          stationName: stage.rds_station_name(),
          radioText: stage.rds_radio_text(),
          programId: stage.rds_program_id(),
          blockErrorRate: stage.rds_block_error_rate(),
        },
      })
      lastReport = now
    }
  }

  audio.close()
}

self.onmessage = async (event: MessageEvent<ToDsp>) => {
  const message = event.data
  try {
    switch (message.type) {
      case 'start':
        await start(message.config)
        break
      case 'mode':
        // Changing mode changes the channel rate and every filter with it, so the stage is
        // rebuilt rather than reconfigured. Doing that here would leave the views above
        // dangling, so the pipeline restarts the worker instead.
        post({ type: 'error', message: 'mode changes require a pipeline restart', fatal: false })
        break
      case 'offset':
        stage?.set_channel_offset(message.hz)
        break
      case 'squelch':
        stage?.set_squelch(message.enabled, message.threshold)
        break
      case 'deemphasis':
        stage?.set_deemphasis_us(message.microseconds)
        break
      case 'mono':
        stage?.set_forced_mono(message.forced)
        break
      case 'agc':
        stage?.set_agc_enabled(message.enabled)
        break
      case 'stop':
        running = false
        break
    }
  } catch (error) {
    post({
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
      fatal: true,
    })
  }
}
