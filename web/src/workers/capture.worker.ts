/// <reference lib="webworker" />
/**
 * Owns the USB endpoint and turns transfers into corrected baseband.
 *
 * Runs in a worker for two reasons. Keeping the transfer queue off the main thread means a
 * slow frame cannot stall it — and with no transfer outstanding, WebUSB drops whatever the
 * device sends rather than buffering it. And unpacking bytes into floats is per-sample work
 * at the full capture rate, which is exactly what should not share a thread with layout.
 *
 * `requestDevice()` is exposed only to `Window`, so permission is granted from the page and
 * this worker reacquires the device through `getDevices()`. A `USBDevice` cannot be
 * structured-cloned, so it genuinely has to be picked up again rather than passed across.
 */

import { CaptureStage, Receiver, simd_backend } from '../wasm/sdr_wasm.js'
import { byteView, floatView, loadWasm } from '../wasm/load.js'
import { IqProducer } from '../ipc/iq-ring.js'
import { LatestFrame } from '../ipc/ring.js'
import { UsbTransport } from '../usb/webusb.js'
import type { CaptureConfig, FromWorker, ToCapture } from './protocol.js'

/** Bounds one transfer, and fixes every buffer size in the stage for good. */
const MAX_TRANSFER_BYTES = 256 * 1024

let transport: UsbTransport | undefined
let receiver: Receiver | undefined
let running = false

function post(message: FromWorker) {
  self.postMessage(message)
}

async function start(config: CaptureConfig) {
  const { memory } = await loadWasm()
  post({ type: 'ready', simdBackend: simd_backend() })

  transport = await UsbTransport.open()
  receiver = await Receiver.open(transport, config.isV4 || transport.identity.isV4)

  const actualRate = await receiver.setSampleRate(config.sampleRate)
  await receiver.setGain(config.gainTenths)
  await receiver.setFrequency(config.centreHz)
  await receiver.resetBuffer()

  // Constructed before any view is built, because its allocations would detach them.
  const stage = new CaptureStage(MAX_TRANSFER_BYTES)
  const rawInput = byteView(memory, stage.input_ptr(), stage.input_capacity())

  const iq = new IqProducer(config.iq)
  const spectrum = new LatestFrame(config.spectrum, config.spectrumFrameLength)
  const constellation = new LatestFrame(config.constellation, config.spectrumFrameLength)
  const spectrumFrame = new Float32Array(config.spectrumFrameLength)
  const spectrumSamples = config.spectrumFrameLength / 2

  transport.startStream()
  running = true

  let bytesSinceReport = 0
  let lastReport = performance.now()

  while (running) {
    let block: Uint8Array
    try {
      block = await transport.readSamples()
    } catch (error) {
      if (!running) break
      const message = error instanceof Error ? error.message : String(error)
      // A stall is recoverable — the endpoint is cleared and the next transfer proceeds.
      // A disconnect is not, and retrying through it just spins.
      const fatal = !message.includes('stalled')
      post({ type: 'error', message: `capture: ${message}`, fatal })
      if (fatal) break
      continue
    }

    const usable = Math.min(block.length, rawInput.length)
    rawInput.set(block.subarray(0, usable))
    const samples = stage.process(usable)
    if (samples === 0) continue

    const i = floatView(memory, stage.i_ptr(), samples)
    const q = floatView(memory, stage.q_ptr(), samples)

