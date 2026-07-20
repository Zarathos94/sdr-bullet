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

/**
 * Control operations (retune, gain, bias-tee, …) waiting to run against the device.
 *
 * They cannot run the moment their message arrives: the sample stream keeps several bulk
 * transfers in flight, and a control write that lands among them — especially the buffer
 * reset a retune performs, which flushes the endpoint FIFO — cancels those transfers. The
 * read loop then sees the cancellation, treats it as fatal, and closes the device, which in
 * turn cancels the control write itself ("controlTransferOut … The transfer was cancelled").
 * So the loop drains this queue between reads with the stream stopped instead, serialising
 * every access to the device onto one thread of control.
 */
const controlQueue: Array<() => Promise<void>> = []

function post(message: FromWorker) {
  self.postMessage(message)
}

async function start(config: CaptureConfig) {
  const { memory } = await loadWasm()
  post({ type: 'ready', simdBackend: simd_backend() })

  transport = await UsbTransport.open()
  const isV4 = config.isV4 || transport.identity.isV4
  // V4 detection sets the tuner reference clock; getting it wrong is nothing but static, so
  // make the resolved identity visible for diagnosis.
  console.info(
    `[sdr] device "${transport.identity.manufacturer ?? '?'}" / "${transport.identity.product ?? '?'}"` +
      ` — V4=${isV4} (28.8 MHz reference ${isV4 ? 'on' : 'OFF — will mistune unless this is a generic R828D'})`,
  )
  receiver = await Receiver.open(transport, isV4)

  const actualRate = await receiver.setSampleRate(config.sampleRate)
  await receiver.setGain(config.gainTenths)
  await receiver.setFrequency(config.centreHz)
  await receiver.resetBuffer()

  const stage = new CaptureStage(MAX_TRANSFER_BYTES)

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
    // Drain queued control operations with the stream stopped, so their control writes
    // never overlap the in-flight sample transfers. This is the one place the device is
    // touched besides the read itself, which keeps every access serialised.
    if (controlQueue.length > 0) {
      await transport.stopStream()
      while (controlQueue.length > 0) {
        const op = controlQueue.shift()!
        try {
          await op()
        } catch (error) {
          post({
            type: 'error',
            message: `control: ${error instanceof Error ? error.message : String(error)}`,
            fatal: false,
          })
        }
      }
      if (!running) break
      transport.startStream()
      continue
    }

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

    // Re-view the wasm input buffer each pass: a cached typed array over the wasm memory
    // detaches the moment that memory grows, and reads/writes through a detached view are
    // silently dropped. Re-creating it over the current buffer is cheap and immune.
    const rawInput = byteView(memory, stage.input_ptr(), stage.input_capacity())
    const usable = Math.min(block.length, rawInput.length)
    rawInput.set(block.subarray(0, usable))
    const samples = stage.process(usable)
    if (samples === 0) continue

    const i = floatView(memory, stage.i_ptr(), samples)
    const q = floatView(memory, stage.q_ptr(), samples)

    iq.write(i, q)

    // The display only ever draws the newest frame, so a single overwritten slot is both
    // cheaper than a queue and stops a slow display applying back pressure to capture. The
    // spectrum worker and the main-thread constellation each read their own slot.
    if (samples >= spectrumSamples) {
      for (let k = 0; k < spectrumSamples; k++) {
        spectrumFrame[k * 2] = i[k]!
        spectrumFrame[k * 2 + 1] = q[k]!
      }
      spectrum.publish(spectrumFrame)
      constellation.publish(spectrumFrame)
    }

    bytesSinceReport += usable
    const now = performance.now()
    if (now - lastReport > 500) {
      const elapsed = (now - lastReport) / 1000
      post({
        type: 'status',
        tunedHz: receiver.tunedHz,
        sampleRate: actualRate,
        band: receiver.band,
        tunerFrequencyHz: receiver.tunerFrequencyHz,
        locked: await receiver.isLocked(),
        dropped: iq.dropped(),
        bytesPerSecond: bytesSinceReport / elapsed,
      })
      bytesSinceReport = 0
      lastReport = now
    }
  }

  iq.close()
  await teardown()
}

async function teardown() {
  running = false
  controlQueue.length = 0
  if (transport) {
    await transport.close().catch(() => {})
    transport = undefined
  }
  receiver = undefined
}

self.onmessage = async (event: MessageEvent<ToCapture>) => {
  const message = event.data
  try {
    switch (message.type) {
      case 'start':
        await start(message.config)
        break
      case 'tune':
        controlQueue.push(async () => {
          if (!receiver) return
          await receiver.setFrequency(message.hz)
          // Whatever is already in flight was captured at the previous frequency.
          await receiver.resetBuffer()
        })
        break
      case 'gain':
        controlQueue.push(async () => {
          await receiver?.setGain(message.tenths)
        })
        break
      case 'biasTee':
        controlQueue.push(async () => {
          await receiver?.setBiasTee(message.enabled)
        })
        break
      case 'correction':
        controlQueue.push(async () => {
          await receiver?.setFrequencyCorrection(message.ppm)
        })
        break
      case 'stop':
        await teardown()
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
