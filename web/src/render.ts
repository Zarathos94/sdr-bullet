/**
 * Drives the GPU displays from the pipeline, on the main thread's animation frame.
 *
 * The renderers live here rather than in a worker because `OffscreenCanvas` plus WebGPU in
 * a worker is still uneven across browsers, and the display is the one part of the pipeline
 * that can afford to run on the main thread: it only ever draws the newest frame, so a
 * dropped animation frame costs nothing but a skipped redraw. The audio path, which cannot
 * afford that, is the part kept off the main thread.
 */

import { acquireDevice, adapterOf, describeAdapter } from './gpu/device.js'
import { WaterfallRenderer } from './gpu/waterfall.js'
import { SpectrumRenderer } from './gpu/spectrum.js'
import { ConstellationRenderer } from './gpu/constellation.js'
import type { Pipeline } from './pipeline.js'

export interface RenderTargets {
  spectrum: HTMLCanvasElement
  waterfall: HTMLCanvasElement
  constellation: HTMLCanvasElement
}

export interface RenderHandle {
  stop: () => void
  usingGpu: boolean
  adapterInfo: string
}

/**
 * Starts the render loop, returning a handle that stops it.
 *
 * Returns `usingGpu: false` when WebGPU is unavailable — no shipping browser exposes a
 * software fallback adapter, so on that path the displays stay dark and the rest of the
 * app, audio included, carries on. The receiver working without a picture is a far better
 * failure than the whole app refusing to start over a missing GPU.
 */
export async function startRendering(
  pipeline: Pipeline,
  targets: RenderTargets,
): Promise<RenderHandle> {
  const device = await acquireDevice()

  if (!device) {
    return {
      stop: () => {},
      usingGpu: false,
      adapterInfo: 'WebGPU unavailable — displays disabled, audio unaffected',
    }
  }

  const waterfall = new WaterfallRenderer(device, targets.waterfall)
  const spectrum = new SpectrumRenderer(device, targets.spectrum)
  const constellation = new ConstellationRenderer(device, targets.constellation)

  // GPUDevice does not reference its adapter, so the identity comes from the map the
  // device module keeps. Some drivers withhold every field, hence the fallback string.
  const adapter = adapterOf(device)
  const info = adapter ? describeAdapter(adapter) : undefined
  const adapterInfo =
    info && (info.vendor || info.architecture)
      ? `${info.vendor} ${info.architecture}`.trim()
      : 'WebGPU'

  const resize = () => {
    for (const canvas of [targets.spectrum, targets.waterfall, targets.constellation]) {
      const rect = canvas.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.max(1, Math.round(rect.width * dpr))
      canvas.height = Math.max(1, Math.round(rect.height * dpr))
    }
    spectrum.resize(targets.spectrum.width, targets.spectrum.height)
    waterfall.resize(targets.waterfall.width, targets.waterfall.height)
    constellation.resize(targets.constellation.width, targets.constellation.height)
  }
  resize()
  const observer = new ResizeObserver(resize)
  observer.observe(targets.spectrum)
  observer.observe(targets.waterfall)

  // Deinterleave scratch for the constellation, sized to the frame the pipeline hands out.
  // Reused every frame rather than allocated in the loop.
  let iScratch = new Float32Array(0)
  let qScratch = new Float32Array(0)

  let raf = 0
  let stopped = false

  const frame = () => {
    if (stopped) return

    // A new spectrum row feeds both the trace and the scrolling waterfall. There may be no
    // new row on a given frame — the display runs faster than the pipeline offers rows —
    // in which case the previous image simply persists.
    const bins = pipeline.latestSpectrum()
    if (bins) {
      spectrum.pushRow(bins)
      waterfall.pushRow(bins)
    }

    // The constellation wants separate I and Q arrays; the pipeline hands over interleaved
    // baseband, so split it into the reused scratch buffers.
    const iq = pipeline.latestConstellation()
    if (iq) {
      const samples = iq.length >> 1
      if (iScratch.length !== samples) {
        iScratch = new Float32Array(samples)
        qScratch = new Float32Array(samples)
      }
      for (let k = 0; k < samples; k++) {
        iScratch[k] = iq[k * 2]!
        qScratch[k] = iq[k * 2 + 1]!
      }
      constellation.pushSamples(iScratch, qScratch)
    }

    spectrum.render()
    waterfall.render()
    constellation.render()

    raf = requestAnimationFrame(frame)
  }
  raf = requestAnimationFrame(frame)

  return {
    stop: () => {
      stopped = true
      cancelAnimationFrame(raf)
      observer.disconnect()
      waterfall.dispose()
      spectrum.dispose()
      constellation.dispose()
    },
    usingGpu: true,
    adapterInfo,
  }
}
