/**
 * The instantaneous spectrum trace: a line, optionally filled underneath.
 *
 * The geometry is generated on the GPU from a storage buffer of dB values, indexed by
 * `vertex_index`, so nothing is built on the CPU per frame — a new spectrum is one buffer
 * write, and the draw expands it into vertices. Two pipelines share that buffer: a triangle
 * strip that fills the area under the curve, and a line strip for the trace itself. The fill
 * is a strip of alternating top/bottom vertices, which is why the vertex count is twice the
 * bin count for the fill and equal to it for the line.
 */

import { type PaletteKey, readPalette } from './theme.js'

/** Upper bound on bins, fixing the storage-buffer size. Comfortably above any FFT this app runs. */
const MAX_BINS = 1 << 14

/** Opacity of the filled area, kept low so the trace and grid read through it. */
const AREA_ALPHA = 0.28

const SHADER_WGSL = /* wgsl */ `
struct Params {
  minDb: f32, maxDb: f32, binCount: u32, pad0: u32,
  trace: vec4<f32>,
  area: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> bins: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;

// dB to clip-space y, with a small margin top and bottom so peaks are not clipped at the edge.
fn mapY(db: f32) -> f32 {
  let span = max(p.maxDb - p.minDb, 1.0e-6);
  let n = clamp((db - p.minDb) / span, 0.0, 1.0);
  return n * 1.8 - 0.9;
}

fn xForBin(bin: u32) -> f32 {
  let denom = f32(max(p.binCount - 1u, 1u));
  return f32(bin) / denom * 2.0 - 1.0;
}

// Fill: two vertices per bin. Even index is the point on the curve, odd is the floor below it,
// so consecutive quads tile the area under the trace.
@vertex
fn vsFill(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  let bin = vid / 2u;
  let onCurve = (vid & 1u) == 0u;
  let y = select(-1.0, mapY(bins[bin]), onCurve);
  return vec4<f32>(xForBin(bin), y, 0.0, 1.0);
}

@fragment
fn fsFill() -> @location(0) vec4<f32> {
  return p.area;
}

// Line: one vertex per bin, drawn as a line strip.
@vertex
fn vsLine(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(xForBin(vid), mapY(bins[vid]), 0.0, 1.0);
}

@fragment
fn fsLine() -> @location(0) vec4<f32> {
  return p.trace;
}
`

/** Standard straight-alpha blending, so the fill sits over the background and the line over the fill. */
const BLEND: GPUBlendState = {
  color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
}

export class SpectrumRenderer {
  private readonly context: GPUCanvasContext
  private readonly format: GPUTextureFormat

  private readonly fillPipeline: GPURenderPipeline
  private readonly linePipeline: GPURenderPipeline

  private readonly dataBuf: GPUBuffer
  private readonly paramsBuf: GPUBuffer
  private readonly paramsData = new ArrayBuffer(48)
  private readonly paramsView = new DataView(this.paramsData)

  private readonly fillBind: GPUBindGroup
  private readonly lineBind: GPUBindGroup

  private binCount = 0
  private minDb = -100
  private maxDb = -20
  private filled = true

  private trace: [number, number, number]
  private area: [number, number, number]
  private background: [number, number, number]

  constructor(
    private readonly device: GPUDevice,
    private readonly canvas: HTMLCanvasElement,
  ) {
    const context = canvas.getContext('webgpu')
    if (!context) throw new Error('SpectrumRenderer: could not get a webgpu canvas context')
    this.context = context
    this.format = navigator.gpu.getPreferredCanvasFormat()
    context.configure({ device, format: this.format, alphaMode: 'opaque' })

    // Seed colours from the page theme so the trace matches the surrounding UI out of the box.
    const palette = readPalette()
    const pick = (key: PaletteKey): [number, number, number] => palette[key]
    this.trace = pick('trace')
    this.area = pick('accent')
    this.background = pick('background')

    const module = device.createShaderModule({ code: SHADER_WGSL })
    this.fillPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vsFill' },
      fragment: { module, entryPoint: 'fsFill', targets: [{ format: this.format, blend: BLEND }] },
      primitive: { topology: 'triangle-strip' },
    })
    this.linePipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vsLine' },
      fragment: { module, entryPoint: 'fsLine', targets: [{ format: this.format, blend: BLEND }] },
      primitive: { topology: 'line-strip' },
    })

