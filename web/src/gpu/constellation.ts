/**
 * The I/Q constellation, drawn as a density plot.
 *
 * A constellation is a scatter of hundreds of thousands of points per second, and the reading
 * that matters is where they pile up — the density, not the individual dots. So points are
 * accumulated into a grid of counters and the grid is coloured, rather than drawing sprites.
 *
 * The counters have to live in a storage buffer of `atomic<u32>`, not in a texture. WGSL has
 * no storage-texture atomics — gpuweb issue #4329 is still open, blocked on Metal — so a
 * scatter that writes the same cell from many invocations at once cannot target a texture and
 * stay correct. The path is therefore: scatter with `atomicAdd` into the buffer; a second
 * compute pass reads the buffer, applies a log scale and a colormap, and `textureStore`s the
 * result into an ordinary storage texture; the render pass samples that texture. The texture
 * is only ever written by one invocation per texel, which is exactly the case textures allow.
 *
 * Persistence is a geometric decay of the counters each frame, so a cleared region fades as a
 * trail instead of the plot being a permanent union of everything ever seen.
 */

import { colormapTexels, type ColormapName } from './colormap.js'

/** Density grid resolution. 512 x 512 counters is 1 MiB and ample for a scope. */
const GRID = 512
const CELLS = GRID * GRID

/** Cap on samples uploaded per scatter. Bounds the sample buffers; longer inputs are chunked. */
const MAX_SAMPLES = 1 << 16

/** 2D colour-pass tiling: 16 x 16 = 256, the guaranteed workgroup ceiling. */
const TILE = 16

const SPLAT_WGSL = /* wgsl */ `
struct SplatParams { count: u32, gridW: u32, gridH: u32, scale: f32 };

// Density counters. atomic because many samples land in the same cell in the same dispatch;
// a texture could not be the target here, per issue #4329.
@group(0) @binding(0) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> si: array<f32>;
@group(0) @binding(2) var<storage, read> sq: array<f32>;
@group(0) @binding(3) var<uniform> sp: SplatParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= sp.count) { return; }
  // Unit amplitude maps to the edge of the plot; scale zooms in on the origin cluster.
  let x = si[idx] * sp.scale * 0.5 + 0.5;
  let y = sq[idx] * sp.scale * 0.5 + 0.5;
  if (x < 0.0 || x >= 1.0 || y < 0.0 || y >= 1.0) { return; }
  let cx = u32(x * f32(sp.gridW));
  let cy = u32(y * f32(sp.gridH));
  atomicAdd(&accum[cy * sp.gridW + cx], 1u);
}
`

const DECAY_WGSL = /* wgsl */ `
struct DecayParams { alpha: f32, cells: u32, pad0: u32, pad1: u32 };

@group(0) @binding(0) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> dp: DecayParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= dp.cells) { return; }
  // One cell per invocation, so the load then store races nothing and needs no read-modify-write
  // atomic. Multiplying by alpha < 1 turns standing density into a fading trail.
  let v = atomicLoad(&accum[idx]);
  atomicStore(&accum[idx], u32(f32(v) * dp.alpha));
}
`

const COLOUR_WGSL = /* wgsl */ `
struct ColourParams { gridW: u32, gridH: u32, logCap: f32, pad0: u32 };

@group(0) @binding(0) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(1) var lut: texture_2d<f32>;
@group(0) @binding(2) var lutSamp: sampler;
@group(0) @binding(3) var<uniform> cp: ColourParams;
@group(0) @binding(4) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= cp.gridW || gid.y >= cp.gridH) { return; }
  let count = atomicLoad(&accum[gid.y * cp.gridW + gid.x]);
  // The origin cluster is orders of magnitude denser than the arms, so a linear map would show
  // a white blob and nothing else. A log scale compresses that range so core and detail coexist.
  let t = clamp(log2(1.0 + f32(count)) / log2(1.0 + cp.logCap), 0.0, 1.0);
  let rgb = textureSampleLevel(lut, lutSamp, vec2<f32>(t, 0.5), 0.0).rgb;
  textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb, 1.0));
}
`

const DRAW_WGSL = /* wgsl */ `
@group(0) @binding(0) var densityTex: texture_2d<f32>;
@group(0) @binding(1) var densitySamp: sampler;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
  var corners = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
  );
  let xy = corners[vid];
  var o: VOut;
  o.pos = vec4<f32>(xy, 0.0, 1.0);
  // Map clip space straight to texture space so +Q is up and +I is right, the conventional
  // orientation: uv.y = 1 at the top of the screen fetches the high-Q rows.
  o.uv = vec2<f32>((xy.x + 1.0) * 0.5, (xy.y + 1.0) * 0.5);
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  return textureSample(densityTex, densitySamp, in.uv);
}
`

export class ConstellationRenderer {
  private readonly context: GPUCanvasContext
  private readonly format: GPUTextureFormat

  private readonly splatPipeline: GPUComputePipeline
  private readonly decayPipeline: GPUComputePipeline
  private readonly colourPipeline: GPUComputePipeline
  private readonly drawPipeline: GPURenderPipeline

  private readonly accumBuf: GPUBuffer
  private readonly iBuf: GPUBuffer
  private readonly qBuf: GPUBuffer
  private readonly densityTex: GPUTexture
  private readonly lutTex: GPUTexture
  private readonly lutSampler: GPUSampler
  private readonly densitySampler: GPUSampler

  private readonly splatBuf: GPUBuffer
  private readonly decayBuf: GPUBuffer
  private readonly colourBuf: GPUBuffer
  private readonly splatData = new ArrayBuffer(16)
  private readonly splatView = new DataView(this.splatData)
  private readonly decayData = new ArrayBuffer(16)
  private readonly decayView = new DataView(this.decayData)

  private readonly decayBind: GPUBindGroup
  private readonly colourBind: GPUBindGroup
  private readonly drawBind: GPUBindGroup
  // The scatter's bind group is rebuilt whenever the sample buffers change; here they are fixed,
  // so it is built once alongside the rest.
  private readonly splatBind: GPUBindGroup

  private alpha = 0.92
  private scale = 1
  private logCap = 64

  constructor(
    private readonly device: GPUDevice,
    private readonly canvas: HTMLCanvasElement,
  ) {
    const context = canvas.getContext('webgpu')
    if (!context) throw new Error('ConstellationRenderer: could not get a webgpu canvas context')
    this.context = context
    this.format = navigator.gpu.getPreferredCanvasFormat()
    // Opaque, cleared to black. inferno's zero is essentially black, so empty cells vanish into
    // the background the way a phosphor scope's do.
    context.configure({ device, format: this.format, alphaMode: 'opaque' })

    const splatModule = device.createShaderModule({ code: SPLAT_WGSL })
    const decayModule = device.createShaderModule({ code: DECAY_WGSL })
    const colourModule = device.createShaderModule({ code: COLOUR_WGSL })
    const drawModule = device.createShaderModule({ code: DRAW_WGSL })

    this.splatPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: splatModule, entryPoint: 'main' },
    })
    this.decayPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: decayModule, entryPoint: 'main' },
    })
    this.colourPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: colourModule, entryPoint: 'main' },
    })
    this.drawPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: drawModule, entryPoint: 'vs' },
      fragment: { module: drawModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    })

    this.accumBuf = device.createBuffer({
      size: CELLS * 4,
      usage: GPUBufferUsage.STORAGE,
    })
    this.iBuf = device.createBuffer({
      size: MAX_SAMPLES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.qBuf = device.createBuffer({
      size: MAX_SAMPLES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.densityTex = device.createTexture({
      size: { width: GRID, height: GRID },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })

