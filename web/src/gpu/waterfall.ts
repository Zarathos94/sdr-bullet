/**
 * The scrolling spectrogram.
 *
 * The load-bearing idea is that the history is never moved. A spectrogram scrolls by one row
 * per new spectrum, and the obvious implementation — shift every row down and write the new
 * one at the top — copies the entire image every frame. Instead the rows live in a texture
 * used as a ring buffer: a new row is written at a moving cursor with a single `writeTexture`,
 * and the render pass reads the ring with the cursor as an offset, so the scroll is an index
 * calculation rather than a copy. Nothing is blitted, and the per-frame cost is one row
 * uploaded plus one fullscreen pass, independent of how deep the history is.
 *
 * The magnitudes are kept as raw decibels in an `r32float` ring (core in WebGPU, no feature
 * needed). Colour is applied by a compute pass that maps dB through the current range and a
 * colormap LUT into a second ring of `rgba8unorm`; the render pass then only samples colour.
 * Recolouring the whole ring each frame — rather than colouring each row once as it arrives —
 * is deliberate: the auto-range drifts, and a row must show the same colour for the same power
 * regardless of what the range happened to be when it was captured.
 */

import { colormapTexels, type ColormapName } from './colormap.js'

/** 2D colour pass tiling. 16 x 16 = 256 invocations, the guaranteed workgroup-size ceiling. */
const TILE = 16

/** Reduction workgroup size. Also 256; the shared arrays below are sized to match. */
const REDUCE_WG = 256

/** Fraction of the gap to a fresh auto-range reading closed per readback. Low enough not to flicker. */
const SMOOTH_RATE = 0.1

const REDUCE_WGSL = /* wgsl */ `
struct RParams { total: u32, width: u32, pad0: u32, pad1: u32 };
struct MinMax { minv: f32, maxv: f32 };

@group(0) @binding(0) var ringTex: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var<uniform> rp: RParams;
@group(0) @binding(2) var<storage, read_write> outMM: MinMax;

// One row of scratch per lane. 2 x 256 x 4 = 2048 bytes, far under the 16384-byte limit.
var<workgroup> smin: array<f32, ${REDUCE_WG}>;
var<workgroup> smax: array<f32, ${REDUCE_WG}>;

// A single workgroup reduces the whole ring. WGSL has no float atomics, so a multi-workgroup
// reduction would need the bit-twiddling ordered-integer trick to combine partial results;
// one workgroup grid-striding the texture sidesteps that entirely, and a few hundred thousand
// texels is a sub-millisecond job for 256 lanes hiding each other's memory latency.
@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let width = max(rp.width, 1u);
  var mn = 3.0e38;
  var mx = -3.0e38;
  var i = lid.x;
  loop {
    if (i >= rp.total) { break; }
    let coord = vec2<i32>(i32(i % width), i32(i / width));
    let db = textureLoad(ringTex, coord).r;
    mn = min(mn, db);
    mx = max(mx, db);
    i = i + ${REDUCE_WG}u;
  }
  smin[lid.x] = mn;
  smax[lid.x] = mx;
  workgroupBarrier();

  // Tree reduction: halve the active range each step until lane 0 holds the extremes.
  var stride = ${REDUCE_WG / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      smin[lid.x] = min(smin[lid.x], smin[lid.x + stride]);
      smax[lid.x] = max(smax[lid.x], smax[lid.x + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lid.x == 0u) {
    outMM.minv = smin[0];
    outMM.maxv = smax[0];
  }
}
`

const COLOUR_WGSL = /* wgsl */ `
struct Params {
  minDb: f32, maxDb: f32,
  writeRow: u32, binCount: u32, rowCount: u32,
  pad0: u32, pad1: u32, pad2: u32,
};

@group(0) @binding(0) var ringTex: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var lut: texture_2d<f32>;
@group(0) @binding(2) var lutSamp: sampler;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.binCount || gid.y >= p.rowCount) { return; }
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let db = textureLoad(ringTex, coord).r;
  let span = max(p.maxDb - p.minDb, 1.0e-6);
  let t = clamp((db - p.minDb) / span, 0.0, 1.0);
  // textureSampleLevel (not textureSample) because a compute stage has no implicit derivatives;
  // sampling the LUT gives smooth interpolation between its 256 entries for free.
  let rgb = textureSampleLevel(lut, lutSamp, vec2<f32>(t, 0.5), 0.0).rgb;
  textureStore(dst, coord, vec4<f32>(rgb, 1.0));
}
`

const DRAW_WGSL = /* wgsl */ `
struct Params {
  minDb: f32, maxDb: f32,
  writeRow: u32, binCount: u32, rowCount: u32,
  pad0: u32, pad1: u32, pad2: u32,
};

@group(0) @binding(0) var colourTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> p: Params;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
  // One oversized triangle covering the viewport — cheaper than a quad and with no seam.
  var corners = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
  );
  let xy = corners[vid];
  var o: VOut;
  o.pos = vec4<f32>(xy, 0.0, 1.0);
  // uv.y = 0 at the top of the screen, where the newest row belongs.
  o.uv = vec2<f32>((xy.x + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let rows = i32(p.rowCount);
  let col = clamp(i32(in.uv.x * f32(p.binCount)), 0, i32(p.binCount) - 1);

  // The ring stores rows in the order they were written; the newest sits one slot before the
  // write cursor. Descending the screen walks backwards in time, so the row index is the
  // newest minus the number of rows down. The result is taken modulo the row count to hide the
  // seam where the cursor wrapped; the extra "+ rows" before the final modulo keeps the value
  // non-negative, since WGSL's remainder follows the sign of the dividend.
  let back = i32(round(in.uv.y * f32(rows - 1)));
  let newest = i32(p.writeRow) - 1;
  var row = (newest - back) % rows;
  row = (row + rows) % rows;

  let c = textureLoad(colourTex, vec2<i32>(col, row), 0);
  return vec4<f32>(c.rgb, 1.0);
}
`

export class WaterfallRenderer {
  private readonly context: GPUCanvasContext
  private readonly format: GPUTextureFormat

  private readonly reducePipeline: GPUComputePipeline
  private readonly colourPipeline: GPUComputePipeline
  private readonly drawPipeline: GPURenderPipeline

  // Explicit layouts for the two passes that read the ring, so the r32float ring binds as a
  // read-only storage texture and the colour target as write-only — the access and format are
  // pinned rather than left to inference.
  private readonly reduceLayout: GPUBindGroupLayout
  private readonly colourLayout: GPUBindGroupLayout

  // Static resources, sized once. The LUT is 256 x 1 and shared by every colour pass.
  private readonly lutTex: GPUTexture
  private readonly lutSampler: GPUSampler
  private readonly paramsBuf: GPUBuffer
  private readonly reduceBuf: GPUBuffer
  private readonly minmaxBuf: GPUBuffer
  private readonly stagingBuf: GPUBuffer
  private readonly paramsData = new ArrayBuffer(32)
  private readonly paramsView = new DataView(this.paramsData)
  private readonly reduceData = new ArrayBuffer(16)
  private readonly reduceView = new DataView(this.reduceData)

  // Per-size resources, rebuilt whenever the bin or row count changes.
  private ringTex: GPUTexture | undefined
  private colourTex: GPUTexture | undefined
  private reduceBind: GPUBindGroup | undefined
  private colourBind: GPUBindGroup | undefined
  private drawBind: GPUBindGroup | undefined

  private binCount = 0
  private rowCount = 0
  /** Next row the ring will be written to; the newest row is `cursor - 1`. */
  private cursor = 0
  /**
   * Rows written since the last (re)allocation, capped at the ring height. Before the ring has
   * filled, the written rows are exactly [0, cursor); the auto-range reduction reads only those
   * so the prefilled floor does not peg the computed minimum until real rows have replaced it.
   */
  private rowsPushed = 0

  private manualMin = -100
  private manualMax = -20
  private autoRange = true
  private smMin = 0
  private smMax = 0
  private rangeReady = false
  /** True while a range readback is mapped; a fresh copy must not be encoded until it clears. */
  private rangePending = false

  constructor(
    private readonly device: GPUDevice,
    private readonly canvas: HTMLCanvasElement,
  ) {
    const context = canvas.getContext('webgpu')
    if (!context) throw new Error('WaterfallRenderer: could not get a webgpu canvas context')
    this.context = context
    this.format = navigator.gpu.getPreferredCanvasFormat()
    // Opaque: the waterfall fills every pixel it owns, so there is nothing to blend with.
    context.configure({ device, format: this.format, alphaMode: 'opaque' })

    const reduceModule = device.createShaderModule({ code: REDUCE_WGSL })
    const colourModule = device.createShaderModule({ code: COLOUR_WGSL })
    const drawModule = device.createShaderModule({ code: DRAW_WGSL })

    // The ring is an r32float storage texture, read-only in the shaders: writeTexture fills it a
    // row at a time (its COPY_DST usage) and the passes only ever textureLoad it. r32float storage
    // is core, so nothing here needs a requiredFeatures entry.
    const ringEntry: GPUBindGroupLayoutEntry = {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'read-only', format: 'r32float', viewDimension: '2d' },
    }
    this.reduceLayout = device.createBindGroupLayout({
      entries: [
        ringEntry,
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    this.colourLayout = device.createBindGroupLayout({
      entries: [
        ringEntry,
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba8unorm' },
        },
      ],
    })

    this.reducePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.reduceLayout] }),
      compute: { module: reduceModule, entryPoint: 'main' },
    })
    this.colourPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.colourLayout] }),
      compute: { module: colourModule, entryPoint: 'main' },
    })
    // The draw pass samples only the rgba8unorm colour ring, which is filterable, so an auto
    // layout infers a compatible 'float' binding without help.
    this.drawPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: drawModule, entryPoint: 'vs' },
      fragment: { module: drawModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    })

    this.lutSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })
    this.lutTex = device.createTexture({
      size: { width: 256, height: 1 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.setColormap('viridis')

    this.paramsBuf = device.createBuffer({
      size: this.paramsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.reduceBuf = device.createBuffer({
      size: this.reduceData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.minmaxBuf = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })
    this.stagingBuf = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })

    // Adopt whatever size the canvas already has; a later resize() re-sizes cleanly.
    if (canvas.width > 0 && canvas.height > 0) {
      this.rowCount = Math.min(canvas.height, device.limits.maxTextureDimension2D)
    }
  }

  /** Replaces the colour scale. Takes effect on the next frame; history recolours with it. */
  setColormap(name: ColormapName): void {
    this.device.queue.writeTexture(
      { texture: this.lutTex },
      // Freshly allocated, so plain-backed; the cast states that for TypeScript 5.7's
      // buffer-generic array types, which the GPU queue's non-shared requirement needs.
      colormapTexels(name) as Uint8Array<ArrayBuffer>,
      { bytesPerRow: 256 * 4, rowsPerImage: 1 },
      { width: 256, height: 1, depthOrArrayLayers: 1 },
    )
  }

  /** Fixes the dB window used when auto-range is off. */
  setRange(minDb: number, maxDb: number): void {
    this.manualMin = minDb
    this.manualMax = maxDb
  }

  /** Turns the reduction-driven auto-range on or off. When off, {@link setRange} governs. */
  setAutoRange(enabled: boolean): void {
    this.autoRange = enabled
  }

  /**
   * Sizes the backing store. `width`/`height` arrive as physical device pixels — the caller
   * has already multiplied the CSS box by `devicePixelRatio`, which is what makes the surface
   * sharp on a hi-dpi display instead of upscaled. History depth follows the panel height, one
   * texture row per screen row, so a resize restarts the history; for a live display that is a
   * fair trade against carrying stale rows at the wrong scale.
   */
  resize(width: number, height: number): void {
    const max = this.device.limits.maxTextureDimension2D
    const w = Math.max(1, Math.min(Math.round(width), max))
    const h = Math.max(1, Math.min(Math.round(height), max))
    if (this.canvas.width !== w) this.canvas.width = w
    if (this.canvas.height !== h) this.canvas.height = h
    if (h !== this.rowCount) {
      this.rowCount = h
      if (this.binCount > 0) this.allocate()
    }
  }

  /** Appends one spectrum row, dB values in display order (lowest frequency first). */
  pushRow(bins: Float32Array): void {
    if (bins.length === 0) return
    if (bins.length !== this.binCount) {
      this.binCount = bins.length
      if (this.rowCount > 0) this.allocate()
    }
    if (!this.ringTex || this.rowCount === 0) return

    this.device.queue.writeTexture(
      { texture: this.ringTex, origin: { x: 0, y: this.cursor, z: 0 } },
      // The spectrum frame from the pipeline is a plain array, not a view over the shared
      // ring; the cast records the non-shared backing the GPU queue requires.
      bins as Float32Array<ArrayBuffer>,
      { bytesPerRow: this.binCount * 4, rowsPerImage: 1 },
      { width: this.binCount, height: 1, depthOrArrayLayers: 1 },
    )
    this.cursor = (this.cursor + 1) % this.rowCount
    if (this.rowsPushed < this.rowCount) this.rowsPushed++
  }

  render(): void {
    if (
      !this.ringTex ||
      !this.colourTex ||
      !this.colourBind ||
      !this.drawBind ||
      !this.reduceBind ||
      this.binCount === 0 ||
      this.rowCount === 0
    ) {
      return
    }

    let minDb = this.manualMin
    let maxDb = this.manualMax
    if (this.autoRange && this.rangeReady) {
      minDb = this.smMin
      maxDb = this.smMax
    }
    if (maxDb <= minDb) maxDb = minDb + 1
    this.writeParams(minDb, maxDb)

    const enc = this.device.createCommandEncoder()

    let doMap = false
    // Only reduce once there is real data; reducing an empty ring would report the sentinels.
    if (this.autoRange && this.rowsPushed > 0) {
      this.writeReduceParams()
      const pass = enc.beginComputePass()
      pass.setPipeline(this.reducePipeline)
      pass.setBindGroup(0, this.reduceBind)
      pass.dispatchWorkgroups(1, 1, 1)
      pass.end()
      // Only stage a readback when the previous one has been consumed; a mapped buffer cannot
      // be a copy target. Frames between readbacks reuse the last smoothed range.
      if (!this.rangePending) {
        enc.copyBufferToBuffer(this.minmaxBuf, 0, this.stagingBuf, 0, 8)
        doMap = true
      }
    }

    const colour = enc.beginComputePass()
    colour.setPipeline(this.colourPipeline)
    colour.setBindGroup(0, this.colourBind)
    colour.dispatchWorkgroups(
      Math.ceil(this.binCount / TILE),
      Math.ceil(this.rowCount / TILE),
      1,
    )
    colour.end()

    const view = this.context.getCurrentTexture().createView()
    const draw = enc.beginRenderPass({
      colorAttachments: [
        { view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
    })
    draw.setPipeline(this.drawPipeline)
    draw.setBindGroup(0, this.drawBind)
    draw.draw(3)
    draw.end()

    this.device.queue.submit([enc.finish()])

    if (doMap) {
      this.rangePending = true
      void this.stagingBuf
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const arr = new Float32Array(this.stagingBuf.getMappedRange())
          const mn = arr[0] ?? Number.NaN
          const mx = arr[1] ?? Number.NaN
          this.stagingBuf.unmap()
          this.absorbRange(mn, mx)
          this.rangePending = false
        })
        .catch(() => {
          this.rangePending = false
        })
    }
  }

  dispose(): void {
    this.ringTex?.destroy()
    this.colourTex?.destroy()
    this.lutTex.destroy()
    this.paramsBuf.destroy()
    this.reduceBuf.destroy()
    this.minmaxBuf.destroy()
    this.stagingBuf.destroy()
    this.ringTex = undefined
    this.colourTex = undefined
    this.reduceBind = undefined
    this.colourBind = undefined
    this.drawBind = undefined
  }

  private absorbRange(mn: number, mx: number): void {
    if (!Number.isFinite(mn) || !Number.isFinite(mx)) return
    if (!this.rangeReady) {
      this.smMin = mn
      this.smMax = mx
      this.rangeReady = true
      return
    }
    this.smMin += (mn - this.smMin) * SMOOTH_RATE
    this.smMax += (mx - this.smMax) * SMOOTH_RATE
  }

  private writeParams(minDb: number, maxDb: number): void {
    const v = this.paramsView
    v.setFloat32(0, minDb, true)
    v.setFloat32(4, maxDb, true)
    v.setUint32(8, this.cursor >>> 0, true)
    v.setUint32(12, this.binCount >>> 0, true)
    v.setUint32(16, this.rowCount >>> 0, true)
    this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsData)
  }

  private writeReduceParams(): void {
    const v = this.reduceView
    // Only the rows written so far, which are contiguous at [0, rowsPushed) until the ring wraps
    // and rowsPushed saturates at the full height.
    v.setUint32(0, (this.binCount * this.rowsPushed) >>> 0, true)
    v.setUint32(4, this.binCount >>> 0, true)
    this.device.queue.writeBuffer(this.reduceBuf, 0, this.reduceData)
  }

  /** (Re)allocates the size-dependent textures and their bind groups, and clears the history. */
  private allocate(): void {
    this.ringTex?.destroy()
    this.colourTex?.destroy()

    const size = { width: this.binCount, height: this.rowCount }
    this.ringTex = this.device.createTexture({
      size,
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.colourTex = this.device.createTexture({
      size,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })

    // Prime the ring with the floor value so unfilled history renders as the colormap's low
    // end rather than the zero a fresh texture would otherwise show as a bright band.
    const fill = new Float32Array(this.binCount * this.rowCount).fill(this.manualMin)
    this.device.queue.writeTexture(
      { texture: this.ringTex },
      fill,
      { bytesPerRow: this.binCount * 4, rowsPerImage: this.rowCount },
      { width: this.binCount, height: this.rowCount, depthOrArrayLayers: 1 },
    )
    this.cursor = 0
    this.rowsPushed = 0

    const ringView = this.ringTex.createView()
    const colourView = this.colourTex.createView()

    this.reduceBind = this.device.createBindGroup({
      layout: this.reduceLayout,
      entries: [
        { binding: 0, resource: ringView },
        { binding: 1, resource: { buffer: this.reduceBuf } },
        { binding: 2, resource: { buffer: this.minmaxBuf } },
      ],
    })
    this.colourBind = this.device.createBindGroup({
      layout: this.colourLayout,
      entries: [
        { binding: 0, resource: ringView },
        { binding: 1, resource: this.lutTex.createView() },
        { binding: 2, resource: this.lutSampler },
        { binding: 3, resource: { buffer: this.paramsBuf } },
        { binding: 4, resource: colourView },
      ],
    })
    this.drawBind = this.device.createBindGroup({
      layout: this.drawPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: colourView },
        { binding: 1, resource: { buffer: this.paramsBuf } },
      ],
    })
  }
}
