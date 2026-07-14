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

