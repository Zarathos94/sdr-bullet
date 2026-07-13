/**
 * Colormaps for the spectral displays, as RGB lookup tables.
 *
 * A spectrogram encodes power as colour, and the eye reads that colour as a magnitude — so
 * the mapping has to be perceptually uniform, meaning equal steps in decibels look like
 * equal steps in brightness. A naive rainbow (hue swept linearly) fails this badly: the
 * yellow-to-green band is far brighter than the blue band, so it invents a bright ridge
 * where the signal is flat and hides real detail inside the dim blues. Viridis and inferno
 * were designed against exactly that failure — their lightness rises monotonically — which
 * is why a weak carrier or a faint sideband stays legible against the noise floor rather
 * than being lost to a colour artefact.
 *
 * The tables are filled from the polynomial fits that approximate the matplotlib maps, so
 * the values here are accurate to a few least-significant bits without carrying a thousand
 * hand-typed literals. That is close enough for a display; nothing downstream measures off
 * the colours.
 */

export type ColormapName = 'viridis' | 'inferno' | 'grayscale'

/** Entries per table. 256 is the resolution an 8-bit display can actually show. */
export const COLORMAP_SIZE = 256

/**
 * Six-term polynomial fits of viridis and inferno, evaluated per channel with Horner's
 * method. The coefficient sets are the widely reproduced approximations of the matplotlib
 * originals; the fit error is well under one 8-bit level across the range.
 */
type Coefficients = readonly (readonly [number, number, number])[]

const VIRIDIS: Coefficients = [
  [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
  [0.1050930431085774, 1.404613529898575, 1.384590162594685],
  [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
  [-4.634230498983486, -5.799100973351585, -19.33244095627987],
  [6.228269936347081, 14.17993336680509, 56.69055260068105],
  [4.776384997670288, -13.74514537774601, -65.35303263337234],
  [-5.435455855934631, 4.645852612178535, 26.3124352495832],
]

const INFERNO: Coefficients = [
  [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
  [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
  [11.60249308247187, -3.972853965665698, -15.9423941062914],
  [-41.70399613139459, 17.43639888205313, 44.35414519872813],
  [77.162935699427, -33.40235894210092, -81.80730925738993],
  [-71.31942824499214, 32.62606426397723, 73.20951985803202],
  [25.13112622477341, -12.24266895238567, -23.07032500287172],
]

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

/** Evaluates a fit at `t` in [0, 1], highest-order term first so Horner runs cleanly. */
function evaluate(coefficients: Coefficients): Float32Array {
  const table = new Float32Array(COLORMAP_SIZE * 3)
  for (let i = 0; i < COLORMAP_SIZE; i++) {
    const t = i / (COLORMAP_SIZE - 1)
    let r = 0
    let g = 0
    let b = 0
    for (let k = coefficients.length - 1; k >= 0; k--) {
      const [cr, cg, cb] = coefficients[k]!
      r = r * t + cr
      g = g * t + cg
      b = b * t + cb
    }
    table[i * 3 + 0] = clamp01(r)
    table[i * 3 + 1] = clamp01(g)
    table[i * 3 + 2] = clamp01(b)
  }
  return table
}

function grayscale(): Float32Array {
  const table = new Float32Array(COLORMAP_SIZE * 3)
  for (let i = 0; i < COLORMAP_SIZE; i++) {
    const t = i / (COLORMAP_SIZE - 1)
    table[i * 3 + 0] = t
    table[i * 3 + 1] = t
    table[i * 3 + 2] = t
  }
  return table
}

/**
 * The maps, keyed by name. Each is `COLORMAP_SIZE` RGB triples of floats in [0, 1].
 *
 * Viridis reads a spectrogram best; inferno's zero is very nearly black, which makes it the
 * natural choice for a scope where empty space should stay dark rather than tinted.
 */
export const COLORMAPS: Record<ColormapName, Float32Array> = {
  viridis: evaluate(VIRIDIS),
  inferno: evaluate(INFERNO),
  grayscale: grayscale(),
}

/**
 * The same table packed as `rgba8unorm` texels, ready for `writeTexture` into the LUT the
 * shaders sample. Alpha is opaque; the display never blends by the colormap's own alpha.
 */
export function colormapTexels(name: ColormapName): Uint8Array {
  const table = COLORMAPS[name]
  const texels = new Uint8Array(COLORMAP_SIZE * 4)
  for (let i = 0; i < COLORMAP_SIZE; i++) {
    texels[i * 4 + 0] = Math.round((table[i * 3 + 0] ?? 0) * 255)
    texels[i * 4 + 1] = Math.round((table[i * 3 + 1] ?? 0) * 255)
    texels[i * 4 + 2] = Math.round((table[i * 3 + 2] ?? 0) * 255)
    texels[i * 4 + 3] = 255
  }
  return texels
}
