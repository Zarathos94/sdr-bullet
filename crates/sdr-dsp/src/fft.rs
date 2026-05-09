//! In-place radix-2 FFT over split real and imaginary arrays.
//!
//! Decimation in time, so the input is bit-reversed once up front and the butterflies then
//! run over contiguous memory. Twiddle factors are precomputed into a single flat table
//! laid out per stage: the stage whose half-block is `h` reads `h` consecutive entries
//! starting at `h - 1`. Summing over stages gives `n - 1` entries total, so the whole table
//! costs no more than the transform buffer itself while every stage still reads
//! sequentially.
//!
//! Stages with a half-block of four or more run vectorised. The last two stages are scalar,
//! which is unavoidable and cheap — together they are a vanishing share of the work.

use crate::simd::{F32x4, LANES};
use crate::window;

/// Precomputed plan for one transform size.
///
/// Building a plan allocates; running it does not. Construct once and reuse across blocks.
pub struct Fft {
    n: usize,
    /// Where each element travels during the bit-reversal permutation.
    reversal: Vec<u32>,
    tw_re: Vec<f32>,
    tw_im: Vec<f32>,
}

impl Fft {
    /// # Panics
    /// If `n` is not a power of two, or is smaller than two.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "FFT size must be at least 2");
        assert!(
            n.is_power_of_two(),
            "FFT size must be a power of two, got {n}"
        );

        let bits = n.trailing_zeros();
        let reversal = (0..n)
            .map(|i| (i as u32).reverse_bits() >> (32 - bits))
            .collect();

        // Stage with half-block h occupies [h - 1, 2h - 1), so the table totals n - 1.
        let mut tw_re = vec![0.0f32; n - 1];
        let mut tw_im = vec![0.0f32; n - 1];
        let mut h = 1usize;
        while h < n {
            let m = 2 * h;
            for j in 0..h {
                // exp(-j*2*pi*k/m), computed in double so large transforms stay accurate.
                let angle = -core::f64::consts::TAU * j as f64 / m as f64;
                tw_re[h - 1 + j] = angle.cos() as f32;
                tw_im[h - 1 + j] = angle.sin() as f32;
            }
            h = m;
        }

        Self {
            n,
            reversal,
            tw_re,
            tw_im,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Forward transform, in place.
    ///
    /// # Panics
    /// If either buffer's length differs from the plan size.
    pub fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        assert_eq!(re.len(), self.n, "real buffer must match the plan size");
        assert_eq!(
            im.len(),
            self.n,
            "imaginary buffer must match the plan size"
        );

        self.permute(re, im);

        let mut h = 1usize;
        while h < self.n {
            let m = 2 * h;
            let tw_off = h - 1;
            for base in (0..self.n).step_by(m) {
                let (a_re, b_re) = re[base..base + m].split_at_mut(h);
                let (a_im, b_im) = im[base..base + m].split_at_mut(h);
                if h >= LANES {
                    butterfly_vector(
                        a_re,
                        a_im,
                        b_re,
                        b_im,
                        &self.tw_re[tw_off..tw_off + h],
                        &self.tw_im[tw_off..tw_off + h],
                    );
                } else {
                    butterfly_scalar(
                        a_re,
                        a_im,
                        b_re,
                        b_im,
                        &self.tw_re[tw_off..tw_off + h],
                        &self.tw_im[tw_off..tw_off + h],
                    );
                }
            }
            h = m;
        }
    }

    /// Inverse transform, in place, including the `1/n` scaling.
    ///
    /// Implemented as `conj -> forward -> conj -> scale`. The extra passes are irrelevant
    /// here: nothing in the receive path runs an inverse transform, it exists so the tests
    /// can assert a round trip.
    ///
    /// # Panics
    /// If either buffer's length differs from the plan size.
    pub fn inverse(&self, re: &mut [f32], im: &mut [f32]) {
        for v in im.iter_mut() {
            *v = -*v;
        }
        self.forward(re, im);
        let scale = 1.0 / self.n as f32;
        for v in re.iter_mut() {
            *v *= scale;
        }
        for v in im.iter_mut() {
            *v = -*v * scale;
        }
    }

    fn permute(&self, re: &mut [f32], im: &mut [f32]) {
        for i in 0..self.n {
            let j = self.reversal[i] as usize;
            // Swap once per pair rather than twice.
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
    }

    /// Power of each bin in decibels, relative to a full-scale sine.
    ///
    /// `win` is the window that was applied before the transform; its coherent gain is
    /// divided out so a full-scale tone reads 0 dB regardless of which window was chosen.
    /// Bins are returned in transform order, DC first — see [`shift`] for display order.
    ///
    /// # Panics
    /// If `out` is shorter than the plan size.
    pub fn power_db(&self, re: &[f32], im: &[f32], win: &[f32], out: &mut [f32]) {
        assert!(out.len() >= self.n, "output buffer too short");
        let gain = window::gain(win);
        // Half the spectrum's energy sits in the mirror bin for a real input, and the
        // transform itself scales by n. Fold both into one constant.
        let norm = 1.0 / (self.n as f32 * gain.coherent).max(f32::MIN_POSITIVE);
        let norm_sq = norm * norm;

        // Clamps the log's argument so silence maps to a floor instead of -inf. -200 dB is
        // far below an 8-bit ADC's noise, so nothing real is ever clipped by it.
        const FLOOR: f32 = 1e-20;

        for k in 0..self.n {
            let p = (re[k] * re[k] + im[k] * im[k]) * norm_sq;
            out[k] = 10.0 * (p + FLOOR).log10();
        }
    }

    /// Magnitude of each bin, without normalisation or a logarithm.
    ///
    /// # Panics
    /// If `out` is shorter than the plan size.
    pub fn magnitude(&self, re: &[f32], im: &[f32], out: &mut [f32]) {
        assert!(out.len() >= self.n, "output buffer too short");
        let chunks = self.n / LANES;
        for c in 0..chunks {
            let off = c * LANES;
            // SAFETY: `off + LANES <= n`, and all three buffers are at least `n` long.
            unsafe {
                let r = F32x4::load_unchecked(re, off);
                let i = F32x4::load_unchecked(im, off);
                (r * r + i * i).sqrt().store_unchecked(out, off);
            }
        }
        for k in chunks * LANES..self.n {
            out[k] = (re[k] * re[k] + im[k] * im[k]).sqrt();
        }
    }
}

impl core::fmt::Debug for Fft {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fft")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

/// Rotates a spectrum so negative frequencies come first and DC lands in the middle.
///
/// Complex input produces a two-sided spectrum with DC in bin zero and negative
/// frequencies in the upper half. A waterfall wants them laid out left to right.
///
/// # Panics
/// If `bins` has odd length.
pub fn shift(bins: &mut [f32]) {
    assert!(bins.len() % 2 == 0, "spectrum length must be even to shift");
    let half = bins.len() / 2;
    let (lo, hi) = bins.split_at_mut(half);
    lo.swap_with_slice(hi);
}

/// Butterfly over four bins at a time. Requires `a.len() >= LANES`.
fn butterfly_vector(
    a_re: &mut [f32],
    a_im: &mut [f32],
    b_re: &mut [f32],
    b_im: &mut [f32],
    tw_re: &[f32],
    tw_im: &[f32],
) {
    let h = a_re.len();
    let chunks = h / LANES;
    for c in 0..chunks {
        let off = c * LANES;
        // SAFETY: `off + LANES <= h`, and every slice here is exactly `h` long.
        unsafe {
            let wr = F32x4::load_unchecked(tw_re, off);
            let wi = F32x4::load_unchecked(tw_im, off);
            let br = F32x4::load_unchecked(b_re, off);
            let bi = F32x4::load_unchecked(b_im, off);

            // t = w * b
            let tr = wr * br - wi * bi;
            let ti = wr * bi + wi * br;

            let ar = F32x4::load_unchecked(a_re, off);
            let ai = F32x4::load_unchecked(a_im, off);

            (ar + tr).store_unchecked(a_re, off);
            (ai + ti).store_unchecked(a_im, off);
            (ar - tr).store_unchecked(b_re, off);
            (ai - ti).store_unchecked(b_im, off);
        }
    }
    if chunks * LANES < h {
        let rest = chunks * LANES;
        butterfly_scalar(
            &mut a_re[rest..],
            &mut a_im[rest..],
            &mut b_re[rest..],
            &mut b_im[rest..],
            &tw_re[rest..],
            &tw_im[rest..],
        );
    }
}

fn butterfly_scalar(
    a_re: &mut [f32],
    a_im: &mut [f32],
    b_re: &mut [f32],
    b_im: &mut [f32],
    tw_re: &[f32],
    tw_im: &[f32],
) {
    for j in 0..a_re.len() {
        let (wr, wi) = (tw_re[j], tw_im[j]);
        let (br, bi) = (b_re[j], b_im[j]);
        let tr = wr * br - wi * bi;
        let ti = wr * bi + wi * br;
        let (ar, ai) = (a_re[j], a_im[j]);
        a_re[j] = ar + tr;
        a_im[j] = ai + ti;
        b_re[j] = ar - tr;
        b_im[j] = ai - ti;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    /// Direct evaluation of the transform definition. Quadratic, so only for small sizes,
    /// but it depends on none of the machinery under test.
    fn naive_dft(re: &[f32], im: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = re.len();
        let mut out_re = vec![0.0; n];
        let mut out_im = vec![0.0; n];
        for k in 0..n {
            let (mut sr, mut si) = (0.0f64, 0.0f64);
            for t in 0..n {
                let ang = -core::f64::consts::TAU * (k as f64) * (t as f64) / n as f64;
                let (c, s) = (ang.cos(), ang.sin());
                sr += re[t] as f64 * c - im[t] as f64 * s;
                si += re[t] as f64 * s + im[t] as f64 * c;
            }
            out_re[k] = sr as f32;
            out_im[k] = si as f32;
        }
        (out_re, out_im)
    }

    fn pseudo_random(n: usize) -> (Vec<f32>, Vec<f32>) {
        // A deterministic, poorly-correlated sequence. Good enough to catch index errors
        // and avoids pulling in a dependency for a handful of numbers.
        let mut state = 0x2545_F491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let re = (0..n).map(|_| next()).collect();
        let im = (0..n).map(|_| next()).collect();
        (re, im)
    }

    #[test]
    fn matches_the_definition_across_sizes() {
        for &n in &[2usize, 4, 8, 16, 32, 64, 128, 256] {
            let (re, im) = pseudo_random(n);
            let (want_re, want_im) = naive_dft(&re, &im);

            let mut got_re = re.clone();
            let mut got_im = im.clone();
            Fft::new(n).forward(&mut got_re, &mut got_im);

            // Error grows with the number of stages, hence the size-dependent tolerance.
            let tol = 1e-4 * n as f32;
            for k in 0..n {
                assert!(
                    (got_re[k] - want_re[k]).abs() < tol && (got_im[k] - want_im[k]).abs() < tol,
                    "n={n} bin {k}: got ({}, {}) want ({}, {})",
                    got_re[k],
                    got_im[k],
                    want_re[k],
                    want_im[k]
                );
            }
        }
    }

    #[test]
    fn dc_input_puts_all_energy_in_bin_zero() {
        let n = 64;
        let mut re = vec![1.0f32; n];
        let mut im = vec![0.0f32; n];
        Fft::new(n).forward(&mut re, &mut im);

        assert!((re[0] - n as f32).abs() < 1e-3);
        for k in 1..n {
            assert!(
                re[k].abs() < 1e-3 && im[k].abs() < 1e-3,
                "leakage into bin {k}"
            );
        }
    }

