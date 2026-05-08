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

