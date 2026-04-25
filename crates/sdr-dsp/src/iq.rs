//! Front-end conditioning: byte unpacking, DC removal, and I/Q imbalance correction.
//!
//! The RTL2832U delivers interleaved unsigned bytes in offset binary, centred on 127.5.
//! Everything downstream wants deinterleaved floats, so this is the one place that walks
//! interleaved memory.

use crate::simd::{F32x4, LANES};

/// Number of distinct sample values the 8-bit ADC can produce.
const LEVELS: usize = 256;

/// Maps every possible input byte to its float value once, at construction.
///
/// Only 256 inputs exist, so the conversion is a table lookup rather than an integer
/// widen and a divide. That also keeps the unpack loop free of SIMD integer work, which
/// the [`F32x4`] abstraction deliberately does not carry.
fn build_level_table() -> [f32; LEVELS] {
    let mut table = [0.0f32; LEVELS];
    for (i, slot) in table.iter_mut().enumerate() {
        *slot = (i as f32 - 127.5) / 127.5;
    }
    table
}

/// Running estimates used to correct the front end's systematic errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corrections {
    pub dc_i: f32,
    pub dc_q: f32,
    /// Q gain relative to I. 1.0 means the two branches are balanced.
    pub gain_q: f32,
    /// Quadrature skew in radians. 0.0 means the branches are orthogonal.
    pub phase: f32,
}

impl Default for Corrections {
    fn default() -> Self {
        Self {
            dc_i: 0.0,
            dc_q: 0.0,
            gain_q: 1.0,
            phase: 0.0,
        }
    }
}

/// Unpacks receiver bytes and removes the front end's DC offset and quadrature errors.
///
/// DC is removed by subtracting a slowly adapting estimate of the block mean rather than
/// by running a one-pole high-pass. The offset in these tuners is a static mixer and ADC
/// artefact, not a drifting quantity, and mean subtraction leaves genuine near-DC content
/// intact — a high-pass steep enough to track the offset also eats the centre of the
/// passband, which matters because a tuned channel sits exactly there.
#[derive(Debug, Clone)]
pub struct IqConverter {
    table: [f32; LEVELS],
    corrections: Corrections,
    /// Per-block smoothing factor for the correction estimates.
    adapt: f32,
    /// Estimates stay frozen until the first block has been observed.
    primed: bool,
    correct_imbalance: bool,
}

impl IqConverter {
    /// `adapt` is the per-block weight given to a new observation, in `(0, 1]`. At the
    /// default of 0.05 an estimate settles over roughly twenty blocks.
    pub fn new() -> Self {
        Self {
            table: build_level_table(),
            corrections: Corrections::default(),
            adapt: 0.05,
            primed: false,
            correct_imbalance: true,
        }
    }

    pub fn with_adapt_rate(mut self, adapt: f32) -> Self {
        assert!(adapt > 0.0 && adapt <= 1.0, "adapt rate must be in (0, 1]");
        self.adapt = adapt;
        self
    }

    pub fn set_imbalance_correction(&mut self, enabled: bool) {
        self.correct_imbalance = enabled;
    }

    pub fn corrections(&self) -> Corrections {
        self.corrections
    }

    /// Resets the adaptive state. Call after retuning, since the offsets are
    /// frequency-dependent and the old estimate is worse than no estimate.
    pub fn reset(&mut self) {
        self.corrections = Corrections::default();
        self.primed = false;
    }

    /// Unpacks `bytes` into `i` and `q`, applying the corrections learned so far, then
    /// updates those estimates from what it just saw.
    ///
    /// Returns the number of complex samples written.
    ///
    /// # Panics
    /// If `bytes` holds an odd number of elements, or the outputs are too short.
    pub fn process(&mut self, bytes: &[u8], i: &mut [f32], q: &mut [f32]) -> usize {
        assert!(
            bytes.len() % 2 == 0,
            "IQ byte stream must be a whole number of pairs"
        );
        let n = bytes.len() / 2;
        assert!(i.len() >= n && q.len() >= n, "output buffers too short");

        for k in 0..n {
            i[k] = self.table[bytes[2 * k] as usize];
            q[k] = self.table[bytes[2 * k + 1] as usize];
        }

        let i = &mut i[..n];
        let q = &mut q[..n];

        // Correct using the previous block's estimate before folding this block in. Using
        // a block's own statistics to correct itself would partially cancel real signal
        // that happens to sit near DC.
        if self.primed {
            apply_dc(i, self.corrections.dc_i);
            apply_dc(q, self.corrections.dc_q);
            if self.correct_imbalance {
                apply_imbalance(i, q, self.corrections.gain_q, self.corrections.phase);
            }
        }

        self.observe(i, q);
        n
    }

    /// Folds one block's statistics into the running estimates.
    fn observe(&mut self, i: &[f32], q: &[f32]) {
        let n = i.len();
        if n == 0 {
            return;
        }
        let inv_n = 1.0 / n as f32;

        let (sum_i, sum_q) = sum_pair(i, q);
        let mean_i = sum_i * inv_n;
        let mean_q = sum_q * inv_n;

        let (pow_i, pow_q, cross) = second_moments(i, q, mean_i, mean_q);
        let pow_i = pow_i * inv_n;
        let pow_q = pow_q * inv_n;
        let cross = cross * inv_n;

        // A gain ratio is only meaningful once there is signal to measure it on. Below
        // this the estimate is dominated by quantisation noise and starts chasing it, so
        // report "nothing further to correct" instead.
        const POWER_FLOOR: f32 = 1e-9;
        let (gain_residual, phase_residual) = if pow_i > POWER_FLOOR && pow_q > POWER_FLOOR {
            // Amplitude ratio between branches, and the correlation that quadrature
            // signals should not have. For orthogonal branches E[I*Q] is zero.
            (
                (pow_i / pow_q).sqrt(),
                (cross / (pow_i.sqrt() * pow_q.sqrt())).clamp(-0.9, 0.9),
            )
        } else {
            (1.0, 0.0)
        };

        if self.primed {
            let a = self.adapt;
            // Every quantity here was measured on a block that has already been corrected,
            // so what comes back is the error still remaining, not the total error. It has
            // to be composed onto the running estimate — averaging the estimate towards a
            // residual of "almost none" would steadily undo the correction that produced
            // that residual in the first place.
            //
            // DC and phase compose by addition; gain is multiplicative, so it composes by
            // scaling. Smoothing is applied to the step rather than the target.
            self.corrections.dc_i += mean_i * a;
            self.corrections.dc_q += mean_q * a;
            self.corrections.gain_q *= 1.0 + (gain_residual - 1.0) * a;
            self.corrections.phase += phase_residual * a;
        } else {
            // Nothing has been corrected yet, so the residual is the whole error.
            self.corrections = Corrections {
                dc_i: mean_i,
                dc_q: mean_q,
                gain_q: gain_residual,
                phase: phase_residual,
            };
            self.primed = true;
        }
    }
}

impl Default for IqConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Subtracts a constant from every element.
fn apply_dc(x: &mut [f32], dc: f32) {
    let v = F32x4::splat(dc);
    let chunks = x.len() / LANES;
    for c in 0..chunks {
        let off = c * LANES;
        // SAFETY: `off + LANES <= chunks * LANES <= x.len()`.
        unsafe {
            let s = F32x4::load_unchecked(x, off);
            (s - v).store_unchecked(x, off);
        }
    }
    for s in &mut x[chunks * LANES..] {
        *s -= dc;
    }
}

/// Applies the Gram-Schmidt style correction that restores orthogonality between branches.
///
/// Q is rescaled to match I's amplitude, then the component of Q that correlates with I is
/// projected out and the result renormalised so total power is preserved.
fn apply_imbalance(i: &mut [f32], q: &mut [f32], gain_q: f32, phase: f32) {
    let denom = (1.0 - phase * phase).sqrt();
    if !denom.is_finite() || denom < 1e-6 {
        return;
    }
    let vg = F32x4::splat(gain_q);
    let vp = F32x4::splat(phase);
    let vn = F32x4::splat(1.0 / denom);

    let n = i.len().min(q.len());
    let chunks = n / LANES;
    for c in 0..chunks {
        let off = c * LANES;
        // SAFETY: `off + LANES <= n`, and both slices are at least `n` long.
        unsafe {
            let vi = F32x4::load_unchecked(i, off);
            let vq = F32x4::load_unchecked(q, off) * vg;
            ((vq - vi * vp) * vn).store_unchecked(q, off);
        }
    }
    for k in chunks * LANES..n {
        let scaled = q[k] * gain_q;
        q[k] = (scaled - i[k] * phase) / denom;
    }
}

/// Sums both slices in one pass so they share a single trip through cache.
fn sum_pair(i: &[f32], q: &[f32]) -> (f32, f32) {
    let n = i.len().min(q.len());
    let chunks = n / LANES;
    let mut acc_i = F32x4::zero();
    let mut acc_q = F32x4::zero();
    for c in 0..chunks {
        let off = c * LANES;
        // SAFETY: `off + LANES <= n`, and both slices are at least `n` long.
        unsafe {
            acc_i = acc_i + F32x4::load_unchecked(i, off);
            acc_q = acc_q + F32x4::load_unchecked(q, off);
        }
    }
    let mut si = acc_i.sum();
    let mut sq = acc_q.sum();
    for k in chunks * LANES..n {
        si += i[k];
        sq += q[k];
    }
    (si, sq)
}

/// Returns `(sum(i^2), sum(q^2), sum(i*q))` about the supplied means.
fn second_moments(i: &[f32], q: &[f32], mean_i: f32, mean_q: f32) -> (f32, f32, f32) {
    let n = i.len().min(q.len());
    let chunks = n / LANES;
    let vmi = F32x4::splat(mean_i);
    let vmq = F32x4::splat(mean_q);
    let mut acc_ii = F32x4::zero();
    let mut acc_qq = F32x4::zero();
    let mut acc_iq = F32x4::zero();
    for c in 0..chunks {
        let off = c * LANES;
        // SAFETY: `off + LANES <= n`, and both slices are at least `n` long.
        unsafe {
            let vi = F32x4::load_unchecked(i, off) - vmi;
            let vq = F32x4::load_unchecked(q, off) - vmq;
            acc_ii = acc_ii + vi * vi;
            acc_qq = acc_qq + vq * vq;
            acc_iq = acc_iq + vi * vq;
        }
    }
    let mut ii = acc_ii.sum();
    let mut qq = acc_qq.sum();
    let mut iq = acc_iq.sum();
    for k in chunks * LANES..n {
        let vi = i[k] - mean_i;
        let vq = q[k] - mean_q;
        ii += vi * vi;
        qq += vq * vq;
        iq += vi * vq;
    }
    (ii, qq, iq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    fn mean(x: &[f32]) -> f32 {
        x.iter().sum::<f32>() / x.len() as f32
    }

    #[test]
    fn midscale_bytes_map_near_zero() {
        let table = build_level_table();
        // Offset binary has no exact zero: 127 and 128 straddle it symmetrically.
        assert!((table[127] + table[128]).abs() < 1e-6);
        assert_eq!(table[0], -1.0);
        assert!((table[255] - 1.0).abs() < 0.01);
    }

    #[test]
    fn unpacks_interleaved_pairs_in_order() {
        let mut c = IqConverter::new();
        c.set_imbalance_correction(false);
        let bytes = [0u8, 255, 64, 192];
        let mut i = [0.0; 2];
        let mut q = [0.0; 2];
        assert_eq!(c.process(&bytes, &mut i, &mut q), 2);

        let t = build_level_table();
        // First block is uncorrected, so values pass through the table untouched.
        assert_eq!(i, [t[0], t[64]]);
        assert_eq!(q, [t[255], t[192]]);
    }

    #[test]
    fn converges_on_a_constant_dc_offset() {
        let mut c = IqConverter::new().with_adapt_rate(0.5);
        c.set_imbalance_correction(false);

        // A tone offset so that I sits high and Q sits low.
        let n = 1024;
        let bytes: Vec<u8> = (0..n)
            .flat_map(|k| {
                let ph = TAU * 0.01 * k as f32;
                let i = 160.0 + 40.0 * ph.cos();
                let q = 96.0 + 40.0 * ph.sin();
                [i as u8, q as u8]
            })
            .collect();

        let mut i = vec![0.0; n];
        let mut q = vec![0.0; n];
        for _ in 0..40 {
            c.process(&bytes, &mut i, &mut q);
        }

        assert!(mean(&i).abs() < 0.01, "I mean not removed: {}", mean(&i));
        assert!(mean(&q).abs() < 0.01, "Q mean not removed: {}", mean(&q));
    }

    #[test]
    fn corrects_a_gain_imbalance_between_branches() {
        let mut c = IqConverter::new().with_adapt_rate(0.5);

        // Q deliberately given half the amplitude of I.
        let n = 2048;
        let bytes: Vec<u8> = (0..n)
            .flat_map(|k| {
                let ph = TAU * 0.013 * k as f32;
                let i = 127.5 + 60.0 * ph.cos();
                let q = 127.5 + 30.0 * ph.sin();
                [i as u8, q as u8]
            })
            .collect();

        let mut i = vec![0.0; n];
        let mut q = vec![0.0; n];
        for _ in 0..60 {
            c.process(&bytes, &mut i, &mut q);
        }

        let pi = i.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let pq = q.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let ratio = pi / pq;
        assert!(
            (0.8..1.25).contains(&ratio),
            "branch powers still unbalanced after correction: ratio {ratio}"
        );
    }

    #[test]
    fn reset_discards_learned_state() {
        let mut c = IqConverter::new();
        let bytes = vec![200u8; 512];
        let mut i = vec![0.0; 256];
        let mut q = vec![0.0; 256];
        c.process(&bytes, &mut i, &mut q);
        assert_ne!(c.corrections(), Corrections::default());

        c.reset();
        assert_eq!(c.corrections(), Corrections::default());
    }

    #[test]
    fn imbalance_correction_is_identity_when_balanced() {
        let mut i: Vec<f32> = (0..64).map(|k| (k as f32 * 0.1).cos()).collect();
        let mut q: Vec<f32> = (0..64).map(|k| (k as f32 * 0.1).sin()).collect();
        let before_q = q.clone();

        apply_imbalance(&mut i, &mut q, 1.0, 0.0);
        for (a, b) in q.iter().zip(&before_q) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn vector_and_scalar_tails_agree() {
        // Lengths that are not multiples of the vector width exercise both paths.
        for n in [1usize, 3, 4, 5, 7, 8, 17, 64, 67] {
            let mut a: Vec<f32> = (0..n).map(|k| k as f32 * 0.25).collect();
            let expected: Vec<f32> = a.iter().map(|v| v - 1.5).collect();
            apply_dc(&mut a, 1.5);
            for (got, want) in a.iter().zip(&expected) {
                assert!((got - want).abs() < 1e-6, "mismatch at n={n}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "whole number of pairs")]
    fn rejects_a_truncated_pair() {
        let mut c = IqConverter::new();
        let mut i = [0.0; 4];
        let mut q = [0.0; 4];
        c.process(&[1, 2, 3], &mut i, &mut q);
    }
}
