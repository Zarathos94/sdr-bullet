//! Numerically controlled oscillator and complex mixer.
//!
//! Phase is held in a `u32` accumulator, so it wraps exactly at a full cycle and never
//! accumulates the drift that a recursive rotator would. The table is looked up with
//! linear interpolation between neighbouring entries, which puts spurious products around
//! -90 dBc — below the dynamic range of an 8-bit ADC, so the oscillator is not the limit.

use crate::simd::{F32x4, LANES};

/// Table entries per cycle. A power of two so the index is a shift, not a divide.
const TABLE_BITS: u32 = 12;
const TABLE_LEN: usize = 1 << TABLE_BITS;
/// Bits of the accumulator left over for interpolating between entries.
const FRAC_BITS: u32 = 32 - TABLE_BITS;
const FRAC_SCALE: f32 = 1.0 / (1u64 << FRAC_BITS) as f32;

/// Shared sine table. One extra entry wraps the end so interpolation needs no branch.
struct SineTable([f32; TABLE_LEN + 1]);

impl SineTable {
    fn new() -> Self {
        let mut t = [0.0f32; TABLE_LEN + 1];
        for (k, slot) in t.iter_mut().enumerate() {
            let phase = core::f64::consts::TAU * (k as f64) / TABLE_LEN as f64;
            *slot = phase.sin() as f32;
        }
        Self(t)
    }

    #[inline(always)]
    fn sin_at(&self, phase: u32) -> f32 {
        let idx = (phase >> FRAC_BITS) as usize;
        let frac = (phase & ((1 << FRAC_BITS) - 1)) as f32 * FRAC_SCALE;
        let a = self.0[idx];
        let b = self.0[idx + 1];
        a + (b - a) * frac
    }

    #[inline(always)]
    fn cos_at(&self, phase: u32) -> f32 {
        // cos(x) = sin(x + pi/2); a quarter turn is a quarter of the accumulator's range.
        self.sin_at(phase.wrapping_add(1 << 30))
    }
}

/// Complex oscillator used to shift a channel to or from zero frequency.
pub struct Nco {
    table: SineTable,
    phase: u32,
    step: u32,
    sample_rate: f64,
    /// Held across calls so mixing allocates nothing. This runs at the capture rate, where
    /// a per-block allocation is enough to cause audible dropouts.
    cos_scratch: Vec<f32>,
    sin_scratch: Vec<f32>,
}

impl Nco {
    pub fn new(sample_rate: f64) -> Self {
        assert!(sample_rate > 0.0, "sample rate must be positive");
        Self {
            table: SineTable::new(),
            phase: 0,
            step: 0,
            sample_rate,
            cos_scratch: Vec::new(),
            sin_scratch: Vec::new(),
        }
    }

    /// Sets the oscillator frequency in hertz. Negative values rotate the other way, and
    /// frequencies beyond Nyquist alias, both of which fall out of the wrapping cast.
    pub fn set_frequency(&mut self, hz: f64) {
        let cycles_per_sample = hz / self.sample_rate;
        // Wrapping into u32 is the modulo-one-cycle reduction, and handles sign for free.
        self.step = (cycles_per_sample * 4294967296.0) as i64 as u32;
    }

    pub fn frequency(&self) -> f64 {
        (self.step as i32) as f64 / 4294967296.0 * self.sample_rate
    }

    pub fn set_phase(&mut self, radians: f32) {
        let turns = radians as f64 / core::f64::consts::TAU;
        self.phase = (turns.rem_euclid(1.0) * 4294967296.0) as u64 as u32;
    }

    pub fn reset(&mut self) {
        self.phase = 0;
    }

    /// Writes the oscillator's cosine and sine into the supplied buffers.
    ///
    /// # Panics
    /// If the buffers differ in length.
    pub fn generate(&mut self, cos_out: &mut [f32], sin_out: &mut [f32]) {
        assert_eq!(
            cos_out.len(),
            sin_out.len(),
            "quadrature buffers must match in length"
        );
        for k in 0..cos_out.len() {
            cos_out[k] = self.table.cos_at(self.phase);
            sin_out[k] = self.table.sin_at(self.phase);
            self.phase = self.phase.wrapping_add(self.step);
        }
    }

    /// Multiplies the signal by `exp(-j*2*pi*f*t)`, shifting frequency `f` down to zero.
    ///
    /// This is the channel-selection step: set the oscillator to the offset of the wanted
    /// signal from the tuner's centre and the wanted signal lands at DC.
    ///
    /// # Panics
    /// If `i` and `q` differ in length.
    pub fn mix_down(&mut self, i: &mut [f32], q: &mut [f32]) {
        assert_eq!(i.len(), q.len(), "I and Q must match in length");
        self.mix(i, q, -1.0);
    }

    /// Multiplies the signal by `exp(+j*2*pi*f*t)`, shifting zero up to frequency `f`.
    ///
    /// # Panics
    /// If `i` and `q` differ in length.
    pub fn mix_up(&mut self, i: &mut [f32], q: &mut [f32]) {
        assert_eq!(i.len(), q.len(), "I and Q must match in length");
        self.mix(i, q, 1.0);
    }

    fn mix(&mut self, i: &mut [f32], q: &mut [f32], sense: f32) {
        let n = i.len();
        // Generating a whole block of the oscillator up front keeps the table lookups —
        // which are scalar gathers — in their own loop, so the arithmetic that follows
        // stays vectorisable.
        let mut cos = core::mem::take(&mut self.cos_scratch);
        let mut sin = core::mem::take(&mut self.sin_scratch);
        cos.resize(n, 0.0);
        sin.resize(n, 0.0);

        self.generate(&mut cos, &mut sin);
        if sense < 0.0 {
            for v in sin.iter_mut() {
                *v = -*v;
            }
        }
        complex_multiply_in_place(i, q, &cos, &sin);

        self.cos_scratch = cos;
        self.sin_scratch = sin;
    }
}

impl core::fmt::Debug for Nco {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Nco")
            .field("frequency_hz", &self.frequency())
            .field("sample_rate", &self.sample_rate)
            .finish_non_exhaustive()
    }
}

/// `(i + jq) *= (c + js)`, elementwise, in place.
///
/// Working on split arrays means this is four real multiplies and two adds per sample with
/// no lane shuffling — the reason the whole crate keeps I and Q in separate slices.
pub fn complex_multiply_in_place(i: &mut [f32], q: &mut [f32], c: &[f32], s: &[f32]) {
    let n = i.len().min(q.len()).min(c.len()).min(s.len());
    let chunks = n / LANES;
    for k in 0..chunks {
        let off = k * LANES;
        // SAFETY: `off + LANES <= n`, and every slice is at least `n` long.
        unsafe {
            let vi = F32x4::load_unchecked(i, off);
            let vq = F32x4::load_unchecked(q, off);
            let vc = F32x4::load_unchecked(c, off);
            let vs = F32x4::load_unchecked(s, off);
            (vi * vc - vq * vs).store_unchecked(i, off);
            (vi * vs + vq * vc).store_unchecked(q, off);
        }
    }
    for k in chunks * LANES..n {
        let (vi, vq) = (i[k], q[k]);
        i[k] = vi * c[k] - vq * s[k];
        q[k] = vi * s[k] + vq * c[k];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    const SR: f64 = 48_000.0;

    #[test]
    fn table_reproduces_sine_accurately() {
        let t = SineTable::new();
        for k in 0..1000 {
            let turns = k as f64 / 1000.0;
            let phase = (turns * 4294967296.0) as u64 as u32;
            let expected = (turns * core::f64::consts::TAU).sin() as f32;
            assert!(
                (t.sin_at(phase) - expected).abs() < 1e-4,
                "sine mismatch at turn {turns}"
            );
        }
    }

    #[test]
    fn cosine_leads_sine_by_a_quarter_cycle() {
        let t = SineTable::new();
        for k in 0..500 {
            let phase = (k as u64 * 8_000_000) as u32;
            let expected = (phase as f64 / 4294967296.0 * core::f64::consts::TAU).cos() as f32;
            assert!((t.cos_at(phase) - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn frequency_survives_a_roundtrip() {
        let mut nco = Nco::new(SR);
        for hz in [0.0, 1.0, 1000.0, -1000.0, 23_999.0] {
            nco.set_frequency(hz);
            assert!(
                (nco.frequency() - hz).abs() < 0.001,
                "wanted {hz}, got {}",
                nco.frequency()
            );
        }
    }

    #[test]
    fn generates_a_tone_at_the_requested_rate() {
        let mut nco = Nco::new(SR);
        nco.set_frequency(1000.0);
        let n = 480;
        let mut c = vec![0.0; n];
        let mut s = vec![0.0; n];
        nco.generate(&mut c, &mut s);

        // Ten full cycles of a 1 kHz tone fit in 480 samples at 48 kHz.
        for k in 0..n {
            let expected = (TAU * 1000.0 * k as f32 / SR as f32).cos();
            assert!((c[k] - expected).abs() < 1e-3, "cos mismatch at {k}");
        }
    }

    #[test]
    fn quadrature_outputs_stay_on_the_unit_circle() {
        let mut nco = Nco::new(SR);
        nco.set_frequency(7333.0);
        let mut c = vec![0.0; 4096];
        let mut s = vec![0.0; 4096];
        nco.generate(&mut c, &mut s);
        for k in 0..4096 {
            let mag = (c[k] * c[k] + s[k] * s[k]).sqrt();
            assert!((mag - 1.0).abs() < 1e-3, "magnitude {mag} at {k}");
        }
    }

    #[test]
    fn mixing_down_by_the_signal_frequency_lands_it_at_dc() {
        let n = 4096;
        let tone = 5000.0f32;

        let mut i: Vec<f32> = (0..n)
            .map(|k| (TAU * tone * k as f32 / SR as f32).cos())
            .collect();
        let mut q: Vec<f32> = (0..n)
            .map(|k| (TAU * tone * k as f32 / SR as f32).sin())
            .collect();

        let mut nco = Nco::new(SR);
        nco.set_frequency(tone as f64);
        nco.mix_down(&mut i, &mut q);

        // A complex exponential mixed down by its own frequency becomes a constant.
        for k in 0..n {
            assert!((i[k] - 1.0).abs() < 5e-3, "I not flat at {k}: {}", i[k]);
            assert!(q[k].abs() < 5e-3, "Q not zero at {k}: {}", q[k]);
        }
    }

    #[test]
    fn mixing_up_then_down_restores_the_input() {
        let n = 1024;
        let orig_i: Vec<f32> = (0..n).map(|k| (k as f32 * 0.05).cos()).collect();
        let orig_q: Vec<f32> = (0..n).map(|k| (k as f32 * 0.05).sin()).collect();
        let mut i = orig_i.clone();
        let mut q = orig_q.clone();

        let mut up = Nco::new(SR);
        up.set_frequency(3000.0);
        up.mix_up(&mut i, &mut q);

        let mut down = Nco::new(SR);
        down.set_frequency(3000.0);
        down.mix_down(&mut i, &mut q);

        for k in 0..n {
            assert!((i[k] - orig_i[k]).abs() < 1e-3, "I drift at {k}");
            assert!((q[k] - orig_q[k]).abs() < 1e-3, "Q drift at {k}");
        }
    }

    #[test]
    fn negative_frequency_rotates_the_opposite_way() {
        let mut a = Nco::new(SR);
        a.set_frequency(1000.0);
        let mut b = Nco::new(SR);
        b.set_frequency(-1000.0);

        let (mut ca, mut sa) = (vec![0.0; 64], vec![0.0; 64]);
        let (mut cb, mut sb) = (vec![0.0; 64], vec![0.0; 64]);
        a.generate(&mut ca, &mut sa);
        b.generate(&mut cb, &mut sb);

        for k in 0..64 {
            assert!((ca[k] - cb[k]).abs() < 1e-4, "cosine should be even");
            assert!((sa[k] + sb[k]).abs() < 1e-4, "sine should be odd");
        }
    }

    #[test]
    fn phase_accumulator_wraps_without_drifting() {
        // Half a million samples is far past where a recursive rotator would visibly decay.
        let mut nco = Nco::new(SR);
        nco.set_frequency(11_000.0);
        let mut c = vec![0.0; 8192];
        let mut s = vec![0.0; 8192];
        for _ in 0..64 {
            nco.generate(&mut c, &mut s);
        }
        for k in 0..8192 {
            let mag = (c[k] * c[k] + s[k] * s[k]).sqrt();
            assert!((mag - 1.0).abs() < 1e-3, "amplitude drifted to {mag}");
        }
    }

    #[test]
    fn complex_multiply_handles_lengths_that_straddle_the_vector_width() {
        for n in [1usize, 3, 4, 7, 8, 13, 64] {
            let mut i: Vec<f32> = (0..n).map(|k| k as f32).collect();
            let mut q: Vec<f32> = (0..n).map(|k| -(k as f32)).collect();
            let c = vec![0.0f32; n];
            let s = vec![1.0f32; n];

            let want_i: Vec<f32> = (0..n).map(|k| k as f32).collect();
            complex_multiply_in_place(&mut i, &mut q, &c, &s);

            // Multiplying by j maps (a + jb) to (-b + ja).
            for k in 0..n {
                assert!((i[k] - want_i[k]).abs() < 1e-6, "n={n} k={k}");
                assert!((q[k] - k as f32).abs() < 1e-6, "n={n} k={k}");
            }
        }
    }
}
