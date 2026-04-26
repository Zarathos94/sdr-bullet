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

