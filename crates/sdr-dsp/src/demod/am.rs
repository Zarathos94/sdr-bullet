//! Amplitude modulation.
//!
//! Envelope detection: the magnitude of the complex baseband is the modulation envelope,
//! carrier included. The carrier shows up as a constant offset, so it is removed with a
//! slow high-pass that tracks it — slow enough that programme material well below the
//! lowest audio frequency of interest is not also removed.

use crate::simd::{F32x4, LANES};

/// Envelope-detecting AM demodulator.
#[derive(Debug, Clone)]
pub struct AmDemod {
    /// Tracks the carrier level so it can be subtracted out.
    carrier: f32,
    carrier_smoothing: f32,
    primed: bool,
}

impl AmDemod {
    /// # Panics
    /// If `sample_rate` is not positive.
    pub fn new(sample_rate: f32) -> Self {
        assert!(sample_rate > 0.0, "sample rate must be positive");
        Self {
            carrier: 0.0,
            // A 20 Hz corner sits below the audio band but tracks fading fast enough to
            // keep the output centred as the signal comes and goes.
            carrier_smoothing: 1.0 - (-core::f32::consts::TAU * 20.0 / sample_rate).exp(),
            primed: false,
        }
    }

    /// Current carrier level, which doubles as a signal-strength reading.
    pub fn carrier_level(&self) -> f32 {
        self.carrier
    }

    pub fn reset(&mut self) {
        self.carrier = 0.0;
        self.primed = false;
    }

    /// Demodulates a block into `out`.
    ///
    /// # Panics
    /// If `i` and `q` differ in length, or `out` is shorter than either.
    pub fn process(&mut self, i: &[f32], q: &[f32], out: &mut [f32]) {
        assert_eq!(i.len(), q.len(), "I and Q must match in length");
        assert!(out.len() >= i.len(), "output buffer too short");
        let n = i.len();
        if n == 0 {
            return;
        }

        // Magnitude first, vectorised, then the carrier tracker runs over the result.
        // Splitting it this way keeps the recursive part out of the vector loop.
        let chunks = n / LANES;
        for c in 0..chunks {
            let off = c * LANES;
            // SAFETY: `off + LANES <= n`, and all three buffers are at least `n` long.
            unsafe {
                let vi = F32x4::load_unchecked(i, off);
                let vq = F32x4::load_unchecked(q, off);
                (vi * vi + vq * vq).sqrt().store_unchecked(out, off);
            }
        }
        for k in chunks * LANES..n {
            out[k] = (i[k] * i[k] + q[k] * q[k]).sqrt();
        }

        if !self.primed {
            // Starting the tracker at zero would let the full carrier through as a thump
            // before the filter settles.
            self.carrier = out[0];
            self.primed = true;
        }

        for sample in out[..n].iter_mut() {
            self.carrier += (*sample - self.carrier) * self.carrier_smoothing;
            *sample -= self.carrier;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    const SR: f32 = 48_000.0;

    /// Amplitude-modulates a carrier at complex baseband, with the carrier at zero offset.
    fn modulate(message: &[f32], depth: f32) -> (Vec<f32>, Vec<f32>) {
        let i: Vec<f32> = message.iter().map(|m| 1.0 + depth * m).collect();
        let q = vec![0.0; message.len()];
        (i, q)
    }

    fn tone_amplitude(x: &[f32], freq: f32) -> f32 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (k, v) in x.iter().enumerate() {
            let ang = TAU * freq * k as f32 / SR;
            re += (*v * ang.cos()) as f64;
            im += (*v * ang.sin()) as f64;
        }
        2.0 * ((re * re + im * im).sqrt() / x.len() as f64) as f32
    }

    #[test]
    fn recovers_the_modulating_tone() {
        let n = 24_000;
        let message: Vec<f32> = (0..n)
            .map(|k| (TAU * 1000.0 * k as f32 / SR).sin())
            .collect();
        let (i, q) = modulate(&message, 0.5);

        let mut demod = AmDemod::new(SR);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);

