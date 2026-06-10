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

        let amp = tone_amplitude(&out[4800..], 1000.0);
        assert!(
            (amp - 0.5).abs() < 0.03,
            "recovered {amp}, expected 0.5 modulation depth"
        );
    }

    #[test]
    fn removes_the_carrier_offset() {
        let n = 48_000;
        let message: Vec<f32> = (0..n)
            .map(|k| (TAU * 1000.0 * k as f32 / SR).sin())
            .collect();
        let (i, q) = modulate(&message, 0.3);

        let mut demod = AmDemod::new(SR);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);

        let tail = &out[24_000..];
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(
            mean.abs() < 0.01,
            "residual carrier of {mean} left in the output"
        );
        assert!(
            (demod.carrier_level() - 1.0).abs() < 0.05,
            "carrier tracked to {}",
            demod.carrier_level()
        );
    }

    #[test]
    fn depth_scales_the_output() {
        let n = 24_000;
        let message: Vec<f32> = (0..n)
            .map(|k| (TAU * 900.0 * k as f32 / SR).sin())
            .collect();

        let measure = |depth: f32| {
            let (i, q) = modulate(&message, depth);
            let mut demod = AmDemod::new(SR);
            let mut out = vec![0.0; n];
            demod.process(&i, &q, &mut out);
            tone_amplitude(&out[4800..], 900.0)
        };

        let quarter = measure(0.25);
        let half = measure(0.5);
        assert!(
            (half / quarter - 2.0).abs() < 0.1,
            "expected a 2:1 ratio, got {}",
            half / quarter
        );
    }

    #[test]
    fn an_unmodulated_carrier_produces_silence() {
        let n = 24_000;
        let (i, q) = modulate(&vec![0.0; n], 0.5);
        let mut demod = AmDemod::new(SR);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);
        let tail = &out[4800..];
        let rms = (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
        assert!(rms < 1e-4, "steady carrier produced {rms} of output");
    }

    #[test]
    fn is_insensitive_to_carrier_phase() {
        // Envelope detection should not care where the carrier sits in phase, which is
        // the whole reason to use it over a coherent detector.
        let n = 24_000;
        let message: Vec<f32> = (0..n)
            .map(|k| (TAU * 1000.0 * k as f32 / SR).sin())
            .collect();

        let amp_at = |phase: f32| {
            let i: Vec<f32> = message
                .iter()
                .map(|m| (1.0 + 0.5 * m) * phase.cos())
                .collect();
            let q: Vec<f32> = message
                .iter()
                .map(|m| (1.0 + 0.5 * m) * phase.sin())
                .collect();
            let mut demod = AmDemod::new(SR);
            let mut out = vec![0.0; n];
            demod.process(&i, &q, &mut out);
            tone_amplitude(&out[4800..], 1000.0)
        };

        let a = amp_at(0.0);
        let b = amp_at(1.0);
        assert!((a - b).abs() < 0.01, "phase changed the result: {a} vs {b}");
    }

    #[test]
    fn vector_and_scalar_tails_agree() {
        for n in [1usize, 3, 5, 7, 9, 33] {
            let i: Vec<f32> = (0..n).map(|k| 1.0 + 0.1 * k as f32).collect();
            let q: Vec<f32> = (0..n).map(|k| 0.05 * k as f32).collect();
            let mut demod = AmDemod::new(SR);
            let mut out = vec![0.0; n];
            demod.process(&i, &q, &mut out);
            for v in &out {
                assert!(v.is_finite(), "non-finite output at n={n}");
            }
        }
    }

    #[test]
    fn reset_clears_the_carrier_estimate() {
        let mut demod = AmDemod::new(SR);
        let i = vec![5.0f32; 1000];
        let q = vec![0.0f32; 1000];
        let mut out = vec![0.0; 1000];
        demod.process(&i, &q, &mut out);
        assert!(demod.carrier_level() > 0.0);
        demod.reset();
        assert_eq!(demod.carrier_level(), 0.0);
    }
}
