//! Single sideband, by Weaver's third method.
//!
//! With the receiver tuned to the suppressed carrier, the upper sideband occupies positive
//! baseband frequencies and the lower sideband occupies negative ones. Selecting one means
//! building a filter that is asymmetric about zero — which a real-valued filter cannot be.
//!
//! Weaver's method sidesteps that. Shift the wanted sideband so it straddles zero, and an
//! ordinary symmetric low-pass now covers exactly it while the unwanted sideband lands
//! outside; shift back and take the real part. Two real low-pass filters replace the
//! wideband Hilbert transformer the phasing method would need, and the sideband rejection
//! depends on the filters rather than on how flat a 90-degree phase shift stays across the
//! band.

use crate::fir::{design_lowpass, Fir};
use crate::nco::Nco;
use crate::window::Window;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sideband {
    Upper,
    Lower,
}

/// Weaver-method SSB demodulator.
///
/// For CW, use [`Sideband::Upper`] with a narrow bandwidth and tune the receiver off the
/// carrier by the pitch you want to hear — which is what the beat frequency oscillator in
/// a conventional receiver is doing.
#[derive(Debug)]
pub struct SsbDemod {
    sideband: Sideband,
    /// Shifts the wanted sideband to straddle zero.
    shift: Nco,
    /// Shifts it back. Same frequency, but started a filter delay behind.
    recombine: Nco,
    lowpass_i: Fir,
    lowpass_q: Fir,
    work_i: Vec<f32>,
    work_q: Vec<f32>,
    filtered_i: Vec<f32>,
    filtered_q: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl SsbDemod {
    /// `bandwidth_hz` is the full audio width, so 2700 for voice and a few hundred for CW.
    ///
    /// # Panics
    /// If the sample rate is not positive, or the bandwidth does not fit below Nyquist.
    pub fn new(sample_rate: f32, sideband: Sideband, bandwidth_hz: f32) -> Self {
        assert!(sample_rate > 0.0, "sample rate must be positive");
        assert!(
            bandwidth_hz > 0.0 && bandwidth_hz < sample_rate / 2.0,
            "bandwidth must fit below Nyquist"
        );

        let half = bandwidth_hz / 2.0;
        // Enough taps that the skirt is steep relative to the passband; a narrow CW filter
        // needs proportionally more of them than a voice filter.
        let taps = 129;
        let cutoff = half / sample_rate;
        let proto = design_lowpass(taps, cutoff, Window::kaiser(8.6));
        let delay = (taps - 1) / 2;

        let mut shift = Nco::new(sample_rate as f64);
        shift.set_frequency(half as f64);

        let mut recombine = Nco::new(sample_rate as f64);
        recombine.set_frequency(half as f64);
        // The low-pass delays the signal, so the sample leaving the filter now was mixed
        // down a `delay` ago. Starting the recombining oscillator that far back in phase
        // undoes the original shift exactly. Without it the two oscillators disagree by a
        // fixed angle, which survives into the audio as a constant phase rotation.
        recombine.set_phase(-core::f32::consts::TAU * half * delay as f32 / sample_rate);

        Self {
            sideband,
            shift,
            recombine,
            lowpass_i: Fir::new(proto.clone()),
            lowpass_q: Fir::new(proto),
            work_i: Vec::new(),
            work_q: Vec::new(),
            filtered_i: Vec::new(),
            filtered_q: Vec::new(),
            cos: Vec::new(),
            sin: Vec::new(),
        }
    }

    pub fn sideband(&self) -> Sideband {
        self.sideband
    }

    pub fn reset(&mut self) {
        self.shift.reset();
        self.recombine.reset();
        self.lowpass_i.reset();
        self.lowpass_q.reset();
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

        self.work_i.clear();
        self.work_i.extend_from_slice(i);
        self.work_q.clear();
        self.work_q.extend_from_slice(q);

        // Move the wanted sideband down across zero. Upper sits above the carrier so it
        // shifts down; lower sits below so it shifts up.
        match self.sideband {
            Sideband::Upper => self.shift.mix_down(&mut self.work_i, &mut self.work_q),
            Sideband::Lower => self.shift.mix_up(&mut self.work_i, &mut self.work_q),
        }

        // The unwanted sideband is now a full bandwidth away and falls outside this.
        self.filtered_i.resize(n, 0.0);
        self.filtered_q.resize(n, 0.0);
        self.lowpass_i.process(&self.work_i, &mut self.filtered_i);
        self.lowpass_q.process(&self.work_q, &mut self.filtered_q);

        self.cos.resize(n, 0.0);
        self.sin.resize(n, 0.0);
        self.recombine.generate(&mut self.cos, &mut self.sin);

        // Shift back and take the real part in one step. Re{z * e^{j0}} for the upper
        // sideband, and the conjugate rotation for the lower.
        match self.sideband {
            Sideband::Upper => {
                for k in 0..n {
                    out[k] = self.filtered_i[k] * self.cos[k] - self.filtered_q[k] * self.sin[k];
                }
            }
            Sideband::Lower => {
                for k in 0..n {
                    out[k] = self.filtered_i[k] * self.cos[k] + self.filtered_q[k] * self.sin[k];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    const SR: f32 = 48_000.0;

    /// A single complex exponential. Positive frequencies are upper sideband content,
    /// negative frequencies lower.
    fn exponential(freq: f32, n: usize) -> (Vec<f32>, Vec<f32>) {
        let i = (0..n).map(|k| (TAU * freq * k as f32 / SR).cos()).collect();
        let q = (0..n).map(|k| (TAU * freq * k as f32 / SR).sin()).collect();
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

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn upper_sideband_recovers_a_positive_frequency_tone() {
        let n = 48_000;
        let (i, q) = exponential(1000.0, n);
        let mut demod = SsbDemod::new(SR, Sideband::Upper, 2700.0);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);

        let amp = tone_amplitude(&out[2000..], 1000.0);
        assert!(
            (amp - 1.0).abs() < 0.1,
            "recovered amplitude {amp}, expected 1.0"
        );
    }

    #[test]
    fn upper_sideband_rejects_a_negative_frequency_tone() {
        let n = 48_000;
        let (i, q) = exponential(-1000.0, n);
        let mut demod = SsbDemod::new(SR, Sideband::Upper, 2700.0);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);

        let residual = rms(&out[2000..]);
        assert!(
            residual < 0.02,
            "opposite sideband leaked through at {residual}"
        );
    }

    #[test]
    fn lower_sideband_recovers_a_negative_frequency_tone() {
        let n = 48_000;
        let (i, q) = exponential(-1000.0, n);
        let mut demod = SsbDemod::new(SR, Sideband::Lower, 2700.0);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);

        let amp = tone_amplitude(&out[2000..], 1000.0);
        assert!(
            (amp - 1.0).abs() < 0.1,
            "recovered amplitude {amp}, expected 1.0"
        );
    }

    #[test]
    fn lower_sideband_rejects_a_positive_frequency_tone() {
        let n = 48_000;
        let (i, q) = exponential(1000.0, n);
        let mut demod = SsbDemod::new(SR, Sideband::Lower, 2700.0);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);

        let residual = rms(&out[2000..]);
        assert!(
            residual < 0.02,
            "opposite sideband leaked through at {residual}"
        );
    }

    #[test]
    fn sideband_rejection_is_at_least_forty_decibels() {
        let n = 48_000;
        let measure = |sb: Sideband, freq: f32| {
            let (i, q) = exponential(freq, n);
            let mut demod = SsbDemod::new(SR, sb, 2700.0);
            let mut out = vec![0.0; n];
            demod.process(&i, &q, &mut out);
            rms(&out[4000..])
        };

        let wanted = measure(Sideband::Upper, 1200.0);
        let unwanted = measure(Sideband::Upper, -1200.0);
        let rejection = 20.0 * (wanted / unwanted.max(1e-12)).log10();
        assert!(
            rejection > 40.0,
            "only {rejection:.1} dB of sideband rejection"
        );
    }

    #[test]
    fn rejects_content_beyond_the_passband() {
        // 5 kHz is well outside a 2.7 kHz voice filter and must not appear in the audio.
        let n = 48_000;
        let (i, q) = exponential(5000.0, n);
        let mut demod = SsbDemod::new(SR, Sideband::Upper, 2700.0);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);
        assert!(rms(&out[4000..]) < 0.02, "out-of-band signal got through");
    }

    #[test]
    fn passes_multiple_tones_within_the_band() {
        let n = 48_000;
        let mut i = vec![0.0f32; n];
        let mut q = vec![0.0f32; n];
        for freq in [500.0f32, 1200.0, 2000.0] {
            for k in 0..n {
                i[k] += (TAU * freq * k as f32 / SR).cos() / 3.0;
                q[k] += (TAU * freq * k as f32 / SR).sin() / 3.0;
            }
        }

        let mut demod = SsbDemod::new(SR, Sideband::Upper, 2700.0);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);

        for freq in [500.0f32, 1200.0, 2000.0] {
            let amp = tone_amplitude(&out[4000..], freq);
            assert!(
                (amp - 1.0 / 3.0).abs() < 0.06,
                "tone at {freq} came out at {amp}"
            );
        }
    }

    #[test]
    fn narrow_bandwidth_works_for_cw() {
        // A carrier tuned 700 Hz off produces a 700 Hz beat note through a 500 Hz filter
        // centred at the same offset.
        let n = 48_000;
        let (i, q) = exponential(700.0, n);
        let mut demod = SsbDemod::new(SR, Sideband::Upper, 1400.0);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);

        let amp = tone_amplitude(&out[4000..], 700.0);
        assert!(amp > 0.8, "CW beat note came out at {amp}");
    }

    #[test]
    fn state_carries_across_blocks() {
        let n = 24_000;
        let (i, q) = exponential(1000.0, n);

        let mut whole = vec![0.0; n];
        SsbDemod::new(SR, Sideband::Upper, 2700.0).process(&i, &q, &mut whole);

        let mut split = vec![0.0; n];
        let mut d = SsbDemod::new(SR, Sideband::Upper, 2700.0);
        d.process(&i[..5000], &q[..5000], &mut split[..5000]);
        d.process(&i[5000..], &q[5000..], &mut split[5000..]);

        for k in 0..n {
            assert!((whole[k] - split[k]).abs() < 1e-4, "discontinuity at {k}");
        }
    }

    #[test]
    #[should_panic(expected = "bandwidth must fit below Nyquist")]
    fn rejects_a_bandwidth_beyond_nyquist() {
        SsbDemod::new(48_000.0, Sideband::Upper, 50_000.0);
    }
}
