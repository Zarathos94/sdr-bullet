//! Frequency modulation: discriminator, noise squelch, and stereo multiplex decoding.

use super::{fast_atan2, Pll};
use crate::fir::{design_bandpass, design_lowpass, Fir};
use crate::window::Window;

/// Recovers instantaneous frequency from complex baseband.
///
/// The discriminator takes the phase of `z[n] * conj(z[n-1])`, which is the phase advanced
/// during one sample interval and therefore proportional to the instantaneous frequency.
/// Taking the difference this way rather than unwrapping absolute phase means there is no
/// accumulator to drift and no wrap to handle: the product's argument is already in the
/// principal range for any deviation below Nyquist.
#[derive(Debug, Clone)]
pub struct FmDemod {
    prev_i: f32,
    prev_q: f32,
    /// Converts radians per sample into a normalised amplitude, so full deviation is 1.0.
    scale: f32,
    squelch: Squelch,
}

impl FmDemod {
    /// `deviation_hz` is the peak deviation the mode uses: 75 kHz for broadcast, 2.5 to
    /// 5 kHz for narrowband voice.
    ///
    /// # Panics
    /// If the sample rate or deviation is not positive.
    pub fn new(sample_rate: f32, deviation_hz: f32) -> Self {
        assert!(sample_rate > 0.0, "sample rate must be positive");
        assert!(deviation_hz > 0.0, "deviation must be positive");
        Self {
            prev_i: 0.0,
            prev_q: 0.0,
            scale: sample_rate / (core::f32::consts::TAU * deviation_hz),
            squelch: Squelch::new(sample_rate),
        }
    }

    pub fn squelch_mut(&mut self) -> &mut Squelch {
        &mut self.squelch
    }

    pub fn reset(&mut self) {
        self.prev_i = 0.0;
        self.prev_q = 0.0;
        self.squelch.reset();
    }

    /// Demodulates a block, writing normalised audio into `out`.
    ///
    /// Returns whether the squelch is open. When it is closed the output is silenced, but
    /// the discriminator still runs so the noise estimate stays current.
    ///
    /// # Panics
    /// If `i` and `q` differ in length, or `out` is shorter than either.
    pub fn process(&mut self, i: &[f32], q: &[f32], out: &mut [f32]) -> bool {
        assert_eq!(i.len(), q.len(), "I and Q must match in length");
        assert!(out.len() >= i.len(), "output buffer too short");

        for k in 0..i.len() {
            let (ci, cq) = (i[k], q[k]);
            // z[k] * conj(z[k-1])
            let re = ci * self.prev_i + cq * self.prev_q;
            let im = cq * self.prev_i - ci * self.prev_q;
            out[k] = fast_atan2(im, re) * self.scale;
            self.prev_i = ci;
            self.prev_q = cq;
        }

        self.squelch.process(&mut out[..i.len()])
    }
}

/// Mutes the output when the channel carries only noise.
///
/// An FM discriminator turns a weak signal into a wideband hiss whose energy rises with
/// frequency, so the amount of energy sitting above the audio band is a direct measure of
/// how little signal there is. Measuring that is far more reliable than measuring received
/// power, which cannot tell a strong signal from a strong interferer.
#[derive(Debug, Clone)]
pub struct Squelch {
    /// Passes only the band above the audio, where noise lives and programme does not.
    noise_filter: Fir,
    scratch: Vec<f32>,
    /// Smoothed mean square of the out-of-band noise. Held as power rather than amplitude
    /// so the update is a plain leaky integrator with no square root in the loop.
    power: f32,
    smoothing: f32,
    threshold: f32,
    enabled: bool,
    open: bool,
    /// Ramp applied while opening or closing, in samples, to avoid a click.
    ramp_len: usize,
    ramp_pos: usize,
}

impl Squelch {
    pub fn new(sample_rate: f32) -> Self {
        // Above 5 kHz is clear of voice but still inside a narrowband channel.
        let cutoff = (5_000.0 / sample_rate).min(0.45);
        let hp = highpass_from_lowpass(&design_lowpass(31, cutoff, Window::Hamming));
        Self {
            noise_filter: Fir::new(hp),
            scratch: Vec::new(),
            power: 1.0,
            smoothing: 1.0 - (-1.0 / (0.02 * sample_rate)).exp(),
            threshold: 0.08,
            enabled: false,
            open: true,
            ramp_len: ((0.005 * sample_rate) as usize).max(1),
            ramp_pos: 0,
        }
    }

    /// Noise level above which the channel is considered empty. Larger values open the
    /// squelch on weaker signals.
    ///
    /// # Panics
    /// If `threshold` is not positive.
    pub fn set_threshold(&mut self, threshold: f32) {
        assert!(threshold > 0.0, "threshold must be positive");
        self.threshold = threshold;
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.open = true;
            self.ramp_pos = self.ramp_len;
        }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Current smoothed noise measurement, for a signal-strength display.
    pub fn noise_level(&self) -> f32 {
        self.power.sqrt()
    }

    pub fn reset(&mut self) {
        self.noise_filter.reset();
        self.power = 1.0;
        self.open = !self.enabled;
        self.ramp_pos = if self.open { self.ramp_len } else { 0 };
    }

    /// Measures noise on `audio` and gates it. Returns whether the gate is open.
    ///
    /// The estimate is integrated per sample rather than per block. Deriving a
    /// per-sample coefficient and then applying it once per call would make the time
    /// constant depend on whatever block size the caller happened to use — with the
    /// sizes this pipeline actually uses, the level would move by a fraction of a
    /// percent per call and the gate would never respond at all.
    fn process(&mut self, audio: &mut [f32]) -> bool {
        self.scratch.resize(audio.len(), 0.0);
        self.noise_filter.process(audio, &mut self.scratch);

        for (sample, noise) in audio.iter_mut().zip(&self.scratch) {
            self.power += (noise * noise - self.power) * self.smoothing;

            if !self.enabled {
                continue;
            }

            let level = self.power.sqrt();
            // Hysteresis, so a signal hovering at the threshold does not chatter the gate.
            if level < self.threshold * 0.8 {
                self.open = true;
            } else if level > self.threshold * 1.25 {
                self.open = false;
            }

            // Cross-fade rather than switching abruptly; a hard mute is an audible click.
            if self.open && self.ramp_pos < self.ramp_len {
                self.ramp_pos += 1;
            } else if !self.open && self.ramp_pos > 0 {
                self.ramp_pos -= 1;
            }
            *sample *= self.ramp_pos as f32 / self.ramp_len as f32;
        }

        if !self.enabled {
            self.open = true;
        }
        self.open
    }
}

/// Turns a low-pass design into the complementary high-pass by spectral inversion.
///
/// Negating every tap and adding one at the centre subtracts the low-pass response from an
/// all-pass, which is exactly the high-pass. Only valid for an odd, linear-phase design.
fn highpass_from_lowpass(lowpass: &[f32]) -> Vec<f32> {
    assert!(
        lowpass.len() % 2 == 1,
        "spectral inversion needs an odd tap count"
    );
    let centre = lowpass.len() / 2;
    let mut h: Vec<f32> = lowpass.iter().map(|v| -v).collect();
    h[centre] += 1.0;
    h
}

