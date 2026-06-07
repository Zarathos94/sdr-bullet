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

/// Separates a broadcast FM multiplex into left and right channels.
///
/// The multiplex carries the sum channel at baseband, the difference channel on a
/// suppressed-carrier subcarrier at 38 kHz, and a 19 kHz pilot. Because the subcarrier is
/// suppressed, its phase can only come from doubling the pilot — which is why this holds a
/// [`Pll`] rather than a free-running oscillator. A few degrees of phase error here shows
/// up directly as one channel leaking into the other.
#[derive(Debug)]
pub struct StereoDecoder {
    pilot_filter: Fir,
    sum_filter: Fir,
    difference_filter: Fir,
    pll: Pll,
    pilot: Vec<f32>,
    /// Phase the loop recovered for each sample of the last block.
    ///
    /// Kept because the data subcarrier sits at three times this phase, and being
    /// suppressed-carrier it has no other reference. Handing the phase out rather than
    /// having the data decoder run its own loop means the two cannot disagree.
    phase: Vec<f32>,
    difference: Vec<f32>,
    difference_filtered: Vec<f32>,
    sum: Vec<f32>,
    /// Pilot amplitude above which stereo is considered present.
    lock_threshold: f32,
    forced_mono: bool,
}

impl StereoDecoder {
    /// # Panics
    /// If the sample rate cannot carry the 38 kHz subcarrier.
    pub fn new(sample_rate: f32) -> Self {
        assert!(
            sample_rate > 90_000.0,
            "stereo needs headroom above the 38 kHz subcarrier, got {sample_rate}"
        );

        let norm = |hz: f32| hz / sample_rate;
        Self {
            // Narrow enough to reject the programme either side of the pilot.
            pilot_filter: Fir::new(design_bandpass(
                127,
                norm(18_500.0),
                norm(19_500.0),
                Window::kaiser(8.6),
            )),
            sum_filter: Fir::new(design_lowpass(127, norm(15_000.0), Window::kaiser(8.6))),
            difference_filter: Fir::new(design_lowpass(127, norm(15_000.0), Window::kaiser(8.6))),
            pll: Pll::new(sample_rate, 19_000.0, 100.0, 20.0),
            pilot: Vec::new(),
            phase: Vec::new(),
            difference: Vec::new(),
            difference_filtered: Vec::new(),
            sum: Vec::new(),
            lock_threshold: 0.003,
            forced_mono: false,
        }
    }

    /// Forces mono output even when a pilot is present.
    pub fn set_forced_mono(&mut self, mono: bool) {
        self.forced_mono = mono;
    }

    /// Pilot amplitude above which the decoder switches to stereo.
    ///
    /// # Panics
    /// If `threshold` is not positive.
    pub fn set_lock_threshold(&mut self, threshold: f32) {
        assert!(threshold > 0.0, "threshold must be positive");
        self.lock_threshold = threshold;
    }

    pub fn pilot_level(&self) -> f32 {
        self.pll.lock_level().abs()
    }

    /// Pilot phase recovered for each sample of the last block processed.
    ///
    /// The data subcarrier at 57 kHz is three times this. Feed it to
    /// [`super::RdsDecoder::process`] alongside the same multiplex.
    pub fn pilot_phase(&self) -> &[f32] {
        &self.phase
    }

    /// Whether the pilot loop has acquired, which is the precondition for both stereo and
    /// data decoding.
    pub fn is_locked(&self) -> bool {
        self.pll.is_locked(self.lock_threshold)
    }

    pub fn reset(&mut self) {
        self.pilot_filter.reset();
        self.sum_filter.reset();
        self.difference_filter.reset();
        self.pll.reset();
    }

    /// Decodes `mpx` into `left` and `right`. Returns true if stereo was recovered.
    ///
    /// With no pilot, or with stereo forced off, both outputs receive the sum channel and
    /// the result is mono.
    ///
    /// # Panics
    /// If either output is shorter than the input.
    pub fn process(&mut self, mpx: &[f32], left: &mut [f32], right: &mut [f32]) -> bool {
        let n = mpx.len();
        assert!(
            left.len() >= n && right.len() >= n,
            "output buffers too short"
        );

        self.pilot.resize(n, 0.0);
        self.phase.resize(n, 0.0);
        self.difference.resize(n, 0.0);
        self.difference_filtered.resize(n, 0.0);
        self.sum.resize(n, 0.0);

        self.pilot_filter.process(mpx, &mut self.pilot);

        // Track the pilot and, at each sample, mix the multiplex down by twice the
        // recovered phase. Doubling the phase is what turns the 19 kHz reference into a
        // coherent 38 kHz carrier.
        for k in 0..n {
            let phase = self.pll.advance(self.pilot[k]);
            self.phase[k] = phase;
            // The factor of two restores the level that double-sideband mixing halves.
            self.difference[k] = mpx[k] * (2.0 * phase).cos() * 2.0;
        }

        self.sum_filter.process(mpx, &mut self.sum);
        self.difference_filter
            .process(&self.difference, &mut self.difference_filtered);

        let stereo = !self.forced_mono && self.pll.is_locked(self.lock_threshold);

        if stereo {
            for k in 0..n {
                // sum is L+R and difference is L-R, so halving their sum and difference
                // recovers the individual channels.
                left[k] = (self.sum[k] + self.difference_filtered[k]) * 0.5;
                right[k] = (self.sum[k] - self.difference_filtered[k]) * 0.5;
            }
        } else {
            left[..n].copy_from_slice(&self.sum[..n]);
            right[..n].copy_from_slice(&self.sum[..n]);
        }

        stereo
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    const WFM_RATE: f32 = 240_000.0;
    const NFM_RATE: f32 = 48_000.0;

    /// Builds complex baseband frequency-modulated by `message`.
    fn modulate(message: &[f32], deviation: f32, rate: f32) -> (Vec<f32>, Vec<f32>) {
        let mut phase = 0.0f32;
        let mut i = Vec::with_capacity(message.len());
        let mut q = Vec::with_capacity(message.len());
        for m in message {
            phase += TAU * deviation * m / rate;
            i.push(phase.cos());
            q.push(phase.sin());
        }
        (i, q)
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    /// Correlation of a block against a tone, used to measure how much of that tone it holds.
    fn tone_amplitude(x: &[f32], freq: f32, rate: f32) -> f32 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (k, v) in x.iter().enumerate() {
            let ang = TAU * freq * k as f32 / rate;
            re += (*v * ang.cos()) as f64;
            im += (*v * ang.sin()) as f64;
        }
        2.0 * ((re * re + im * im).sqrt() / x.len() as f64) as f32
    }

    #[test]
    fn recovers_a_tone_from_a_modulated_carrier() {
        let n = 24_000;
        let message: Vec<f32> = (0..n)
            .map(|k| (TAU * 1000.0 * k as f32 / NFM_RATE).sin())
            .collect();
        let (i, q) = modulate(&message, 5_000.0, NFM_RATE);

        let mut demod = FmDemod::new(NFM_RATE, 5_000.0);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);

        // Skip the first sample, where the discriminator has no previous sample to use.
        let amp = tone_amplitude(&out[16..], 1000.0, NFM_RATE);
        assert!(
            (amp - 1.0).abs() < 0.05,
            "recovered amplitude {amp}, expected 1.0"
        );
    }

    #[test]
    fn output_scales_with_deviation() {
        let n = 12_000;
        let message: Vec<f32> = (0..n)
            .map(|k| (TAU * 800.0 * k as f32 / NFM_RATE).sin())
            .collect();

        // Modulate at half the deviation the demodulator expects; output should halve.
        let (i, q) = modulate(&message, 2_500.0, NFM_RATE);
        let mut demod = FmDemod::new(NFM_RATE, 5_000.0);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);

        let amp = tone_amplitude(&out[16..], 800.0, NFM_RATE);
        assert!(
            (amp - 0.5).abs() < 0.05,
            "expected half-scale output, got {amp}"
        );
    }

    #[test]
    fn an_unmodulated_carrier_demodulates_to_silence() {
        let n = 4096;
        let (i, q) = modulate(&vec![0.0; n], 5_000.0, NFM_RATE);
        let mut demod = FmDemod::new(NFM_RATE, 5_000.0);
        let mut out = vec![0.0; n];
        demod.process(&i, &q, &mut out);
        assert!(
            rms(&out[16..]) < 1e-4,
            "carrier produced output: {}",
            rms(&out[16..])
        );
    }

    #[test]
    fn state_carries_across_blocks() {
        let n = 4800;
        let message: Vec<f32> = (0..n)
            .map(|k| (TAU * 500.0 * k as f32 / NFM_RATE).sin())
            .collect();
        let (i, q) = modulate(&message, 5_000.0, NFM_RATE);

        let mut whole = vec![0.0; n];
        FmDemod::new(NFM_RATE, 5_000.0).process(&i, &q, &mut whole);

        let mut split = vec![0.0; n];
        let mut d = FmDemod::new(NFM_RATE, 5_000.0);
        d.process(&i[..1000], &q[..1000], &mut split[..1000]);
        d.process(&i[1000..], &q[1000..], &mut split[1000..]);

        for k in 0..n {
            assert!((whole[k] - split[k]).abs() < 1e-5, "discontinuity at {k}");
        }
    }

    #[test]
    fn spectral_inversion_produces_a_complementary_highpass() {
        use crate::fir::response_at;
        let lp = design_lowpass(31, 0.1, Window::Hamming);
        let hp = highpass_from_lowpass(&lp);
        // The two responses should sum to unity at every frequency.
        for f in [0.0f32, 0.05, 0.1, 0.2, 0.4] {
            let sum = response_at(&lp, f) + response_at(&hp, f);
            assert!((sum - 1.0).abs() < 0.06, "at {f}: {sum}");
        }
        assert!(response_at(&hp, 0.0) < 0.01, "high-pass should block DC");
        assert!(
            response_at(&hp, 0.4) > 0.9,
            "high-pass should pass the top of the band"
        );
    }

    #[test]
    fn squelch_opens_on_signal_and_closes_on_noise() {
        let n = 48_000;

        let message: Vec<f32> = (0..n)
            .map(|k| (TAU * 1000.0 * k as f32 / NFM_RATE).sin())
            .collect();
        let (i, q) = modulate(&message, 5_000.0, NFM_RATE);
        let mut demod = FmDemod::new(NFM_RATE, 5_000.0);
        demod.squelch_mut().set_enabled(true);
        let mut out = vec![0.0; n];
        let open = demod.process(&i, &q, &mut out);
        assert!(open, "squelch stayed shut on a clean signal");
        assert!(rms(&out[8000..]) > 0.1, "audio was gated away");

        // Random phase is what an empty channel looks like after the discriminator.
        let mut state = 0xBEEF_1234u32;
        let mut ni = Vec::with_capacity(n);
        let mut nq = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let ph = (state as f32 / u32::MAX as f32) * TAU;
            ni.push(ph.cos());
            nq.push(ph.sin());
        }
        let mut noisy = FmDemod::new(NFM_RATE, 5_000.0);
        noisy.squelch_mut().set_enabled(true);
        let mut nout = vec![0.0; n];
        let open = noisy.process(&ni, &nq, &mut nout);
        assert!(!open, "squelch stayed open on noise");
        assert!(
            rms(&nout[24_000..]) < 0.05,
            "noise leaked through a closed squelch"
        );
    }

    #[test]
    fn disabled_squelch_never_gates() {
        let n = 8192;
        let mut demod = FmDemod::new(NFM_RATE, 5_000.0);
        demod.squelch_mut().set_enabled(false);
        let (i, q) = modulate(&vec![0.0; n], 5_000.0, NFM_RATE);
        let mut out = vec![0.0; n];
        assert!(demod.process(&i, &q, &mut out));
    }

    /// Builds a stereo multiplex from separate left and right channels.
    fn build_mpx(left: &[f32], right: &[f32], pilot_amp: f32, rate: f32) -> Vec<f32> {
        (0..left.len())
            .map(|k| {
                let t = k as f32 / rate;
                let sum = left[k] + right[k];
                let diff = left[k] - right[k];
                sum + diff * (TAU * 38_000.0 * t).cos() + pilot_amp * (TAU * 19_000.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn separates_a_signal_present_on_only_one_channel() {
        let n = 240_000;
        let left: Vec<f32> = (0..n)
            .map(|k| 0.5 * (TAU * 1000.0 * k as f32 / WFM_RATE).sin())
            .collect();
        let right = vec![0.0f32; n];
        let mpx = build_mpx(&left, &right, 0.1, WFM_RATE);

        let mut dec = StereoDecoder::new(WFM_RATE);
        let mut l = vec![0.0; n];
        let mut r = vec![0.0; n];
        let stereo = dec.process(&mpx, &mut l, &mut r);
        assert!(stereo, "pilot present but stereo not detected");

        // Measure once the pilot loop has settled.
        let tail = 180_000;
        let l_amp = tone_amplitude(&l[tail..], 1000.0, WFM_RATE);
        let r_amp = tone_amplitude(&r[tail..], 1000.0, WFM_RATE);

        assert!(
            (l_amp - 0.5).abs() < 0.08,
            "left amplitude {l_amp}, expected 0.5"
        );
        let separation_db = 20.0 * (l_amp / r_amp.max(1e-9)).log10();
        assert!(
            separation_db > 20.0,
            "only {separation_db:.1} dB of channel separation"
        );
    }

    #[test]
    fn falls_back_to_mono_without_a_pilot() {
        let n = 120_000;
        let left: Vec<f32> = (0..n)
            .map(|k| 0.4 * (TAU * 700.0 * k as f32 / WFM_RATE).sin())
            .collect();
        let right: Vec<f32> = (0..n)
            .map(|k| 0.4 * (TAU * 700.0 * k as f32 / WFM_RATE).sin())
            .collect();
        // Zero pilot amplitude: a mono broadcast.
        let mpx = build_mpx(&left, &right, 0.0, WFM_RATE);

        let mut dec = StereoDecoder::new(WFM_RATE);
        let mut l = vec![0.0; n];
        let mut r = vec![0.0; n];
        let stereo = dec.process(&mpx, &mut l, &mut r);

        assert!(!stereo, "claimed stereo with no pilot present");
        for k in 100_000..n {
            assert!((l[k] - r[k]).abs() < 1e-6, "mono channels differ at {k}");
        }
    }

    #[test]
    fn forced_mono_overrides_a_present_pilot() {
        let n = 120_000;
        let left: Vec<f32> = (0..n)
            .map(|k| 0.5 * (TAU * 1000.0 * k as f32 / WFM_RATE).sin())
            .collect();
        let right = vec![0.0f32; n];
        let mpx = build_mpx(&left, &right, 0.1, WFM_RATE);

        let mut dec = StereoDecoder::new(WFM_RATE);
        dec.set_forced_mono(true);
        let mut l = vec![0.0; n];
        let mut r = vec![0.0; n];
        assert!(!dec.process(&mpx, &mut l, &mut r));
        for k in 100_000..n {
            assert_eq!(l[k], r[k], "forced mono still produced a difference at {k}");
        }
    }

    #[test]
    #[should_panic(expected = "headroom above the 38 kHz subcarrier")]
    fn stereo_rejects_a_rate_that_cannot_carry_the_subcarrier() {
        StereoDecoder::new(48_000.0);
    }
}
