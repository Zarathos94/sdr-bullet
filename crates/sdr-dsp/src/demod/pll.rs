//! Second-order phase-locked loop.
//!
//! Used to recover the 19 kHz stereo pilot from a broadcast FM multiplex. Locking to the
//! pilot rather than free-running a local 38 kHz oscillator is what makes stereo work at
//! all: the difference subcarrier is suppressed-carrier, so its phase is only recoverable
//! from the pilot, and a few degrees of error bleeds one channel into the other.
//!
//! The loop is second order, so it tracks a frequency offset with no steady-state phase
//! error. That matters because the transmitter's pilot and the receiver's crystal are
//! never at quite the same frequency.

use core::f32::consts::TAU;

/// Tracks the phase and frequency of a single tone.
#[derive(Debug, Clone)]
pub struct Pll {
    /// Current phase in radians, kept in `[0, TAU)`.
    phase: f32,
    /// Current frequency in radians per sample.
    freq: f32,
    /// Rest frequency, and the centre of the range the loop may pull to.
    centre: f32,
    min_freq: f32,
    max_freq: f32,
    /// Proportional and integral gains of the loop filter.
    alpha: f32,
    beta: f32,
    /// Smoothed magnitude of the in-phase product, used as a lock indicator.
    lock: f32,
    lock_smoothing: f32,
}

impl Pll {
    /// Builds a loop centred on `centre_hz` that may pull `pull_range_hz` either side.
    ///
    /// `damping` of 0.707 and a loop bandwidth of a few tens of hertz is the usual
    /// starting point for a pilot: wide enough to acquire quickly, narrow enough that
    /// programme material in the neighbouring channels does not pull it off.
    ///
    /// # Panics
    /// If the sample rate is not positive, or the centre frequency is beyond Nyquist.
    pub fn new(sample_rate: f32, centre_hz: f32, pull_range_hz: f32, loop_bw_hz: f32) -> Self {
        assert!(sample_rate > 0.0, "sample rate must be positive");
        assert!(
            centre_hz > 0.0 && centre_hz < sample_rate / 2.0,
            "centre frequency must be below Nyquist"
        );

        let centre = TAU * centre_hz / sample_rate;
        let pull = TAU * pull_range_hz / sample_rate;

        // Standard second-order loop coefficients for a given normalised bandwidth and
        // critical-ish damping.
        let damping = 0.707f32;
        let wn = TAU * loop_bw_hz / sample_rate;
        let denom = 1.0 + 2.0 * damping * wn + wn * wn;
        let alpha = 4.0 * damping * wn / denom;
        let beta = 4.0 * wn * wn / denom;

        Self {
            phase: 0.0,
            freq: centre,
            centre,
            min_freq: centre - pull,
            max_freq: centre + pull,
            alpha,
            beta,
            lock: 0.0,
            // Lock is judged over roughly a hundred cycles of the tracked tone.
            lock_smoothing: 1.0 - (-1.0 / (100.0 * sample_rate / centre_hz)).exp(),
        }
    }

    /// Advances the loop by one sample of a real input, returning the loop's phase before
    /// the update.
    ///
    /// The returned phase is the one that corresponds to the sample just consumed, which
    /// is what a downstream mixer wants.
    #[inline]
    pub fn advance(&mut self, sample: f32) -> f32 {
        let phase = self.phase;
        let (sin, cos) = phase.sin_cos();

        // The loop is defined to settle with its own sine aligned to the input, so that a
        // caller can treat `phase` as the tone's phase directly. Multiplying the input by
        // the cosine then averages to sin(input - phase), which is zero at alignment and
        // signed the right way to correct a lag; multiplying by the sine averages to the
        // amplitude, which is the lock indicator.
        //
        // Taking these the other way round also produces a stable lock, but one sitting a
        // quarter cycle away. That is invisible on its own and catastrophic downstream:
        // doubling a phase that is 90 degrees out inverts the stereo difference signal,
        // which silently swaps the left and right channels.
        let error = sample * cos;
        let in_phase = sample * sin;

        self.lock += (in_phase - self.lock) * self.lock_smoothing;

        self.freq = (self.freq + self.beta * error).clamp(self.min_freq, self.max_freq);
        self.phase += self.freq + self.alpha * error;
        if self.phase >= TAU {
            self.phase -= TAU;
        } else if self.phase < 0.0 {
            self.phase += TAU;
        }

        phase
    }

    /// Current phase in radians.
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Current frequency in hertz, given the rate the loop was built for.
    pub fn frequency_hz(&self, sample_rate: f32) -> f32 {
        self.freq * sample_rate / TAU
    }

    /// Smoothed in-phase amplitude. Compare against the expected pilot level to decide
    /// whether the loop has acquired.
    pub fn lock_level(&self) -> f32 {
        self.lock
    }

    /// Whether the loop appears locked to a tone of at least `min_amplitude`.
    pub fn is_locked(&self, min_amplitude: f32) -> bool {
        self.lock.abs() > min_amplitude
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.freq = self.centre;
        self.lock = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 240_000.0;

    fn pilot(freq: f32, amp: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|k| amp * (TAU * freq * k as f32 / SR).sin())
            .collect()
    }

    #[test]
    fn locks_to_a_tone_at_its_centre_frequency() {
        let mut pll = Pll::new(SR, 19_000.0, 100.0, 20.0);
        for s in pilot(19_000.0, 0.1, 48_000) {
            pll.advance(s);
        }
        let f = pll.frequency_hz(SR);
        assert!((f - 19_000.0).abs() < 5.0, "settled at {f} Hz");
    }

    #[test]
    fn pulls_in_an_offset_tone_and_holds_zero_phase_error() {
        // A second-order loop should track a frequency offset exactly, unlike a
        // first-order loop which would sit at a constant phase error proportional to it.
        let mut pll = Pll::new(SR, 19_000.0, 200.0, 30.0);
        let offset = 19_050.0;
        for s in pilot(offset, 0.1, 200_000) {
            pll.advance(s);
        }
        let f = pll.frequency_hz(SR);
        assert!((f - offset).abs() < 5.0, "tracked to {f}, wanted {offset}");
    }

    #[test]
    fn refuses_to_pull_beyond_its_configured_range() {
        let mut pll = Pll::new(SR, 19_000.0, 50.0, 30.0);
        // A tone far outside the pull range must not drag the loop away with it.
        for s in pilot(21_000.0, 0.3, 100_000) {
            pll.advance(s);
        }
        let f = pll.frequency_hz(SR);
        assert!(
            (18_950.0..=19_050.0).contains(&f),
            "loop escaped its clamp, sitting at {f}"
        );
    }

    #[test]
    fn reports_lock_only_when_a_tone_is_present() {
        let mut locked = Pll::new(SR, 19_000.0, 100.0, 20.0);
        for s in pilot(19_000.0, 0.2, 100_000) {
            locked.advance(s);
        }
        assert!(locked.is_locked(0.01), "lock level {}", locked.lock_level());

        let mut unlocked = Pll::new(SR, 19_000.0, 100.0, 20.0);
        // Deterministic noise-like input with no 19 kHz component.
        let mut state = 0x1234_5678u32;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let n = (state as f32 / u32::MAX as f32) * 0.4 - 0.2;
            unlocked.advance(n);
        }
        assert!(
            !unlocked.is_locked(0.01),
            "reported lock on noise, level {}",
            unlocked.lock_level()
        );
    }

    #[test]
    fn recovered_phase_aligns_with_the_input_tone() {
        let mut pll = Pll::new(SR, 19_000.0, 100.0, 20.0);
        let samples = pilot(19_000.0, 0.1, 200_000);
        for s in &samples[..150_000] {
            pll.advance(*s);
        }

        // Once locked, the loop's sine should correlate strongly with the input and its
        // cosine should not — that is what having zero phase error means, and it is the
        // convention the stereo decoder depends on when it doubles this phase.
        let (mut corr_sin, mut corr_cos) = (0.0f64, 0.0f64);
        for s in &samples[150_000..] {
            let ph = pll.advance(*s);
            corr_sin += (*s * ph.sin()) as f64;
            corr_cos += (*s * ph.cos()) as f64;
        }
        assert!(
            corr_sin.abs() > corr_cos.abs() * 8.0,
            "phase misaligned: sin correlation {corr_sin}, cos {corr_cos}"
        );
    }

    #[test]
    fn reset_returns_the_loop_to_its_rest_frequency() {
        let mut pll = Pll::new(SR, 19_000.0, 200.0, 30.0);
        for s in pilot(19_100.0, 0.2, 50_000) {
            pll.advance(s);
        }
        pll.reset();
        assert!((pll.frequency_hz(SR) - 19_000.0).abs() < 0.1);
        assert_eq!(pll.lock_level(), 0.0);
    }

    #[test]
    #[should_panic(expected = "below Nyquist")]
    fn rejects_a_centre_frequency_above_nyquist() {
        Pll::new(48_000.0, 30_000.0, 100.0, 20.0);
    }
}
