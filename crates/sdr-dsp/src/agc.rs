//! Automatic gain control.
//!
//! Separate attack and decay time constants, because the two directions want very
//! different behaviour: gain has to come down fast enough that a sudden strong signal does
//! not clip, but back up slowly enough that the noise floor between words in a
//! transmission is not pumped up into a roar. Attack of a few milliseconds against a decay
//! of a second or so is the usual shape, and is what the defaults encode.

/// Envelope-following gain control.
#[derive(Debug, Clone)]
pub struct Agc {
    target: f32,
    max_gain: f32,
    attack: f32,
    decay: f32,
    envelope: f32,
    gain: f32,
    enabled: bool,
}

impl Agc {
    /// Builds a controller for the given sample rate with a 5 ms attack and 500 ms decay.
    ///
    /// # Panics
    /// If `sample_rate` is not positive.
    pub fn new(sample_rate: f32) -> Self {
        assert!(sample_rate > 0.0, "sample rate must be positive");
        let mut agc = Self {
            target: 0.25,
            max_gain: 512.0,
            attack: 0.0,
            decay: 0.0,
            envelope: 0.0,
            gain: 1.0,
            enabled: true,
        };
        agc.set_times(sample_rate, 0.005, 0.5);
        agc
    }

    /// Sets attack and decay time constants in seconds.
    ///
    /// # Panics
    /// If either time is not positive.
    pub fn set_times(&mut self, sample_rate: f32, attack_s: f32, decay_s: f32) {
        assert!(
            attack_s > 0.0 && decay_s > 0.0,
            "time constants must be positive"
        );
        // One-pole step response: the coefficient that reaches 1 - 1/e in tau seconds.
        self.attack = 1.0 - (-1.0 / (attack_s * sample_rate)).exp();
        self.decay = 1.0 - (-1.0 / (decay_s * sample_rate)).exp();
    }

    /// Output amplitude the controller aims for, as a fraction of full scale.
    ///
    /// # Panics
    /// If `target` is outside `(0, 1]`.
    pub fn set_target(&mut self, target: f32) {
        assert!(target > 0.0 && target <= 1.0, "target must be in (0, 1]");
        self.target = target;
    }

    /// Caps how far a weak signal can be lifted, which bounds how loud the noise floor
    /// gets when there is nothing to receive.
    ///
    /// # Panics
    /// If `max_gain` is less than one.
    pub fn set_max_gain(&mut self, max_gain: f32) {
        assert!(max_gain >= 1.0, "maximum gain must be at least 1");
        self.max_gain = max_gain;
    }

    /// When disabled the signal passes through untouched, but the envelope keeps tracking
    /// so re-enabling does not start from a stale estimate.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn envelope(&self) -> f32 {
        self.envelope
    }

    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.gain = 1.0;
    }

    /// Applies gain control in place.
    ///
    /// The gain is read straight off the envelope rather than being smoothed again on top
    /// of it. The envelope's own attack and decay already carry the asymmetry the gain
    /// needs — it falls quickly when the signal gets loud and rises slowly when it goes
    /// quiet — so a second one-pole stage in series would only double the settling time
    /// while adding nothing. The envelope is smooth, so the gain derived from it is too.
    pub fn process(&mut self, x: &mut [f32]) {
        // Below this the envelope is treated as silence, which stops the gain from
        // diverging towards max_gain on a digitally-silent input.
        const FLOOR: f32 = 1e-6;

        for sample in x.iter_mut() {
            let level = sample.abs();
            let coeff = if level > self.envelope {
                self.attack
            } else {
                self.decay
            };
            self.envelope += (level - self.envelope) * coeff;

            self.gain = if self.envelope > FLOOR {
                (self.target / self.envelope).min(self.max_gain)
            } else {
                self.max_gain
            };

            if self.enabled {
                *sample *= self.gain;
            }
        }
    }
}

/// First-order de-emphasis for FM audio.
///
/// Broadcast FM pre-emphasises the transmitted treble to improve the signal-to-noise ratio
/// of the high end, and the receiver has to undo it with the matching time constant. The
/// constant is regional: 50 microseconds across Europe, 75 in the Americas. Getting it
/// wrong does not break anything, it just leaves the audio sounding dull or harsh.
#[derive(Debug, Clone, Copy)]
pub struct Deemphasis {
    alpha: f32,
    state: f32,
}

/// Regional de-emphasis time constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emphasis {
    /// 50 microseconds. Europe, Africa, Asia, Australia.
    Us50,
    /// 75 microseconds. North and South America, South Korea.
    Us75,
    /// No de-emphasis. Correct for narrowband voice, which is not pre-emphasised the
    /// same way, and useful when feeding a decoder rather than a speaker.
    None,
}

impl Emphasis {
    pub fn seconds(self) -> f32 {
        match self {
            Emphasis::Us50 => 50e-6,
            Emphasis::Us75 => 75e-6,
            Emphasis::None => 0.0,
        }
    }
}

impl Deemphasis {
    /// # Panics
    /// If `sample_rate` is not positive.
    pub fn new(sample_rate: f32, emphasis: Emphasis) -> Self {
        assert!(sample_rate > 0.0, "sample rate must be positive");
        let tau = emphasis.seconds();
        let alpha = if tau > 0.0 {
            1.0 - (-1.0 / (tau * sample_rate)).exp()
        } else {
            1.0
        };
        Self { alpha, state: 0.0 }
    }

    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Applies the filter in place. An alpha of one is a pass-through.
    pub fn process(&mut self, x: &mut [f32]) {
        if self.alpha >= 1.0 {
            return;
        }
        for sample in x.iter_mut() {
            self.state += (*sample - self.state) * self.alpha;
            *sample = self.state;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn peak(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    }

    #[test]
    fn lifts_a_quiet_signal_towards_the_target() {
        let mut agc = Agc::new(SR);
        agc.set_target(0.5);
        // Long enough for the decay path, which governs upward gain, to settle.
        let mut x: Vec<f32> = (0..SR as usize * 3)
            .map(|k| 0.01 * (k as f32 * 0.05).sin())
            .collect();
        agc.process(&mut x);

        let tail = &x[x.len() - 4800..];
        assert!(
            (peak(tail) - 0.5).abs() < 0.1,
            "settled peak {} should approach the 0.5 target",
            peak(tail)
        );
    }

    #[test]
    fn pulls_a_loud_signal_down_quickly() {
        let mut agc = Agc::new(SR);
        agc.set_target(0.25);
        // 20 ms is only a few attack constants, so a correct attack has already acted.
        let mut x: Vec<f32> = (0..960).map(|k| 4.0 * (k as f32 * 0.05).sin()).collect();
        agc.process(&mut x);

        let tail = &x[480..];
        assert!(
            peak(tail) < 0.6,
            "attack too slow, peak still {}",
            peak(tail)
        );
    }

    #[test]
    fn attack_is_faster_than_decay() {
        let agc = Agc::new(SR);
        assert!(
            agc.attack > agc.decay,
            "attack coefficient {} should exceed decay {}",
            agc.attack,
            agc.decay
        );
    }

    #[test]
    fn silence_does_not_run_the_gain_away() {
        let mut agc = Agc::new(SR);
        agc.set_max_gain(100.0);
        let mut x = vec![0.0f32; 48_000];
        agc.process(&mut x);
        for v in &x {
            assert_eq!(*v, 0.0, "silence should stay silent");
        }
        assert!(
            agc.gain() <= 100.0 + 1e-3,
            "gain exceeded its cap: {}",
            agc.gain()
        );
    }

    #[test]
    fn disabled_agc_leaves_the_signal_alone_but_keeps_tracking() {
        let mut agc = Agc::new(SR);
        agc.set_enabled(false);
        let orig: Vec<f32> = (0..1000).map(|k| 0.01 * (k as f32 * 0.1).sin()).collect();
        let mut x = orig.clone();
        agc.process(&mut x);

        assert_eq!(x, orig, "disabled AGC must not alter the signal");
        assert!(agc.envelope() > 0.0, "envelope should still be tracking");
    }

    #[test]
    fn reset_clears_the_envelope() {
        let mut agc = Agc::new(SR);
        let mut x = vec![0.5f32; 1000];
        agc.process(&mut x);
        assert!(agc.envelope() > 0.0);
        agc.reset();
        assert_eq!(agc.envelope(), 0.0);
        assert_eq!(agc.gain(), 1.0);
    }

    #[test]
    fn deemphasis_attenuates_treble_more_than_bass() {
        // Measure steady-state amplitude of a tone through the filter.
        let measure = |freq: f32| {
            let mut d = Deemphasis::new(SR, Emphasis::Us50);
            let mut x: Vec<f32> = (0..8192)
                .map(|k| (core::f32::consts::TAU * freq * k as f32 / SR).sin())
                .collect();
            d.process(&mut x);
            peak(&x[4096..])
        };

        let bass = measure(100.0);
        let treble = measure(10_000.0);
        assert!(
            bass > 0.9,
            "100 Hz should pass nearly untouched, got {bass}"
        );
        assert!(
            treble < 0.4,
            "10 kHz should be well attenuated, got {treble}"
        );
        assert!(bass > treble * 2.0);
    }

    #[test]
    fn the_two_regional_constants_differ_in_the_right_direction() {
        let measure = |em: Emphasis| {
            let mut d = Deemphasis::new(SR, em);
            let mut x: Vec<f32> = (0..8192)
                .map(|k| (core::f32::consts::TAU * 8000.0 * k as f32 / SR).sin())
                .collect();
            d.process(&mut x);
            peak(&x[4096..])
        };
        // The longer 75 us constant rolls off earlier, so it cuts more at a given tone.
        assert!(measure(Emphasis::Us75) < measure(Emphasis::Us50));
    }

    #[test]
    fn no_emphasis_is_a_pass_through() {
        let mut d = Deemphasis::new(SR, Emphasis::None);
        let orig: Vec<f32> = (0..256).map(|k| (k as f32 * 0.3).sin()).collect();
        let mut x = orig.clone();
        d.process(&mut x);
        assert_eq!(x, orig);
    }

    #[test]
    #[should_panic(expected = "target must be in")]
    fn rejects_a_target_above_full_scale() {
        Agc::new(SR).set_target(1.5);
    }
}
