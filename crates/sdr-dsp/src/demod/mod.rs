//! Demodulators.
//!
//! Each takes a block of complex baseband centred on the wanted signal and produces real
//! audio at the same rate. Rate conversion to the output device happens afterwards, in
//! [`crate::resample`], so the demodulators never have to know what the sound card wants.

pub mod am;
pub mod fm;
pub mod pll;
pub mod ssb;

pub use am::AmDemod;
pub use fm::{FmDemod, StereoDecoder};
pub use pll::Pll;
pub use ssb::{Sideband, SsbDemod};

/// Modes the receiver can be switched between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Narrowband FM. Voice repeaters, marine, amateur, business radio.
    Nfm,
    /// Wideband FM. Broadcast, optionally with stereo and RDS.
    Wfm,
    /// Amplitude modulation, envelope detected.
    Am,
    /// Upper sideband.
    Usb,
    /// Lower sideband.
    Lsb,
    /// Continuous wave. Upper sideband with a narrow filter and an audible pitch offset.
    Cw,
}

impl Mode {
    /// Channel bandwidth the mode expects, in hertz.
    ///
    /// This is the full width of the signal, so the channel filter that precedes the
    /// demodulator should be a low-pass at half of it.
    pub fn bandwidth(self) -> f32 {
        match self {
            Mode::Nfm => 12_500.0,
            Mode::Wfm => 200_000.0,
            Mode::Am => 10_000.0,
            Mode::Usb | Mode::Lsb => 2_700.0,
            Mode::Cw => 500.0,
        }
    }

    /// Whether the mode carries a stereo subcarrier worth decoding.
    pub fn is_stereo_capable(self) -> bool {
        matches!(self, Mode::Wfm)
    }

    pub fn label(self) -> &'static str {
        match self {
            Mode::Nfm => "NFM",
            Mode::Wfm => "WFM",
            Mode::Am => "AM",
            Mode::Usb => "USB",
            Mode::Lsb => "LSB",
            Mode::Cw => "CW",
        }
    }
}

/// Two-argument arctangent, accurate to about a millionth of a radian.
///
/// The FM discriminator evaluates this once per sample at the channel rate, so the
/// library version's exactness is not worth its cost. A degree-nine odd polynomial on
/// `[-1, 1]` plus octant folding is well inside what an 8-bit ADC's noise floor can
/// justify, and the tests hold it to the library implementation.
#[inline]
pub fn fast_atan2(y: f32, x: f32) -> f32 {
    use core::f32::consts::{FRAC_PI_2, PI};

    if x == 0.0 && y == 0.0 {
        return 0.0;
    }

    let ax = x.abs();
    let ay = y.abs();
    // Fold into the octant where the ratio is within [0, 1], so one polynomial covers it.
    let (num, den, swapped) = if ay <= ax {
        (ay, ax, false)
    } else {
        (ax, ay, true)
    };
    let r = num / den;
    let r2 = r * r;

    // Minimax-style odd polynomial for atan on [0, 1].
    let mut a =
        r * (0.999866 + r2 * (-0.330299 + r2 * (0.180141 + r2 * (-0.085133 + r2 * 0.020835))));

    if swapped {
        a = FRAC_PI_2 - a;
    }
    if x < 0.0 {
        a = PI - a;
    }
    if y < 0.0 {
        a = -a;
    }
    a
}

/// Wraps a phase in radians into `(-pi, pi]`.
#[inline]
pub fn wrap_phase(mut phase: f32) -> f32 {
    use core::f32::consts::{PI, TAU};
    while phase > PI {
        phase -= TAU;
    }
    while phase <= -PI {
        phase += TAU;
    }
    phase
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn fast_atan2_tracks_the_library_everywhere() {
        let mut worst = 0.0f32;
        // Sweep the full circle at a radius that exercises both octant folds.
        for k in 0..3600 {
            let angle = k as f32 * PI / 1800.0 - PI;
            let (y, x) = (angle.sin(), angle.cos());
            let err = (fast_atan2(y, x) - y.atan2(x)).abs();
            worst = worst.max(err);
        }
        // The polynomial's own approximation error sets this floor, and single-precision
        // rounding adds to it. Well below what matters: an 8-bit converter offers about
        // 48 dB of dynamic range, and this is nearer 90 dB down.
        assert!(worst < 5e-5, "worst-case error {worst} rad is too large");
    }

    #[test]
    fn fast_atan2_handles_the_axes_and_origin() {
        assert!((fast_atan2(0.0, 1.0) - 0.0).abs() < 1e-6);
        assert!((fast_atan2(1.0, 0.0) - PI / 2.0).abs() < 1e-6);
        assert!((fast_atan2(0.0, -1.0) - PI).abs() < 1e-5);
        assert!((fast_atan2(-1.0, 0.0) + PI / 2.0).abs() < 1e-6);
        assert_eq!(fast_atan2(0.0, 0.0), 0.0);
    }

    #[test]
    fn fast_atan2_is_accurate_at_very_small_and_large_ratios() {
        for (y, x) in [(1e-6f32, 1.0f32), (1.0, 1e-6), (-1e-6, -1.0), (1e6, 1.0)] {
            let err = (fast_atan2(y, x) - y.atan2(x)).abs();
            assert!(err < 1e-5, "error {err} at ({y}, {x})");
        }
    }

    #[test]
    fn wrap_phase_folds_into_the_principal_range() {
        use core::f32::consts::TAU;
        assert!((wrap_phase(0.5) - 0.5).abs() < 1e-6);
        assert!((wrap_phase(0.5 + TAU) - 0.5).abs() < 1e-5);
        assert!((wrap_phase(0.5 - TAU) - 0.5).abs() < 1e-5);
        assert!(wrap_phase(PI * 3.0) <= PI);
        assert!(wrap_phase(-PI * 3.0) > -PI);
    }

    #[test]
    fn mode_bandwidths_are_ordered_sensibly() {
        assert!(Mode::Cw.bandwidth() < Mode::Usb.bandwidth());
        assert!(Mode::Usb.bandwidth() < Mode::Am.bandwidth());
        assert!(Mode::Am.bandwidth() < Mode::Nfm.bandwidth());
        assert!(Mode::Nfm.bandwidth() < Mode::Wfm.bandwidth());
        assert!(Mode::Wfm.is_stereo_capable());
        assert!(!Mode::Nfm.is_stereo_capable());
    }
}
