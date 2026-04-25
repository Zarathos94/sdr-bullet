//! Window functions, plus the scaling factors needed to read a spectrum correctly.
//!
//! Two conventions matter and are routinely conflated. Spectral analysis wants the
//! *periodic* form, where the window has period `n` and its last sample is the one that
//! would wrap onto the first; filter design wants the *symmetric* form, which is a
//! palindrome of length `n`. Using the symmetric form for an FFT leaks energy into
//! neighbouring bins, so [`Window::periodic`] and [`Window::symmetric`] are separate calls
//! rather than a flag someone can forget.

use core::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    Rectangular,
    Hann,
    Hamming,
    /// Four-term Blackman-Harris. Sidelobes below -92 dB, which is what a waterfall needs
    /// to keep a strong neighbouring carrier from smearing across the display.
    BlackmanHarris,
    /// Kaiser, parameterised by beta scaled by 100 so the enum stays `Eq` and hashable.
    /// Use [`Window::kaiser`] rather than building this directly.
    Kaiser(u32),
}

impl Window {
    /// Kaiser window with the given beta. Higher beta trades main-lobe width for
    /// sidelobe suppression; 8.6 is the usual starting point for around -90 dB.
    pub fn kaiser(beta: f32) -> Self {
        assert!(
            beta >= 0.0 && beta.is_finite(),
            "kaiser beta must be finite and non-negative"
        );
        Window::Kaiser((beta * 100.0).round() as u32)
    }

    fn beta(self) -> f32 {
        match self {
            Window::Kaiser(b) => b as f32 / 100.0,
            _ => 0.0,
        }
    }

    /// Window of length `n` with period `n`. Correct for FFT analysis.
    pub fn periodic(self, n: usize) -> Vec<f32> {
        self.build(n, n)
    }

    /// Window of length `n` that is symmetric about its centre. Correct for FIR design.
    pub fn symmetric(self, n: usize) -> Vec<f32> {
        self.build(n, n.saturating_sub(1).max(1))
    }

    fn build(self, n: usize, denom: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        let d = denom as f32;
        (0..n)
            .map(|k| {
                let x = k as f32 / d;
                match self {
                    Window::Rectangular => 1.0,
                    Window::Hann => 0.5 - 0.5 * (2.0 * PI * x).cos(),
                    Window::Hamming => 0.54 - 0.46 * (2.0 * PI * x).cos(),
                    Window::BlackmanHarris => {
                        const A: [f32; 4] = [0.35875, 0.48829, 0.14128, 0.01168];
                        A[0] - A[1] * (2.0 * PI * x).cos() + A[2] * (4.0 * PI * x).cos()
                            - A[3] * (6.0 * PI * x).cos()
                    }
                    Window::Kaiser(_) => {
                        let beta = self.beta();
                        // Argument runs from -1 to +1 across the window.
                        let t = 2.0 * x - 1.0;
                        let arg = 1.0 - t * t;
                        // Guard the endpoint, where rounding can push `arg` slightly negative.
                        let arg = if arg < 0.0 { 0.0 } else { arg };
                        bessel_i0(beta * arg.sqrt()) / bessel_i0(beta)
                    }
                }
            })
            .collect()
    }
}

/// Scaling factors that turn a windowed FFT magnitude into a calibrated reading.
#[derive(Debug, Clone, Copy)]
pub struct WindowGain {
    /// Mean of the window. Divide a bin magnitude by this to recover the amplitude of a
    /// coherent tone sitting on that bin.
    pub coherent: f32,
    /// Equivalent noise bandwidth, in bins. Divide a bin's power by this to get a power
    /// spectral density that does not depend on which window was chosen.
    pub enbw: f32,
}

/// Computes the scaling factors for an already-built window.
pub fn gain(w: &[f32]) -> WindowGain {
    if w.is_empty() {
        return WindowGain {
            coherent: 1.0,
            enbw: 1.0,
        };
    }
    let n = w.len() as f32;
    let sum: f32 = w.iter().sum();
    let sum_sq: f32 = w.iter().map(|v| v * v).sum();
    WindowGain {
        coherent: sum / n,
        // n * sum(w^2) / sum(w)^2, the standard definition in bins.
        enbw: if sum > 0.0 {
            n * sum_sq / (sum * sum)
        } else {
            1.0
        },
    }
}

/// Zeroth-order modified Bessel function of the first kind.
///
/// Evaluated by its power series, which converges quickly for the arguments a Kaiser
/// window produces (beta rarely exceeds 20). Terms are built incrementally so no
/// factorial or power is computed directly.
fn bessel_i0(x: f32) -> f32 {
    let half_x = x as f64 / 2.0;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for k in 1..64 {
        term *= half_x / k as f64;
        let contribution = term * term;
        sum += contribution;
        if contribution < sum * 1e-12 {
            break;
        }
    }
    sum as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_periodic_starts_at_zero_and_does_not_return_to_it() {
        let w = Window::Hann.periodic(8);
        assert!(w[0].abs() < 1e-6);
        // The periodic form's last sample is one step short of the wrap point, so it is
        // nonzero. This is exactly what distinguishes it from the symmetric form.
        assert!(
            w[7] > 0.1,
            "periodic Hann should not close at zero: {}",
            w[7]
        );
    }

    #[test]
    fn hann_symmetric_is_a_palindrome_closing_at_zero() {
        let w = Window::Hann.symmetric(9);
        assert!(w[0].abs() < 1e-6);
        assert!(w[8].abs() < 1e-6);
        for k in 0..9 {
            assert!((w[k] - w[8 - k]).abs() < 1e-6, "not symmetric at {k}");
        }
    }

    #[test]
    fn hann_peaks_at_unity_in_the_middle() {
        let w = Window::Hann.symmetric(101);
        assert!((w[50] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hamming_has_a_nonzero_pedestal() {
        let w = Window::Hamming.symmetric(64);
        assert!(
            (w[0] - 0.08).abs() < 1e-3,
            "Hamming endpoint should be 0.08, got {}",
            w[0]
        );
    }

    #[test]
    fn blackman_harris_coefficients_sum_to_the_endpoint() {
        let w = Window::BlackmanHarris.symmetric(128);
        // a0 - a1 + a2 - a3 at both ends.
        let expected = 0.35875 - 0.48829 + 0.14128 - 0.01168;
        assert!((w[0] - expected).abs() < 1e-5);
        assert!((w[127] - expected).abs() < 1e-5);
    }

    #[test]
    fn bessel_i0_matches_reference_values() {
        // Reference values for I0 at small integer arguments.
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-6);
        assert!((bessel_i0(1.0) - 1.2660658).abs() < 1e-5);
        assert!((bessel_i0(2.0) - 2.2795853).abs() < 1e-4);
        assert!((bessel_i0(5.0) - 27.239872).abs() < 1e-2);
    }

    #[test]
    fn kaiser_is_symmetric_and_normalised() {
        let w = Window::kaiser(8.6).symmetric(65);
        assert!(
            (w[32] - 1.0).abs() < 1e-5,
            "centre should be unity, got {}",
            w[32]
        );
        for k in 0..65 {
            assert!((w[k] - w[64 - k]).abs() < 1e-6, "not symmetric at {k}");
        }
        // Higher beta tapers harder at the edges.
        let wide = Window::kaiser(2.0).symmetric(65);
        assert!(w[0] < wide[0]);
    }

    #[test]
    fn rectangular_gain_is_unity() {
        let g = gain(&Window::Rectangular.periodic(256));
        assert!((g.coherent - 1.0).abs() < 1e-6);
        assert!((g.enbw - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hann_gain_matches_known_constants() {
        let g = gain(&Window::Hann.periodic(4096));
        // Hann's mean is 0.5 and its equivalent noise bandwidth is 1.5 bins.
        assert!(
            (g.coherent - 0.5).abs() < 1e-3,
            "coherent gain {}",
            g.coherent
        );
        assert!((g.enbw - 1.5).abs() < 1e-3, "enbw {}", g.enbw);
    }

    #[test]
    fn empty_and_single_sample_windows_are_handled() {
        assert!(Window::Hann.periodic(0).is_empty());
        assert_eq!(Window::Hann.symmetric(1).len(), 1);
        let g = gain(&[]);
        assert_eq!(g.coherent, 1.0);
    }
}
