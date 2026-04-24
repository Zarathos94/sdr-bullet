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

