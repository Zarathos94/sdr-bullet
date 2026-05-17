//! Rational rate conversion by polyphase filtering.
//!
//! Converting by `L/M` conceptually means inserting `L - 1` zeros between samples, filtering,
//! then keeping one in `M`. Doing that literally would multiply by zeros and then throw
//! most of the results away. The polyphase form skips both: the prototype is split into
//! `L` interleaved subfilters, and each output picks the one subfilter its fractional
//! position calls for.
//!
//! The receive chain mostly hits integer ratios — 2.4 MSPS decimates by ten to 240 kHz for
//! wideband FM, then by five to 48 kHz of audio — but the tuner's actual rate is set by a
//! register ratio and does not always land exactly where it was asked to, so the general
//! case has to work too.

use crate::fir::design_lowpass;
use crate::window::Window;

/// Resamples by an exact `interp / decim` ratio.
#[derive(Debug, Clone)]
pub struct RationalResampler {
    interp: usize,
    decim: usize,
    /// Prototype split into `interp` branches, each `taps_per_phase` long. Branch `p`
    /// occupies `phases[p * taps_per_phase .. (p + 1) * taps_per_phase]`, already reversed
    /// so the inner loop walks history forwards.
    phases: Vec<f32>,
    taps_per_phase: usize,
    history: Vec<f32>,
    /// Position within the output cycle, carried across calls so block boundaries do not
    /// disturb the output rate.
    counter: usize,
}

impl RationalResampler {
    /// Builds a resampler with a Kaiser-windowed prototype.
    ///
    /// `taps_per_phase` sets the quality: eight is transparent for audio, sixteen is
    /// comfortable headroom. The total prototype length is `taps_per_phase * interp`.
    ///
    /// # Panics
    /// If either rate is zero, or `taps_per_phase` is zero.
    pub fn new(interp: usize, decim: usize, taps_per_phase: usize) -> Self {
        assert!(interp >= 1 && decim >= 1, "rates must be at least 1");
        assert!(taps_per_phase >= 1, "each phase needs at least one tap");

        let (interp, decim) = reduce(interp, decim);

        // The prototype runs at the interpolated rate, so its cutoff has to protect the
        // lower of the two Nyquist limits. Interpolating alone needs 0.5/interp;
        // decimating needs 0.5/decim to avoid folding. Take whichever is tighter, with a
        // little margin so the transition band completes before the fold point.
        let cutoff = 0.4 / interp.max(decim) as f32;

        // An odd length keeps the prototype linear phase.
        let total = taps_per_phase * interp;
        let total = if total % 2 == 0 { total + 1 } else { total };
        let proto = design_lowpass(total, cutoff, Window::kaiser(8.6));

        // Interpolation spreads each input across `interp` outputs, so the branch gain has
        // to be scaled back up or the signal loses that factor in level.
        let mut phases = vec![0.0f32; interp * taps_per_phase];
        for p in 0..interp {
            for j in 0..taps_per_phase {
                let idx = p + j * interp;
                let v = if idx < proto.len() {
                    proto[idx] * interp as f32
                } else {
                    0.0
                };
                // Reversed within the branch: history is walked oldest to newest.
                phases[p * taps_per_phase + (taps_per_phase - 1 - j)] = v;
            }
        }

        Self {
            interp,
            decim,
            phases,
            taps_per_phase,
            history: vec![0.0; taps_per_phase - 1],
            counter: 0,
        }
    }

    /// Builds a resampler for a pair of sample rates, reducing the ratio first.
    ///
    /// # Panics
    /// If either rate is not positive.
    pub fn for_rates(from_hz: f64, to_hz: f64, taps_per_phase: usize) -> Self {
        assert!(
            from_hz > 0.0 && to_hz > 0.0,
            "sample rates must be positive"
        );
        let (interp, decim) = rational_approximation(to_hz / from_hz, 2048);
        Self::new(interp, decim, taps_per_phase)
    }

    pub fn interp(&self) -> usize {
        self.interp
    }

    pub fn decim(&self) -> usize {
        self.decim
    }

    /// Exact output rate for a given input rate.
    pub fn output_rate(&self, input_rate: f64) -> f64 {
        input_rate * self.interp as f64 / self.decim as f64
    }

    /// Upper bound on outputs from an input block of `n` samples. Use this to size the
    /// destination buffer; the exact count varies by one between calls.
    pub fn output_len(&self, n: usize) -> usize {
        (n * self.interp).div_ceil(self.decim) + 1
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.counter = 0;
    }

    /// Returns the number of output samples written.
    ///
    /// # Panics
    /// If `output` cannot hold [`RationalResampler::output_len`] samples.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        assert!(
            output.len() >= self.output_len(input.len()),
            "output buffer too short"
        );

        let tpp = self.taps_per_phase;
        let hist = tpp - 1;

        // One contiguous window of history followed by the new block, so each branch reads
        // a straight run of memory.
        let mut buf = Vec::with_capacity(hist + input.len());
        buf.extend_from_slice(&self.history);
        buf.extend_from_slice(input);

        let mut written = 0;
        // `counter` counts output slots in units of the input grid scaled by `interp`.
        let mut c = self.counter;
        loop {
            let n = c / self.interp;
            if n >= input.len() {
                break;
            }
            let p = c % self.interp;
            let branch = &self.phases[p * tpp..(p + 1) * tpp];
            let window = &buf[n..n + tpp];

            let mut acc = 0.0f32;
            for j in 0..tpp {
                acc += branch[j] * window[j];
            }
            output[written] = acc;
            written += 1;
            c += self.decim;
        }
        // Rebase the counter onto the next block's grid.
        self.counter = c - input.len() * self.interp;

        let keep = buf.len().saturating_sub(hist);
        self.history.clear();
        self.history.extend_from_slice(&buf[keep..]);
        while self.history.len() < hist {
            self.history.insert(0, 0.0);
        }

        written
    }
}

/// Divides both terms by their greatest common divisor.
fn reduce(a: usize, b: usize) -> (usize, usize) {
    let g = gcd(a, b);
    (a / g, b / g)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

/// Closest rational `p/q` to `value` with `q` no larger than `max_denominator`.
///
/// Continued-fraction expansion, which yields the best approximation for any bound on the
/// denominator — relevant because the tuner's achievable sample rates are quotients of a
/// crystal frequency and rarely round numbers.
///
/// # Panics
/// If `value` is not positive and finite.
pub fn rational_approximation(value: f64, max_denominator: usize) -> (usize, usize) {
    assert!(
        value > 0.0 && value.is_finite(),
        "ratio must be positive and finite"
    );

    // Convergent recurrence seeds: numerators run 0, 1 and denominators run 1, 0. Crossing
    // the two pairs yields the reciprocal of the intended answer.
    let (mut p_prev, mut q_prev) = (0usize, 1usize);
    let (mut p, mut q) = (1usize, 0usize);
    let mut x = value;

    for _ in 0..64 {
        let a = x.floor();
        let ai = a as usize;

        let p_next = ai.saturating_mul(p).saturating_add(p_prev);
        let q_next = ai.saturating_mul(q).saturating_add(q_prev);
        if q_next > max_denominator || q_next == 0 {
            break;
        }

        p_prev = p;
        q_prev = q;
        p = p_next;
        q = q_next;

        let frac = x - a;
        if frac < 1e-12 {
            break;
        }
        x = 1.0 / frac;
    }

    if p == 0 || q == 0 {
        // The value rounds to zero at this denominator bound; fall back to nearest integer.
        return (value.round().max(1.0) as usize, 1);
    }
    reduce(p, q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    fn tone(n: usize, freq: f32, rate: f32) -> Vec<f32> {
        (0..n)
            .map(|k| (TAU * freq * k as f32 / rate).sin())
            .collect()
    }

    fn peak(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    }

    #[test]
    fn gcd_reduces_ratios() {
        assert_eq!(reduce(240_000, 48_000), (5, 1));
        assert_eq!(reduce(48_000, 44_100), (160, 147));
        assert_eq!(reduce(7, 3), (7, 3));
    }

    #[test]
    fn rational_approximation_finds_exact_ratios() {
        assert_eq!(rational_approximation(0.5, 1000), (1, 2));
        assert_eq!(rational_approximation(2.0, 1000), (2, 1));
        // The classic CD to DAT conversion.
        assert_eq!(
            rational_approximation(48_000.0 / 44_100.0, 1000),
            (160, 147)
        );
    }

    #[test]
    fn rational_approximation_respects_the_denominator_bound() {
        let (_, q) = rational_approximation(core::f64::consts::PI, 100);
        assert!(q <= 100, "denominator {q} exceeded the bound");
        // 355/113 is the famous approximation; with a bound of 100 we should get 22/7.
        let (p, q) = rational_approximation(core::f64::consts::PI, 50);
        assert_eq!((p, q), (22, 7));
    }

    #[test]
    fn pure_decimation_produces_the_expected_count() {
        let mut r = RationalResampler::new(1, 5, 8);
        assert_eq!((r.interp(), r.decim()), (1, 5));
        let input = tone(1000, 100.0, 48_000.0);
        let mut out = vec![0.0; r.output_len(1000)];
        let n = r.process(&input, &mut out);
        assert!(
            (n as i32 - 200).abs() <= 1,
            "expected about 200 outputs, got {n}"
        );
    }

    #[test]
    fn pure_interpolation_produces_the_expected_count() {
        let mut r = RationalResampler::new(4, 1, 8);
        let input = tone(250, 100.0, 8_000.0);
        let mut out = vec![0.0; r.output_len(250)];
        let n = r.process(&input, &mut out);
        assert!(
            (n as i32 - 1000).abs() <= 1,
            "expected about 1000 outputs, got {n}"
        );
    }

    #[test]
    fn output_rate_stays_exact_across_many_blocks() {
        // The per-call count varies by one; what must hold is that the total does not drift.
        let mut r = RationalResampler::new(160, 147, 8);
        let block = 512;
        let blocks = 200;
        let input = tone(block, 440.0, 44_100.0);
        let mut out = vec![0.0; r.output_len(block)];

        let mut total = 0usize;
        for _ in 0..blocks {
            total += r.process(&input, &mut out);
        }

        let expected = (block * blocks) as f64 * 160.0 / 147.0;
        let drift = (total as f64 - expected).abs();
        assert!(
            drift <= 2.0,
            "output drifted by {drift} samples over {blocks} blocks"
        );
    }

    #[test]
    fn preserves_a_tone_well_inside_the_passband() {
        let rate_in = 48_000.0;
        let mut r = RationalResampler::new(1, 2, 16);
        let input = tone(8192, 1000.0, rate_in);
        let mut out = vec![0.0; r.output_len(8192)];
        let n = r.process(&input, &mut out);

        // Skip the filter's start-up transient before measuring.
        let settled = &out[64..n];
        assert!(
            (peak(settled) - 1.0).abs() < 0.05,
            "amplitude not preserved: peak {}",
            peak(settled)
        );
    }

    #[test]
    fn interpolation_preserves_amplitude() {
        // Without the per-branch gain scaling this comes out a factor of `interp` low.
        let mut r = RationalResampler::new(3, 1, 16);
        let input = tone(4096, 500.0, 48_000.0);
        let mut out = vec![0.0; r.output_len(4096)];
        let n = r.process(&input, &mut out);
        let settled = &out[128..n];
        assert!(
            (peak(settled) - 1.0).abs() < 0.06,
            "interpolated amplitude {} should stay near unity",
            peak(settled)
        );
    }

    #[test]
    fn rejects_content_above_the_new_nyquist_limit() {
        // 20 kHz cannot survive a drop to 16 kHz, and must be filtered rather than folded
        // back down into the audible band.
        let mut r = RationalResampler::new(1, 3, 16);
        let input = tone(8192, 20_000.0, 48_000.0);
        let mut out = vec![0.0; r.output_len(8192)];
        let n = r.process(&input, &mut out);

        let settled = &out[128..n];
        assert!(
            peak(settled) < 0.05,
            "aliased energy got through: peak {}",
            peak(settled)
        );
    }

    #[test]
    fn block_boundaries_do_not_disturb_the_output() {
        let mut whole = RationalResampler::new(2, 3, 16);
        let mut split = RationalResampler::new(2, 3, 16);

        let input = tone(1024, 700.0, 48_000.0);
        let mut out_whole = vec![0.0; whole.output_len(1024)];
        let n_whole = whole.process(&input, &mut out_whole);

        let mut out_split = vec![0.0; split.output_len(1024) * 2];
        let n1 = split.process(&input[..300], &mut out_split);
        let n2 = split.process(&input[300..700], &mut out_split[n1..]);
        let n3 = split.process(&input[700..], &mut out_split[n1 + n2..]);

        assert_eq!(
            n_whole,
            n1 + n2 + n3,
            "total count changed when blocked differently"
        );
        for k in 0..n_whole {
            assert!(
                (out_whole[k] - out_split[k]).abs() < 1e-5,
                "sample {k} differs across block boundaries"
            );
        }
    }

    #[test]
    fn for_rates_picks_the_right_ratio() {
        let r = RationalResampler::for_rates(240_000.0, 48_000.0, 8);
        assert_eq!((r.interp(), r.decim()), (1, 5));
        assert!((r.output_rate(240_000.0) - 48_000.0).abs() < 1e-6);
    }

    #[test]
    fn reset_clears_the_filter_tail() {
        let mut r = RationalResampler::new(1, 2, 8);
        let mut loud = vec![0.0f32; 64];
        loud[0] = 100.0;
        let mut out = vec![0.0; r.output_len(64)];
        r.process(&loud, &mut out);

        r.reset();
        let n = r.process(&vec![0.0; 64], &mut out);
        for k in 0..n {
            assert!(out[k].abs() < 1e-9, "residual at {k}: {}", out[k]);
        }
    }

    #[test]
    #[should_panic(expected = "rates must be at least 1")]
    fn rejects_a_zero_rate() {
        RationalResampler::new(0, 1, 8);
    }
}
