//! FIR filtering, decimation, and the filter designs the receive chain needs.
//!
//! All three filters share one structure: carry `ntaps - 1` samples of history, prepend it
//! to the incoming block, and slide a dot product along the result. Working over one
//! contiguous scratch buffer rather than a circular history means the inner loop is a
//! straight-line vector reduction with no index wrapping, which is what makes it fast.
//!
//! Taps are stored in reverse. Convolution reads history backwards, and reversing once at
//! construction turns the inner loop into a forward walk over both arrays. For the
//! symmetric low-pass designs this is a no-op, but the Hilbert transformer is
//! antisymmetric and would come out sign-flipped otherwise.

use crate::simd::{F32x4, LANES};
use crate::window::Window;

/// Dot product of two equal-length slices.
///
/// Four independent accumulators keep four multiply-add chains in flight, so the loop is
/// bound by throughput rather than by the latency of a single dependency chain.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let blocks = n / (LANES * 4);

    let mut acc0 = F32x4::zero();
    let mut acc1 = F32x4::zero();
    let mut acc2 = F32x4::zero();
    let mut acc3 = F32x4::zero();

    for k in 0..blocks {
        let off = k * LANES * 4;
        // SAFETY: `off + 4*LANES <= blocks * 4 * LANES <= n`, and both slices are `n` long.
        unsafe {
            acc0 = F32x4::load_unchecked(a, off).mul_add(F32x4::load_unchecked(b, off), acc0);
            acc1 = F32x4::load_unchecked(a, off + LANES)
                .mul_add(F32x4::load_unchecked(b, off + LANES), acc1);
            acc2 = F32x4::load_unchecked(a, off + 2 * LANES)
                .mul_add(F32x4::load_unchecked(b, off + 2 * LANES), acc2);
            acc3 = F32x4::load_unchecked(a, off + 3 * LANES)
                .mul_add(F32x4::load_unchecked(b, off + 3 * LANES), acc3);
        }
    }

    let mut done = blocks * LANES * 4;
    let mut acc = (acc0 + acc1) + (acc2 + acc3);
    while done + LANES <= n {
        // SAFETY: guarded by the loop condition.
        unsafe {
            acc = F32x4::load_unchecked(a, done).mul_add(F32x4::load_unchecked(b, done), acc);
        }
        done += LANES;
    }

    let mut sum = acc.sum();
    for k in done..n {
        sum += a[k] * b[k];
    }
    sum
}

/// Streaming FIR filter.
#[derive(Debug, Clone)]
pub struct Fir {
    /// Taps in reverse order, so the inner loop walks forward over both arrays.
    taps: Vec<f32>,
    /// The previous block's trailing `ntaps - 1` samples.
    history: Vec<f32>,
    scratch: Vec<f32>,
}

impl Fir {
    /// # Panics
    /// If `taps` is empty.
    pub fn new(taps: Vec<f32>) -> Self {
        assert!(!taps.is_empty(), "a filter needs at least one tap");
        let history = vec![0.0; taps.len() - 1];
        let mut reversed = taps;
        reversed.reverse();
        Self {
            taps: reversed,
            history,
            scratch: Vec::new(),
        }
    }

    pub fn taps(&self) -> usize {
        self.taps.len()
    }

    /// Group delay in samples. Linear-phase designs delay by half the tap count.
    pub fn delay(&self) -> usize {
        (self.taps.len() - 1) / 2
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
    }

    /// Filters `input` into `output`, one output sample per input sample.
    ///
    /// # Panics
    /// If `output` is shorter than `input`.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= input.len(), "output buffer too short");
        let ntaps = self.taps.len();
        let hist = ntaps - 1;

        self.scratch.clear();
        self.scratch.extend_from_slice(&self.history);
        self.scratch.extend_from_slice(input);

        for k in 0..input.len() {
            output[k] = dot(&self.scratch[k..k + ntaps], &self.taps);
        }

        let tail = self.scratch.len().saturating_sub(hist);
        self.history.clear();
        self.history.extend_from_slice(&self.scratch[tail..]);
        // A block shorter than the history leaves it under-filled; pad at the front so the
        // filter state stays exactly `ntaps - 1` long.
        while self.history.len() < hist {
            self.history.insert(0, 0.0);
        }
    }
}

/// Decimating FIR: filters, then keeps one sample in `factor`.
///
/// Only the retained outputs are computed. A decimate-by-eight stage therefore costs an
/// eighth of what filtering and then discarding would.
#[derive(Debug, Clone)]
pub struct Decimator {
    taps: Vec<f32>,
    history: Vec<f32>,
    scratch: Vec<f32>,
    factor: usize,
    /// Input samples still owed before the next output is due, carried across blocks.
    phase: usize,
}

impl Decimator {
    /// # Panics
    /// If `taps` is empty or `factor` is zero.
    pub fn new(taps: Vec<f32>, factor: usize) -> Self {
        assert!(!taps.is_empty(), "a filter needs at least one tap");
        assert!(factor >= 1, "decimation factor must be at least 1");
        let history = vec![0.0; taps.len() - 1];
        let mut reversed = taps;
        reversed.reverse();
        Self {
            taps: reversed,
            history,
            scratch: Vec::new(),
            factor,
            phase: 0,
        }
    }

    /// Low-pass at the new Nyquist limit, then decimate.
    ///
    /// The cutoff is set to `0.4 / factor` rather than the theoretical `0.5 / factor`,
    /// leaving a transition band that reaches the stopband before the fold point. Filtering
    /// right up to Nyquist would alias the transition band back over the passband.
    ///
    /// A factor of one is a valid degenerate case — a channel filter that keeps every
    /// sample. It arises when the capture rate already equals the wanted channel rate, and
    /// the caller should not have to special-case it. The cutoff then sits at 0.4 of the
    /// sample rate, cleaning up the band edges without changing the rate.
    ///
    /// # Panics
    /// If `factor` is zero.
    pub fn lowpass(factor: usize, taps: usize) -> Self {
        assert!(factor >= 1, "decimation factor must be at least 1");
        let cutoff = 0.4 / factor as f32;
        Self::new(design_lowpass(taps, cutoff, Window::kaiser(8.6)), factor)
    }

    pub fn factor(&self) -> usize {
        self.factor
    }

    pub fn taps(&self) -> usize {
        self.taps.len()
    }

    /// Upper bound on outputs produced by an input block of `n` samples.
    pub fn output_len(&self, n: usize) -> usize {
        n.div_ceil(self.factor)
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.phase = 0;
    }

    /// Returns the number of output samples written.
    ///
    /// # Panics
    /// If `output` cannot hold [`Decimator::output_len`] samples.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        assert!(
            output.len() >= self.output_len(input.len()),
            "output buffer too short"
        );
        let ntaps = self.taps.len();
        let hist = ntaps - 1;

        self.scratch.clear();
        self.scratch.extend_from_slice(&self.history);
        self.scratch.extend_from_slice(input);

        let mut written = 0;
        let mut k = self.phase;
        while k < input.len() {
            output[written] = dot(&self.scratch[k..k + ntaps], &self.taps);
            written += 1;
            k += self.factor;
        }
        // Carry the leftover step into the next block so the output rate stays exact
        // across block boundaries rather than resetting on every call.
        self.phase = k - input.len();

        let tail = self.scratch.len().saturating_sub(hist);
        self.history.clear();
        self.history.extend_from_slice(&self.scratch[tail..]);
        while self.history.len() < hist {
            self.history.insert(0, 0.0);
        }

        written
    }
}

/// Half-band decimate-by-two stage.
///
/// A half-band filter has every second tap equal to zero either side of the centre, and a
/// centre tap of exactly one half. Skipping the zeros halves the multiply count, which is
/// why a cascade of these is the cheapest way to drop a wideband capture down to a channel
/// rate.
#[derive(Debug, Clone)]
pub struct HalfBand {
    /// Only the nonzero off-centre taps, in reverse order.
    odd_taps: Vec<f32>,
    centre: f32,
    /// Distance from the first tap to the centre, in samples.
    half_span: usize,
    history: Vec<f32>,
    scratch: Vec<f32>,
    ntaps: usize,
    phase: usize,
}

impl HalfBand {
    /// Builds a half-band stage from a design of `taps` length.
    ///
    /// # Panics
    /// If `taps` is not of the form `4k + 3`, which is what makes the zero pattern land
    /// correctly around an integer centre.
    pub fn new(taps: usize) -> Self {
        assert!(
            taps >= 7 && taps % 4 == 3,
            "half-band length must be 4k+3, got {taps}"
        );
        let full = design_halfband(taps);
        let centre_idx = taps / 2;
        let centre = full[centre_idx];

        // Taps at odd offsets from the centre are the only nonzero ones off-centre.
        let mut odd_taps = Vec::new();
        let mut idx = 1usize;
        while idx <= centre_idx {
            if idx % 2 == 1 {
                odd_taps.push(full[centre_idx - idx]);
            }
            idx += 1;
        }
        // Reverse so the loop walks the scratch buffer forward.
        odd_taps.reverse();

        Self {
            odd_taps,
            centre,
            half_span: centre_idx,
            history: vec![0.0; taps - 1],
            scratch: Vec::new(),
            ntaps: taps,
            phase: 0,
        }
    }

    pub fn taps(&self) -> usize {
        self.ntaps
    }

    pub fn output_len(&self, n: usize) -> usize {
        n.div_ceil(2)
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.phase = 0;
    }

    /// Returns the number of output samples written.
    ///
    /// # Panics
    /// If `output` cannot hold [`HalfBand::output_len`] samples.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        assert!(
            output.len() >= self.output_len(input.len()),
            "output buffer too short"
        );
        let hist = self.ntaps - 1;

        self.scratch.clear();
        self.scratch.extend_from_slice(&self.history);
        self.scratch.extend_from_slice(input);

        let mut written = 0;
        let mut k = self.phase;
        while k < input.len() {
            let w = &self.scratch[k..k + self.ntaps];
            // Pair up samples equidistant from the centre: the filter is symmetric, so
            // each tap multiplies a sum rather than two separate products.
            let mut acc = self.centre * w[self.half_span];
            for (t, tap) in self.odd_taps.iter().enumerate() {
                // odd_taps was reversed, so index t counts inwards from the outermost tap.
                let offset = self.half_span - (2 * (self.odd_taps.len() - 1 - t) + 1);
                acc += tap * (w[offset] + w[self.ntaps - 1 - offset]);
            }
            output[written] = acc;
            written += 1;
            k += 2;
        }
        self.phase = k - input.len();

        let tail = self.scratch.len().saturating_sub(hist);
        self.history.clear();
        self.history.extend_from_slice(&self.scratch[tail..]);
        while self.history.len() < hist {
            self.history.insert(0, 0.0);
        }

        written
    }
}

// ---------------------------------------------------------------------------
// Filter design
// ---------------------------------------------------------------------------

/// Windowed-sinc low-pass. `cutoff` is in cycles per sample, so 0.5 is Nyquist.
///
/// Taps are normalised for unit gain at DC, which keeps a cascade of stages from
/// accumulating a level change.
///
/// # Panics
/// If `taps` is even or zero, or `cutoff` is outside `(0, 0.5)`.
pub fn design_lowpass(taps: usize, cutoff: f32, window: Window) -> Vec<f32> {
    assert!(
        taps % 2 == 1 && taps > 0,
        "use an odd tap count for a linear-phase filter"
    );
    assert!(
        cutoff > 0.0 && cutoff < 0.5,
        "cutoff must be in (0, 0.5), got {cutoff}"
    );

    let w = window.symmetric(taps);
    let centre = (taps / 2) as f32;
    let mut h: Vec<f32> = (0..taps)
        .map(|k| {
            let x = k as f32 - centre;
            let ideal = if x.abs() < 1e-6 {
                // sinc's removable singularity at zero.
                2.0 * cutoff
            } else {
                (core::f32::consts::TAU * cutoff * x).sin() / (core::f32::consts::PI * x)
            };
            ideal * w[k]
        })
        .collect();

    let sum: f32 = h.iter().sum();
    if sum.abs() > 1e-12 {
        for v in &mut h {
            *v /= sum;
        }
    }
    h
}

/// Half-band low-pass: cutoff fixed at a quarter of the sample rate.
///
/// The zeros are forced exactly rather than left to rounding, because [`HalfBand`] skips
/// those positions entirely and a residual value there would silently be dropped.
///
/// # Panics
/// If `taps` is not of the form `4k + 3`.
pub fn design_halfband(taps: usize) -> Vec<f32> {
    assert!(taps % 4 == 3, "half-band length must be 4k+3, got {taps}");
    let mut h = design_lowpass(taps, 0.25, Window::kaiser(8.6));
    let centre = taps / 2;
    for (k, v) in h.iter_mut().enumerate() {
        let offset = k as isize - centre as isize;
        if offset != 0 && offset % 2 == 0 {
            *v = 0.0;
        }
    }
    // Renormalise, since zeroing taps disturbs the DC gain.
    let sum: f32 = h.iter().sum();
    for v in &mut h {
        *v /= sum;
    }
    h
}

/// Band-pass built by modulating a low-pass prototype up to the band centre.
///
/// # Panics
/// If the band is not inside `(0, 0.5)` or `low >= high`.
pub fn design_bandpass(taps: usize, low: f32, high: f32, window: Window) -> Vec<f32> {
    assert!(low < high, "low cutoff must be below high cutoff");
    assert!(low > 0.0 && high < 0.5, "band must lie inside (0, 0.5)");

    let centre_freq = (low + high) / 2.0;
    let half_width = (high - low) / 2.0;
    let proto = design_lowpass(taps, half_width, window);
    let centre = (taps / 2) as f32;

    // Multiplying by a cosine shifts the prototype's response to sit either side of it.
    // The factor of two restores the gain that splitting into two images halves.
    proto
        .iter()
        .enumerate()
        .map(|(k, v)| {
            let x = k as f32 - centre;
            2.0 * v * (core::f32::consts::TAU * centre_freq * x).cos()
        })
        .collect()
}

/// Hilbert transformer: a 90-degree phase shift across most of the band.
///
/// Antisymmetric, with zeros at every even offset from the centre. Pair it with a delay of
/// `taps / 2` on the in-phase branch to build an analytic signal.
///
/// # Panics
/// If `taps` is even.
pub fn design_hilbert(taps: usize, window: Window) -> Vec<f32> {
    assert!(taps % 2 == 1, "Hilbert transformer needs an odd tap count");
    let w = window.symmetric(taps);
    let centre = (taps / 2) as isize;
    (0..taps)
        .map(|k| {
            let n = k as isize - centre;
            if n == 0 || n % 2 == 0 {
                0.0
            } else {
                (2.0 / (core::f32::consts::PI * n as f32)) * w[k]
            }
        })
        .collect()
}

/// Response of a filter at a normalised frequency, in cycles per sample.
///
/// Used by the tests to assert passband and stopband behaviour rather than eyeballing taps.
pub fn response_at(taps: &[f32], freq: f32) -> f32 {
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (k, tap) in taps.iter().enumerate() {
        let ang = -core::f64::consts::TAU * freq as f64 * k as f64;
        re += *tap as f64 * ang.cos();
        im += *tap as f64 * ang.sin();
    }
    (re * re + im * im).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db(x: f32) -> f32 {
        20.0 * (x.max(1e-12)).log10()
    }

    #[test]
    fn dot_matches_the_naive_sum_at_every_length() {
        // Lengths chosen to straddle the 16-wide unrolled block and the 4-wide remainder.
        for n in [1usize, 3, 4, 5, 15, 16, 17, 31, 32, 33, 64, 100] {
            let a: Vec<f32> = (0..n).map(|k| (k as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..n).map(|k| (k as f32 * 0.11).cos()).collect();
            let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let got = dot(&a, &b);
            assert!(
                (got - want).abs() < 1e-3 * n as f32,
                "n={n}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn impulse_response_reproduces_the_taps() {
        let taps = vec![0.1, 0.2, 0.3, 0.25, 0.15];
        let mut f = Fir::new(taps.clone());
        let mut input = vec![0.0; 16];
        input[0] = 1.0;
        let mut out = vec![0.0; 16];
        f.process(&input, &mut out);

        for k in 0..taps.len() {
            assert!(
                (out[k] - taps[k]).abs() < 1e-6,
                "tap {k}: {} vs {}",
                out[k],
                taps[k]
            );
        }
        for v in &out[taps.len()..] {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn asymmetric_taps_are_not_reversed_by_the_filter() {
        // A palindrome would hide a tap-ordering bug, so use one that is not.
        let taps = vec![1.0, 2.0, 3.0];
        let mut f = Fir::new(taps.clone());
        let mut input = vec![0.0; 8];
        input[0] = 1.0;
        let mut out = vec![0.0; 8];
        f.process(&input, &mut out);
        assert_eq!(&out[..3], &taps[..]);
    }

    #[test]
    fn state_carries_across_block_boundaries() {
        let taps = design_lowpass(31, 0.1, Window::Hamming);
        let input: Vec<f32> = (0..256).map(|k| (k as f32 * 0.05).sin()).collect();

        let mut whole = vec![0.0; 256];
        Fir::new(taps.clone()).process(&input, &mut whole);

        let mut split = vec![0.0; 256];
        let mut f = Fir::new(taps);
        f.process(&input[..100], &mut split[..100]);
        f.process(&input[100..], &mut split[100..]);

        for k in 0..256 {
            assert!(
                (whole[k] - split[k]).abs() < 1e-6,
                "block boundary drift at {k}"
            );
        }
    }

    #[test]
    fn lowpass_passes_below_cutoff_and_rejects_above() {
        let taps = design_lowpass(101, 0.1, Window::kaiser(8.6));
        assert!(
            db(response_at(&taps, 0.0)).abs() < 0.1,
            "DC gain should be unity"
        );
        assert!(
            db(response_at(&taps, 0.05)) > -1.0,
            "passband droop too large"
        );
        // Kaiser with beta 8.6 gives roughly -90 dB; leave margin for the transition.
        assert!(
            db(response_at(&taps, 0.2)) < -60.0,
            "stopband not deep enough"
        );
        assert!(db(response_at(&taps, 0.4)) < -60.0);
    }

    #[test]
    fn lowpass_is_linear_phase() {
        let taps = design_lowpass(51, 0.15, Window::Hamming);
        for k in 0..51 {
            assert!((taps[k] - taps[50 - k]).abs() < 1e-7, "asymmetry at {k}");
        }
    }

    #[test]
    fn halfband_zeroes_alternate_taps() {
        let h = design_halfband(31);
        let centre = 15;
        for (k, v) in h.iter().enumerate() {
            let offset = k as isize - centre as isize;
            if offset != 0 && offset % 2 == 0 {
                assert_eq!(*v, 0.0, "tap {k} should be exactly zero");
            }
        }
        assert!(
            (h[centre] - 0.5).abs() < 0.02,
            "centre tap should be near 0.5"
        );
    }

    #[test]
    fn halfband_is_symmetric_about_quarter_rate() {
        let h = design_halfband(31);
        // Its defining property: the response at f and at 0.5 - f sum to unity.
        for f in [0.05f32, 0.1, 0.15, 0.2] {
            let sum = response_at(&h, f) + response_at(&h, 0.5 - f);
            assert!((sum - 1.0).abs() < 0.05, "at {f}: sum {sum}");
        }
    }

    #[test]
    fn decimator_halves_the_rate_and_keeps_the_phase_across_blocks() {
        let mut d = Decimator::lowpass(2, 31);
        let input: Vec<f32> = (0..100).map(|k| k as f32).collect();
        let mut out = vec![0.0; 64];

        // 100 samples at factor two yields 50 outputs, and an odd-length block must not
        // reset the phase or the following block would be off by one.
        let n1 = d.process(&input[..51], &mut out);
        let n2 = d.process(&input[51..], &mut out[n1..]);
        assert_eq!(n1 + n2, 50, "decimated count wrong: {n1} + {n2}");
    }

    #[test]
    fn decimator_by_one_is_a_pure_filter_that_keeps_every_sample() {
        // The degenerate case: capture rate already equals the wanted channel rate. It must
        // filter without dropping samples rather than panicking, so a pipeline that lands
        // on a decimation of one needs no special-casing.
        let mut d = Decimator::lowpass(1, 63);
        let input: Vec<f32> = (0..256).map(|k| (k as f32 * 0.02).sin()).collect();
        let mut out = vec![0.0; d.output_len(256)];
        let n = d.process(&input, &mut out);
        assert_eq!(n, 256, "a factor of one must keep every sample");
    }

    #[test]
    fn decimator_output_matches_filter_then_discard() {
        let taps = design_lowpass(31, 0.1, Window::Hamming);
        let input: Vec<f32> = (0..128).map(|k| (k as f32 * 0.07).sin()).collect();

        let mut filtered = vec![0.0; 128];
        Fir::new(taps.clone()).process(&input, &mut filtered);

        let mut decimated = vec![0.0; 32];
        let n = Decimator::new(taps, 4).process(&input, &mut decimated);

        assert_eq!(n, 32);
        for k in 0..32 {
            assert!(
                (decimated[k] - filtered[k * 4]).abs() < 1e-5,
                "sample {k}: {} vs {}",
                decimated[k],
                filtered[k * 4]
            );
        }
    }

    #[test]
    fn halfband_stage_matches_the_general_decimator() {
        let taps = design_halfband(31);
        let input: Vec<f32> = (0..256)
            .map(|k| (k as f32 * 0.03).sin() + 0.3 * (k as f32 * 0.31).cos())
            .collect();

        let mut want = vec![0.0; 128];
        let n_want = Decimator::new(taps, 2).process(&input, &mut want);

        let mut got = vec![0.0; 128];
        let n_got = HalfBand::new(31).process(&input, &mut got);

        assert_eq!(n_want, n_got);
        for k in 0..n_got {
            assert!(
                (got[k] - want[k]).abs() < 1e-5,
                "half-band shortcut disagrees at {k}: {} vs {}",
                got[k],
                want[k]
            );
        }
    }

    #[test]
    fn bandpass_passes_its_band_and_rejects_either_side() {
        let taps = design_bandpass(101, 0.1, 0.2, Window::kaiser(8.6));
        assert!(
            db(response_at(&taps, 0.15)) > -1.5,
            "centre of band attenuated"
        );
        assert!(db(response_at(&taps, 0.02)) < -40.0, "leakage below band");
        assert!(db(response_at(&taps, 0.35)) < -40.0, "leakage above band");
    }

    #[test]
    fn hilbert_is_antisymmetric_with_even_taps_zeroed() {
        let h = design_hilbert(31, Window::Hamming);
        let centre = 15;
        assert_eq!(h[centre], 0.0);
        for k in 0..31 {
            let offset = k as isize - centre as isize;
            if offset % 2 == 0 {
                assert_eq!(h[k], 0.0, "even offset {offset} should be zero");
            }
            assert!((h[k] + h[30 - k]).abs() < 1e-7, "not antisymmetric at {k}");
        }
    }

    #[test]
    fn hilbert_has_flat_response_across_the_middle_of_the_band() {
        let h = design_hilbert(101, Window::kaiser(8.0));
        for f in [0.1f32, 0.2, 0.3, 0.4] {
            let r = response_at(&h, f);
            assert!((r - 1.0).abs() < 0.05, "at {f}: response {r}");
        }
        // It cannot work at DC or Nyquist; those roll off by construction.
        assert!(response_at(&h, 0.0) < 0.01);
    }

    #[test]
    fn reset_clears_the_tail() {
        let mut f = Fir::new(design_lowpass(31, 0.2, Window::Hamming));
        let mut input = vec![0.0; 32];
        input[0] = 100.0;
        let mut out = vec![0.0; 32];
        f.process(&input, &mut out);

        f.reset();
        let zeros = vec![0.0; 32];
        f.process(&zeros, &mut out);
        for (k, v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-9, "residual energy at {k}: {v}");
        }
    }

    #[test]
    #[should_panic(expected = "4k+3")]
    fn halfband_rejects_a_bad_length() {
        HalfBand::new(32);
    }

    #[test]
    #[should_panic(expected = "cutoff must be in")]
    fn lowpass_rejects_a_cutoff_at_nyquist() {
        design_lowpass(31, 0.5, Window::Hamming);
    }
}
