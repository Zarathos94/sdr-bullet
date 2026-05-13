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

