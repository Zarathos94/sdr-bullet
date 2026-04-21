//! A four-wide f32 vector with three backends.
//!
//! WebAssembly has no runtime feature detection — a module either requires SIMD support
//! from the engine or it doesn't. That rules out the usual `is_x86_feature_detected!`
//! dispatch and means the backend has to be chosen at compile time.
//!
//! Keeping an SSE2 backend alongside the wasm one is what makes the test suite worth
//! anything: SSE2 is baseline on x86_64, so `cargo test` exercises a genuine vector path
//! rather than only the scalar fallback, and [`super::tests`] can assert the two agree.
//!
//! Everything downstream works on deinterleaved I/Q (separate slices), which is why this
//! type needs no lane shuffles. Complex arithmetic on split arrays is just parallel real
//! arithmetic. The single deinterleave happens once, in [`crate::iq`].

use core::ops::{Add, Div, Mul, Sub};

pub const LANES: usize = 4;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x4(Repr);

// ---------------------------------------------------------------------------
// wasm32 + simd128
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::arch::wasm32::*;

    pub type Repr = v128;

    #[inline(always)]
    pub fn splat(v: f32) -> Repr {
        f32x4_splat(v)
    }

    /// # Safety
    /// `p` must be valid for reads of 4 `f32`. Alignment is not required.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> Repr {
        unsafe { v128_load(p as *const v128) }
    }

    /// # Safety
    /// `p` must be valid for writes of 4 `f32`. Alignment is not required.
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: Repr) {
        unsafe { v128_store(p as *mut v128, v) }
    }

    #[inline(always)]
    pub fn add(a: Repr, b: Repr) -> Repr {
        f32x4_add(a, b)
    }
    #[inline(always)]
    pub fn sub(a: Repr, b: Repr) -> Repr {
        f32x4_sub(a, b)
    }
    #[inline(always)]
    pub fn mul(a: Repr, b: Repr) -> Repr {
        f32x4_mul(a, b)
    }
    #[inline(always)]
    pub fn div(a: Repr, b: Repr) -> Repr {
        f32x4_div(a, b)
    }
    #[inline(always)]
    pub fn min(a: Repr, b: Repr) -> Repr {
        f32x4_pmin(a, b)
    }
    #[inline(always)]
    pub fn max(a: Repr, b: Repr) -> Repr {
        f32x4_pmax(a, b)
    }
    #[inline(always)]
    pub fn sqrt(a: Repr) -> Repr {
        f32x4_sqrt(a)
    }
    #[inline(always)]
    pub fn abs(a: Repr) -> Repr {
        f32x4_abs(a)
    }

    #[inline(always)]
    pub fn hsum(a: Repr) -> f32 {
        // Pairwise rather than a serial chain: shorter dependency graph, and the
        // rounding matches the SSE2 backend's shuffle-based reduction.
        let hi = i32x4_shuffle::<2, 3, 0, 0>(a, a);
        let s = f32x4_add(a, hi);
        let hi2 = i32x4_shuffle::<1, 0, 0, 0>(s, s);
        f32x4_extract_lane::<0>(f32x4_add(s, hi2))
    }

    #[inline(always)]
    pub fn extract(a: Repr, i: usize) -> f32 {
        match i {
            0 => f32x4_extract_lane::<0>(a),
            1 => f32x4_extract_lane::<1>(a),
            2 => f32x4_extract_lane::<2>(a),
            _ => f32x4_extract_lane::<3>(a),
        }
    }
}

// ---------------------------------------------------------------------------
// x86_64 SSE2 (baseline for the target, so no detection needed)
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
mod imp {
    use core::arch::x86_64::*;

    pub type Repr = __m128;

    #[inline(always)]
    pub fn splat(v: f32) -> Repr {
        unsafe { _mm_set1_ps(v) }
    }

    /// # Safety
    /// `p` must be valid for reads of 4 `f32`. Alignment is not required.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> Repr {
        unsafe { _mm_loadu_ps(p) }
    }

    /// # Safety
    /// `p` must be valid for writes of 4 `f32`. Alignment is not required.
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: Repr) {
        unsafe { _mm_storeu_ps(p, v) }
    }

    #[inline(always)]
    pub fn add(a: Repr, b: Repr) -> Repr {
        unsafe { _mm_add_ps(a, b) }
    }
    #[inline(always)]
    pub fn sub(a: Repr, b: Repr) -> Repr {
        unsafe { _mm_sub_ps(a, b) }
    }
    #[inline(always)]
    pub fn mul(a: Repr, b: Repr) -> Repr {
        unsafe { _mm_mul_ps(a, b) }
    }
    #[inline(always)]
    pub fn div(a: Repr, b: Repr) -> Repr {
        unsafe { _mm_div_ps(a, b) }
    }
    #[inline(always)]
    pub fn min(a: Repr, b: Repr) -> Repr {
        unsafe { _mm_min_ps(a, b) }
    }
    #[inline(always)]
    pub fn max(a: Repr, b: Repr) -> Repr {
        unsafe { _mm_max_ps(a, b) }
    }
    #[inline(always)]
    pub fn sqrt(a: Repr) -> Repr {
        unsafe { _mm_sqrt_ps(a) }
    }

    #[inline(always)]
    pub fn abs(a: Repr) -> Repr {
        // Clear the sign bit rather than comparing; branch-free and exact.
        unsafe { _mm_andnot_ps(_mm_set1_ps(-0.0), a) }
    }

    #[inline(always)]
    pub fn hsum(a: Repr) -> f32 {
        unsafe {
            let hi = _mm_movehl_ps(a, a);
            let s = _mm_add_ps(a, hi);
            let hi2 = _mm_shuffle_ps(s, s, 0x55);
            _mm_cvtss_f32(_mm_add_ss(s, hi2))
        }
    }

    #[inline(always)]
    pub fn extract(a: Repr, i: usize) -> f32 {
        let mut out = [0.0f32; 4];
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), a) };
        out[i]
    }
}

// ---------------------------------------------------------------------------
// Portable fallback
// ---------------------------------------------------------------------------

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "x86_64"
)))]
mod imp {
    pub type Repr = [f32; 4];

    #[inline(always)]
    pub fn splat(v: f32) -> Repr {
        [v; 4]
    }

    /// # Safety
    /// `p` must be valid for reads of 4 `f32`.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> Repr {
        unsafe { [*p, *p.add(1), *p.add(2), *p.add(3)] }
    }

    /// # Safety
    /// `p` must be valid for writes of 4 `f32`.
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: Repr) {
        unsafe {
            *p = v[0];
            *p.add(1) = v[1];
            *p.add(2) = v[2];
            *p.add(3) = v[3];
        }
    }

    macro_rules! lanewise {
        ($name:ident, $op:expr) => {
            #[inline(always)]
            pub fn $name(a: Repr, b: Repr) -> Repr {
                let f: fn(f32, f32) -> f32 = $op;
                [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
            }
        };
    }

    lanewise!(add, |x, y| x + y);
    lanewise!(sub, |x, y| x - y);
    lanewise!(mul, |x, y| x * y);
    lanewise!(div, |x, y| x / y);
    lanewise!(min, |x: f32, y: f32| if x < y { x } else { y });
    lanewise!(max, |x: f32, y: f32| if x > y { x } else { y });

    #[inline(always)]
    pub fn sqrt(a: Repr) -> Repr {
        [a[0].sqrt(), a[1].sqrt(), a[2].sqrt(), a[3].sqrt()]
    }
    #[inline(always)]
    pub fn abs(a: Repr) -> Repr {
        [a[0].abs(), a[1].abs(), a[2].abs(), a[3].abs()]
    }

    #[inline(always)]
    pub fn hsum(a: Repr) -> f32 {
        // Same pairwise association as the vector backends so results match bit for bit.
        (a[0] + a[2]) + (a[1] + a[3])
    }

    #[inline(always)]
    pub fn extract(a: Repr, i: usize) -> f32 {
        a[i]
    }
}

use imp::Repr;

impl F32x4 {
    pub const LANES: usize = LANES;

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(imp::splat(v))
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Loads four lanes starting at `slice[offset]`.
    ///
    /// # Panics
    /// If fewer than four elements remain after `offset`.
    #[inline(always)]
    pub fn load(slice: &[f32], offset: usize) -> Self {
        assert!(offset + LANES <= slice.len(), "F32x4::load out of bounds");
        // SAFETY: bounds checked directly above.
        Self(unsafe { imp::load(slice.as_ptr().add(offset)) })
    }

    /// Loads four lanes without bounds checking.
    ///
    /// # Safety
    /// `offset + 4 <= slice.len()`.
    #[inline(always)]
    pub unsafe fn load_unchecked(slice: &[f32], offset: usize) -> Self {
        Self(unsafe { imp::load(slice.as_ptr().add(offset)) })
    }

    /// Stores four lanes starting at `slice[offset]`.
    ///
    /// # Panics
    /// If fewer than four elements remain after `offset`.
    #[inline(always)]
    pub fn store(self, slice: &mut [f32], offset: usize) {
        assert!(offset + LANES <= slice.len(), "F32x4::store out of bounds");
        // SAFETY: bounds checked directly above.
        unsafe { imp::store(slice.as_mut_ptr().add(offset), self.0) }
    }

    /// Stores four lanes without bounds checking.
    ///
    /// # Safety
    /// `offset + 4 <= slice.len()`.
    #[inline(always)]
    pub unsafe fn store_unchecked(self, slice: &mut [f32], offset: usize) {
        unsafe { imp::store(slice.as_mut_ptr().add(offset), self.0) }
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(imp::min(self.0, other.0))
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(imp::max(self.0, other.0))
    }
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(imp::sqrt(self.0))
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(imp::abs(self.0))
    }

    /// Fused-looking multiply-add. Baseline wasm SIMD has no FMA — that lives in the
    /// relaxed-SIMD proposal, which is deliberately non-deterministic across hardware and
    /// still gated in Safari — so this is a plain multiply followed by an add on every
    /// backend. Keeping it as one call means the choice is in one place if that changes.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(imp::add(imp::mul(self.0, a.0), b.0))
    }

    #[inline(always)]
    pub fn sum(self) -> f32 {
        imp::hsum(self.0)
    }

    #[inline(always)]
    pub fn extract(self, lane: usize) -> f32 {
        debug_assert!(lane < LANES);
        imp::extract(self.0, lane)
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; LANES] {
        let mut out = [0.0; LANES];
        // SAFETY: `out` is exactly four f32 wide.
        unsafe { imp::store(out.as_mut_ptr(), self.0) };
        out
    }

    #[inline(always)]
    pub fn from_array(a: [f32; LANES]) -> Self {
        // SAFETY: `a` is exactly four f32 wide.
        Self(unsafe { imp::load(a.as_ptr()) })
    }
}

impl Add for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(imp::add(self.0, rhs.0))
    }
}

impl Sub for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(imp::sub(self.0, rhs.0))
    }
}

impl Mul for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(imp::mul(self.0, rhs.0))
    }
}

impl Div for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(imp::div(self.0, rhs.0))
    }
}

impl core::fmt::Debug for F32x4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("F32x4").field(&self.to_array()).finish()
    }
}

/// Name of the active backend. Surfaced in the UI's diagnostics panel so a browser
/// silently falling back to scalar code is visible rather than merely slow.
pub const fn backend() -> &'static str {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        "wasm-simd128"
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    {
        "x86_64-sse2"
    }
    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "x86_64"
    )))]
    {
        "scalar"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_matches_scalar() {
        let a = [1.5f32, -2.25, 3.0, 0.5];
        let b = [0.25f32, 4.0, -1.5, 2.0];
        let va = F32x4::from_array(a);
        let vb = F32x4::from_array(b);

        for (i, v) in (va + vb).to_array().iter().enumerate() {
            assert_eq!(*v, a[i] + b[i]);
        }
        for (i, v) in (va - vb).to_array().iter().enumerate() {
            assert_eq!(*v, a[i] - b[i]);
        }
        for (i, v) in (va * vb).to_array().iter().enumerate() {
            assert_eq!(*v, a[i] * b[i]);
        }
        for (i, v) in (va / vb).to_array().iter().enumerate() {
            assert_eq!(*v, a[i] / b[i]);
        }
    }

    #[test]
    fn hsum_uses_pairwise_association() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(F32x4::from_array(a).sum(), 10.0);

        // The association order is load-bearing for reproducibility across backends:
        // (a0+a2) + (a1+a3), not a serial left fold.
        let b = [1e8f32, 1.0, -1e8, 1.0];
        assert_eq!(F32x4::from_array(b).sum(), (1e8 + -1e8) + (1.0 + 1.0));
    }

    #[test]
    fn min_max_and_abs() {
        let a = F32x4::from_array([1.0, -2.0, 3.0, -4.0]);
        let b = F32x4::from_array([-1.0, 2.0, -3.0, 4.0]);
        assert_eq!(a.min(b).to_array(), [-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(a.max(b).to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.abs().to_array(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn mul_add_composes() {
        let a = F32x4::splat(2.0);
        let b = F32x4::splat(3.0);
        let c = F32x4::splat(1.0);
        assert_eq!(a.mul_add(b, c).to_array(), [7.0; 4]);
    }

    #[test]
    fn load_store_roundtrip_at_offset() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 16];
        for off in [0usize, 1, 5, 12] {
            F32x4::load(&src, off).store(&mut dst, off);
            assert_eq!(&dst[off..off + 4], &src[off..off + 4]);
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn load_past_end_panics() {
        let src = [0.0f32; 6];
        let _ = F32x4::load(&src, 3);
    }
}
