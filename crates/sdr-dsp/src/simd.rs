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

