//! Signal processing for a software defined radio receiver.
//!
//! The crate is deliberately free of I/O and of any wasm-specific dependency: everything
//! here runs identically under `cargo test` on the host and inside a browser worker. That
//! is what makes the test suite meaningful — the code paths are the same ones that ship.
//!
//! Buffers are deinterleaved throughout. The receiver hands over interleaved unsigned
//! bytes, [`iq::convert`] splits them into separate I and Q slices once, and every stage
//! after that treats complex arithmetic as parallel real arithmetic. This costs one pass
//! and removes lane shuffles from every subsequent kernel.

pub mod fft;
pub mod fir;
pub mod iq;
pub mod nco;
pub mod simd;
pub mod window;

pub use fft::Fft;
pub use fir::{Decimator, Fir, HalfBand};
pub use iq::IqConverter;
pub use nco::Nco;
pub use simd::F32x4;

// The wasm32 SIMD backend is selected by `cfg(target_feature = "simd128")`. If the
// rustflags in .cargo/config.toml ever stop applying, that cfg silently goes false and the
// scalar fallback gets picked instead — a large, invisible slowdown in a build that still
// succeeds. Fail the build instead.
#[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
compile_error!(
    "simd128 is not enabled for wasm32. Check .cargo/config.toml, and note that setting \
     the RUSTFLAGS environment variable replaces those flags rather than merging with them."
);
