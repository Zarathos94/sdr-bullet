//! Register-level driver for the RTL2832U demodulator and its R828D tuner.
//!
//! Written against the RTL-SDR Blog V4, which differs from a conventional R828D board in
//! three ways that all have to be handled together or the receiver simply does not work:
//! its tuner runs from the 28.8 MHz reference rather than a separate 16 MHz crystal, its
//! single antenna connector is triplexed into the tuner's three inputs, and everything
//! below 28.8 MHz arrives through a built-in upconverter instead of a direct-sampling tap.
//!
//! The driver is generic over a [`Transport`], so the same code runs against libusb on a
//! desktop and against WebUSB in a browser. That is not just tidiness: it means the
//! register sequences can be exercised against real hardware from a terminal, where a
//! failure is debuggable, before anything depends on them inside a worker.

pub mod r82xx;
pub mod regs;
pub mod rtl2832;
pub mod transport;

pub use r82xx::{GainMode, R82xx};
pub use regs::{Band, GainSetting, PllSetting, SampleRate};
pub use rtl2832::Rtl2832;
pub use transport::{ControlRequest, Direction, Transport};
