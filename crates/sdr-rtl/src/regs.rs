//! Register layout and the arithmetic that drives it.
//!
//! Everything here is a pure function of its inputs, which is deliberate: the parts of a
//! USB driver that are actually hard to get right are the divider calculations, and those
//! are the parts you cannot debug by staring at a bus trace. Keeping them separate from
//! the transport means they can be checked against worked values without any hardware
//! attached.

/// Crystal on the RTL2832U. The Blog V4 clocks its tuner from this same reference.
pub const RTL_XTAL: u32 = 28_800_000;

/// Frequency below which the Blog V4 routes through its built-in upconverter, and the
/// amount by which that upconverter shifts.
///
/// Both are the crystal frequency, because the upconverter's mixer is fed from the same
/// oscillator as everything else on the board.
pub const UPCONVERT_CROSSOVER: u32 = RTL_XTAL;

/// Boundary between the triplexer's VHF and UHF ports.
pub const VHF_UHF_BOUNDARY: u32 = 250_000_000;

/// Address spaces reachable over a control transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Block {
    Demod = 0,
    Usb = 1,
    Sys = 2,
    Tuner = 3,
    Rom = 4,
    InfraRed = 5,
    I2c = 6,
}

/// Marks a control transfer as a write rather than a read.
pub const WRITE_FLAG: u16 = 0x10;

/// USB block registers.
pub mod usb {
    pub const SYSCTL: u16 = 0x2000;
    pub const EPA_CTL: u16 = 0x2148;
    pub const EPA_MAXPKT: u16 = 0x2158;
}

/// System block registers.
pub mod sys {
    pub const DEMOD_CTL: u16 = 0x3000;
    pub const GPO: u16 = 0x3001;
    pub const GPD: u16 = 0x3002;
    pub const GPOE: u16 = 0x3003;
    pub const DEMOD_CTL_1: u16 = 0x300B;
}

/// `(wValue, wIndex)` for reading from one of the flat address blocks.
pub fn block_read(addr: u16, block: Block) -> (u16, u16) {
    (addr, (block as u16) << 8)
}

/// `(wValue, wIndex)` for writing to one of the flat address blocks.
pub fn block_write(addr: u16, block: Block) -> (u16, u16) {
    (addr, ((block as u16) << 8) | WRITE_FLAG)
}

/// `(wValue, wIndex)` for reading a demodulator register.
///
/// The demodulator uses a different encoding from every other block — the address moves
/// into the high byte of `wValue` and the page number takes over `wIndex`. Applying the
/// flat-block layout here reads a plausible-looking wrong register.
pub fn demod_read(page: u8, addr: u16) -> (u16, u16) {
    ((addr << 8) | 0x20, page as u16)
}

/// `(wValue, wIndex)` for writing a demodulator register.
pub fn demod_write(page: u8, addr: u16) -> (u16, u16) {
    ((addr << 8) | 0x20, WRITE_FLAG | page as u16)
}

/// `(wValue, wIndex)` for writing to a device on the tuner I2C bus.
pub fn i2c_write(i2c_addr: u8) -> (u16, u16) {
    (i2c_addr as u16, ((Block::I2c as u16) << 8) | WRITE_FLAG)
}

/// `(wValue, wIndex)` for reading from a device on the tuner I2C bus.
pub fn i2c_read(i2c_addr: u8) -> (u16, u16) {
    (i2c_addr as u16, (Block::I2c as u16) << 8)
}

/// Serialises a register value for the wire, most significant byte first.
pub fn encode_value(value: u16, len: usize) -> heapless_bytes::Bytes {
    match len {
        1 => heapless_bytes::Bytes::one((value & 0xFF) as u8),
        _ => heapless_bytes::Bytes::two((value >> 8) as u8, (value & 0xFF) as u8),
    }
}

/// A one- or two-byte payload, without allocating.
pub mod heapless_bytes {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Bytes {
        buf: [u8; 2],
        len: usize,
    }

    impl Bytes {
        pub fn one(a: u8) -> Self {
            Self {
                buf: [a, 0],
                len: 1,
            }
        }
        pub fn two(a: u8, b: u8) -> Self {
            Self {
                buf: [a, b],
                len: 2,
            }
        }
        pub fn as_slice(&self) -> &[u8] {
            &self.buf[..self.len]
        }
    }
}

// ---------------------------------------------------------------------------
// Sample rate
// ---------------------------------------------------------------------------

/// A programmed sample rate and the rate actually achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleRate {
    /// Twenty-eight bit resampler ratio.
    pub ratio: u32,
    /// High half, written to page 1 register 0x9F.
    pub high: u16,
    /// Low half, written to page 1 register 0xA1.
    pub low: u16,
}

/// Why a requested sample rate cannot be programmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateError {
    TooLow,
    TooHigh,
    /// The converter cannot settle in the gap between 300 kHz and 900 kHz.
    UnsupportedGap,
}

/// Computes the resampler ratio for a requested rate.
///
/// The multiply is done in floating point rather than integer arithmetic, matching the
/// reference driver. That is not incidental: the truncation point differs between the two,
/// and an integer version lands on a neighbouring ratio for some rates.
///
/// The low two bits are cleared because the resampler only accepts multiples of four.
pub fn sample_rate(rate: u32, xtal: u32) -> Result<SampleRate, RateError> {
    if rate <= 225_000 {
        return Err(RateError::TooLow);
    }
    if rate > 3_200_000 {
        return Err(RateError::TooHigh);
    }
    if rate > 300_000 && rate <= 900_000 {
        return Err(RateError::UnsupportedGap);
    }

    let ratio = ((xtal as f64 * (1u64 << 22) as f64) / rate as f64) as u32 & 0x0FFF_FFFC;
    Ok(SampleRate {
        ratio,
        high: (ratio >> 16) as u16,
        low: (ratio & 0xFFFF) as u16,
    })
}

/// The rate a given ratio actually produces.
pub fn achieved_rate(ratio: u32, xtal: u32) -> f64 {
    // Bit 27 is sign-extended into bit 28 by the hardware.
    let real = ratio | ((ratio & 0x0800_0000) << 1);
    (xtal as f64 * (1u64 << 22) as f64) / real as f64
}

// ---------------------------------------------------------------------------
// Intermediate frequency
// ---------------------------------------------------------------------------

/// Registers 0x19, 0x1A and 0x1B on page 1, which set the demodulator's IF offset.
///
/// The value is negated and held as a 22-bit two's complement. Missing the negation puts
/// the receiver an IF away from where it was asked to tune — twice the IF from the wanted
/// signal, which looks exactly like a broken tuner.
pub fn if_frequency(if_hz: u32, xtal: u32) -> [u8; 3] {
    let scaled = ((if_hz as f64 * (1u64 << 22) as f64) / xtal as f64) as i32;
    let value = (-scaled) as u32 & 0x3F_FFFF;
    [
        ((value >> 16) & 0x3F) as u8,
        ((value >> 8) & 0xFF) as u8,
        (value & 0xFF) as u8,
    ]
}

/// Registers 0x3E and 0x3F on page 1, which trim the sample clock in parts per million.
pub fn frequency_correction(ppm: i32) -> [u8; 2] {
    let offset = (-ppm as i64 * (1i64 << 24)) / 1_000_000;
    [
        ((offset >> 8) & 0x3F) as u8, // 0x3E
        (offset & 0xFF) as u8,        // 0x3F
    ]
}

// ---------------------------------------------------------------------------
// Tuner synthesiser
// ---------------------------------------------------------------------------

/// Lowest and highest frequency the tuner's oscillator can run at, in kilohertz.
const VCO_MIN_KHZ: u64 = 1_770_000;
const VCO_MAX_KHZ: u64 = 3_540_000;

/// Divider and fractional settings for one tuner frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PllSetting {
    pub mix_div: u32,
    pub div_num: u8,
    pub nint: u32,
    pub sdm: u16,
    /// Register 0x14: the integer divider, split across two fields.
    pub reg_14: u8,
    /// Register 0x15: low half of the fractional divider.
    pub reg_15: u8,
    /// Register 0x16: high half of the fractional divider.
    pub reg_16: u8,
    /// Whether the fractional path should be powered down, which it can be when the
    /// division comes out exact.
    pub power_down_sdm: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PllError {
    /// The frequency cannot be reached by any available divider.
    OutOfRange,
    /// The integer divider overflowed its field.
    DividerTooLarge,
}

/// Works out how to program the tuner's synthesiser for a given frequency.
///
/// `vco_fine_tune` comes from reading register 4 and reports whether the oscillator is
/// running fast or slow for its current setting; nudging the divider in response keeps it
/// near the middle of its range. `vco_power_ref` is 1 for the R828D and 2 for the R820T,
/// and also bounds the integer divider.
pub fn compute_pll(
    freq_hz: u32,
    xtal_hz: u32,
    vco_fine_tune: u8,
    vco_power_ref: u8,
) -> Result<PllSetting, PllError> {
    let freq_khz = (freq_hz as u64 + 500) / 1000;

