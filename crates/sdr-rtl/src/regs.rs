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

    // Pick the smallest divider that puts the oscillator inside its usable range.
    let mut mix_div: u32 = 2;
    let mut div_num: i32 = 0;
    let mut found = false;
    while mix_div <= 64 {
        let vco = freq_khz * mix_div as u64;
        if (VCO_MIN_KHZ..VCO_MAX_KHZ).contains(&vco) {
            found = true;
            break;
        }
        mix_div <<= 1;
        div_num += 1;
    }
    if !found {
        return Err(PllError::OutOfRange);
    }

    // The oscillator reports which side of centre it is sitting on; shifting the divider
    // moves it back before it runs out of tuning range.
    if vco_fine_tune > vco_power_ref {
        div_num -= 1;
    } else if vco_fine_tune < vco_power_ref {
        div_num += 1;
    }
    let div_num = div_num.clamp(0, 5) as u8;

    // Integer-exact fractional division. Working in 64-bit here rather than reproducing
    // the reference's iterative halving loop gives the same answer without accumulating
    // rounding at each step.
    let vco_freq = freq_hz as u64 * mix_div as u64;
    let pll_ref = xtal_hz as u64;
    let vco_div = (pll_ref + 65536 * vco_freq) / (2 * pll_ref);
    let nint = (vco_div / 65536) as u32;
    let sdm = (vco_div % 65536) as u16;

    if nint > (128 / vco_power_ref as u32) - 1 {
        return Err(PllError::DividerTooLarge);
    }

    // The integer divider is stored as a quotient and remainder about 13, not directly.
    let ni = (nint - 13) / 4;
    let si = nint - 4 * ni - 13;

    Ok(PllSetting {
        mix_div,
        div_num,
        nint,
        sdm,
        reg_14: (ni + (si << 6)) as u8,
        reg_15: (sdm & 0xFF) as u8,
        reg_16: (sdm >> 8) as u8,
        power_down_sdm: sdm == 0,
    })
}

// ---------------------------------------------------------------------------
// Blog V4 front end
// ---------------------------------------------------------------------------

/// Which port of the triplexer a frequency arrives on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    /// Through the upconverter and into the tuner's second cable input.
    Hf,
    /// Directly into the tuner's first cable input.
    Vhf,
    /// Directly into the tuner's antenna input.
    Uhf,
}

/// Selects the triplexer port for a frequency.
pub fn band_for(freq_hz: u32) -> Band {
    if freq_hz <= UPCONVERT_CROSSOVER {
        Band::Hf
    } else if freq_hz < VHF_UHF_BOUNDARY {
        Band::Vhf
    } else {
        Band::Uhf
    }
}

/// Frequency the tuner should be asked for, accounting for the upconverter.
///
/// The reference driver uses a strict comparison here while selecting the band with a
/// non-strict one, so a request for exactly the crossover frequency takes the HF path
/// without being upconverted — the tuner then looks 28.8 MHz away from the signal. Both
/// comparisons are non-strict here so the two agree.
pub fn tuner_frequency(freq_hz: u32) -> u32 {
    if freq_hz <= UPCONVERT_CROSSOVER {
        freq_hz + UPCONVERT_CROSSOVER
    } else {
        freq_hz
    }
}

/// Whether the switchable notch filters should be engaged at this frequency.
///
/// The notches attenuate the broadcast AM, broadcast FM and DAB bands, which are strong
/// enough to desensitise the front end from outside the wanted channel. They are bypassed
/// when the receiver is tuned into one of those bands, since notching the thing you are
/// trying to hear defeats the purpose.
pub fn notch_engaged(freq_hz: u32) -> bool {
    let in_broadcast_band = freq_hz <= 2_200_000
        || (85_000_000..=112_000_000).contains(&freq_hz)
        || (172_000_000..=242_000_000).contains(&freq_hz);
    !in_broadcast_band
}

// ---------------------------------------------------------------------------
// Tuner gain
// ---------------------------------------------------------------------------

/// Incremental gain of each low-noise amplifier step, in tenths of a decibel.
pub const LNA_STEPS: [i32; 16] = [0, 9, 13, 40, 38, 13, 31, 22, 26, 31, 26, 14, 19, 5, 35, 13];

/// Incremental gain of each mixer step, in tenths of a decibel.
pub const MIXER_STEPS: [i32; 16] = [0, 5, 10, 10, 19, 9, 10, 25, 17, 10, 8, 16, 13, 6, 3, -8];

/// Gain settings the tuner can actually reach, in tenths of a decibel.
pub const GAIN_VALUES: [i32; 29] = [
    0, 9, 14, 27, 37, 77, 87, 125, 144, 157, 166, 197, 207, 229, 254, 280, 297, 328, 338, 364, 372,
    386, 402, 421, 434, 439, 445, 480, 496,
];

/// Amplifier index positions for a requested gain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GainSetting {
    pub lna_index: u8,
    pub mixer_index: u8,
    /// Total gain actually achieved, in tenths of a decibel.
    pub achieved: i32,
}

/// Distributes a requested gain across the low-noise amplifier and mixer stages.
///
/// The two are stepped alternately rather than filling one before the other, which keeps
/// both away from the ends of their ranges where their steps are least well behaved — note
/// that the tables are not monotonic, and the mixer's last step is negative.
pub fn distribute_gain(target_tenths: i32) -> GainSetting {
    let mut total = 0;
    let mut lna_index = 0u8;
    let mut mixer_index = 0u8;

    for step in 1..16 {
        let next_lna = total + LNA_STEPS[step];
        if next_lna >= target_tenths {
            break;
        }
        total = next_lna;
        lna_index = step as u8;

        let next_mixer = total + MIXER_STEPS[step];
        if next_mixer >= target_tenths {
            break;
        }
        total = next_mixer;
        mixer_index = step as u8;
    }

    GainSetting {
        lna_index,
        mixer_index,
        achieved: total,
    }
}

/// Snaps a gain in tenths of a decibel to the nearest value the tuner supports.
pub fn nearest_gain(target_tenths: i32) -> i32 {
    *GAIN_VALUES
        .iter()
        .min_by_key(|g| (**g - target_tenths).abs())
        .expect("gain table is never empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_transfer_encodings_match_the_wire_format() {
        // Flat blocks put the address in wValue and the block in the high byte of wIndex.
        assert_eq!(block_read(0x2000, Block::Usb), (0x2000, 0x0100));
        assert_eq!(block_write(0x2000, Block::Usb), (0x2000, 0x0110));
        assert_eq!(block_write(0x3000, Block::Sys), (0x3000, 0x0210));

        // The demodulator does not: the address moves into wValue's high byte.
        assert_eq!(demod_read(1, 0x01), (0x0120, 0x0001));
        assert_eq!(demod_write(1, 0x01), (0x0120, 0x0011));
        assert_eq!(demod_write(0, 0x19), (0x1920, 0x0010));

        // Tuner access goes through the I2C block at a fixed index.
        assert_eq!(i2c_write(0x74), (0x0074, 0x0610));
        assert_eq!(i2c_read(0x74), (0x0074, 0x0600));
    }

    #[test]
    fn multi_byte_values_are_sent_most_significant_first() {
        assert_eq!(encode_value(0x1002, 2).as_slice(), &[0x10, 0x02]);
        assert_eq!(encode_value(0x0002, 2).as_slice(), &[0x00, 0x02]);
        assert_eq!(encode_value(0x09, 1).as_slice(), &[0x09]);
    }

    #[test]
    fn sample_rate_ratios_match_worked_values() {
        // Computed from the reference formula; the common rates all divide exactly.
        let cases = [
            (2_048_000u32, 0x0384_0000u32, 0x0384u16, 0x0000u16),
            (2_400_000, 0x0300_0000, 0x0300, 0x0000),
            (3_200_000, 0x0240_0000, 0x0240, 0x0000),
            (250_000, 0x0CCC_CCCC, 0x0CCC, 0xCCCC),
        ];
        for (rate, ratio, high, low) in cases {
            let got = sample_rate(rate, RTL_XTAL).expect("rate should be valid");
            assert_eq!(got.ratio, ratio, "ratio for {rate}");
            assert_eq!(got.high, high, "high half for {rate}");
            assert_eq!(got.low, low, "low half for {rate}");
        }
    }

    #[test]
    fn common_rates_come_back_exactly() {
        for rate in [2_048_000u32, 2_400_000, 3_200_000, 1_024_000] {
            let s = sample_rate(rate, RTL_XTAL).unwrap();
            let actual = achieved_rate(s.ratio, RTL_XTAL);
            assert!(
                (actual - rate as f64).abs() < 1.0,
                "{rate} came back as {actual}"
            );
        }
    }

    #[test]
    fn resampler_ratio_is_always_a_multiple_of_four() {
        for rate in [226_000u32, 250_000, 1_000_000, 2_400_000, 3_200_000] {
            let s = sample_rate(rate, RTL_XTAL).unwrap();
            assert_eq!(s.ratio & 0x3, 0, "ratio for {rate} was not aligned");
        }
    }

