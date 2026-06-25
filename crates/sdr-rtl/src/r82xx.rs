//! The R828D tuner, and the Blog V4 front end wrapped around it.
//!
//! The tuner's registers cannot be read back selectively — a read always starts at zero and
//! streams forward — so the driver keeps a shadow copy and does every masked update against
//! that. Losing the shadow means losing the ability to change one field without disturbing
//! its neighbours.

use crate::regs::{self, Band};
use crate::rtl2832::{Rtl2832, R828D_I2C_ADDR};
use crate::transport::{Transport, TransportError};

/// Lowest and highest writable register.
const FIRST_REG: u8 = 0x05;
const LAST_REG: u8 = 0x1F;
const REG_COUNT: usize = (LAST_REG - FIRST_REG + 1) as usize;

/// Power-on values for registers 0x05 through 0x1F.
const INIT_REGS: [u8; REG_COUNT] = [
    0x83, 0x30, 0x75, // 05 to 07
    0xC0, 0x40, 0xD6, 0x6C, // 08 to 0b
    0xF5, 0x63, 0x75, 0x68, // 0c to 0f
    0x6C, 0x83, 0x80, 0x00, // 10 to 13
    0x0F, 0x00, 0xC0, 0x30, // 14 to 17
    0x48, 0xCC, 0x60, 0x00, // 18 to 1b
    0x54, 0xAE, 0x4A, 0xC0, // 1c to 1f
];

/// The R828D reports its oscillator sitting at this level when correctly centred. The
/// R820T uses 2; the value also bounds the integer divider.
const VCO_POWER_REF: u8 = 1;

/// How the tuner's gain is decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainMode {
    /// The tuner runs its own loops.
    Automatic,
    /// Fixed gain, in tenths of a decibel.
    Manual(i32),
}

/// Driver for the R828D and the switching around it.
#[derive(Debug)]
pub struct R82xx<T> {
    i2c_addr: u8,
    /// Shadow of registers 0x05 to 0x1F, since the part cannot be read selectively.
    shadow: [u8; REG_COUNT],
    /// The Blog V4 clocks the tuner from the 28.8 MHz reference. A conventional R828D
    /// board uses a separate 16 MHz crystal, and assuming that here is the single most
    /// common way to end up tuned to entirely the wrong frequency.
    xtal: u32,
    /// Band currently selected, so the switching is only rewritten when it changes.
    band: Option<Band>,
    tuned_hz: u32,
    if_hz: u32,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Transport> R82xx<T> {
    /// Builds a driver for the Blog V4's tuner.
    pub fn new_v4() -> Self {
        Self {
            i2c_addr: R828D_I2C_ADDR,
            shadow: INIT_REGS,
            xtal: regs::RTL_XTAL,
            band: None,
            tuned_hz: 0,
            if_hz: crate::rtl2832::DEFAULT_IF_HZ,
            _marker: core::marker::PhantomData,
        }
    }

    /// Builds a driver for a conventional R828D board, which clocks its tuner separately.
    pub fn new_legacy_r828d() -> Self {
        let mut tuner = Self::new_v4();
        tuner.xtal = 16_000_000;
        tuner
    }

    pub fn xtal(&self) -> u32 {
        self.xtal
    }

    pub fn tuned_frequency(&self) -> u32 {
        self.tuned_hz
    }

    pub fn band(&self) -> Option<Band> {
        self.band
    }

    /// Current shadow value of a register.
    pub fn shadow(&self, reg: u8) -> u8 {
        assert!(
            (FIRST_REG..=LAST_REG).contains(&reg),
            "register 0x{reg:02X} is out of range"
        );
        self.shadow[(reg - FIRST_REG) as usize]
    }

    /// Writes a whole register.
    async fn write(
        &mut self,
        rtl: &mut Rtl2832<T>,
        reg: u8,
        value: u8,
    ) -> Result<(), TransportError> {
        assert!(
            (FIRST_REG..=LAST_REG).contains(&reg),
            "register 0x{reg:02X} is out of range"
        );
        self.shadow[(reg - FIRST_REG) as usize] = value;
        rtl.tuner_write(self.i2c_addr, reg, value).await
    }

    /// Updates only the bits covered by `mask`, leaving the rest as they were.
    async fn write_mask(
        &mut self,
        rtl: &mut Rtl2832<T>,
        reg: u8,
        value: u8,
        mask: u8,
    ) -> Result<(), TransportError> {
        let current = self.shadow(reg);
        let updated = (current & !mask) | (value & mask);
        self.write(rtl, reg, updated).await
    }

    /// Loads the power-on register set.
    pub async fn init(&mut self, rtl: &mut Rtl2832<T>) -> Result<(), TransportError> {
        self.shadow = INIT_REGS;
        rtl.set_i2c_repeater(true).await?;
        rtl.tuner_write_burst(self.i2c_addr, FIRST_REG, &INIT_REGS)
            .await?;
        rtl.set_i2c_repeater(false).await?;
        self.band = None;
        Ok(())
    }

    /// Tunes to `freq_hz`, handling the upconverter and the triplexer.
    ///
    /// The caller passes the frequency it wants to receive. Everything below the crossover
    /// is upconverted, so what the synthesiser is actually programmed to is not what was
    /// asked for — see [`regs::tuner_frequency`].
    pub async fn set_frequency(
        &mut self,
        rtl: &mut Rtl2832<T>,
        freq_hz: u32,
    ) -> Result<(), TransportError> {
        let band = regs::band_for(freq_hz);
        // The synthesiser is asked for the wanted frequency plus the intermediate
        // frequency, because the tuner mixes from above.
        let lo_hz = regs::tuner_frequency(freq_hz) + self.if_hz;

        rtl.set_i2c_repeater(true).await?;

        // Switching costs transfers and only matters when the band actually changes.
        if self.band != Some(band) {
            self.set_mux(rtl, band).await?;
            self.band = Some(band);
        }
        // The notch decision is per-frequency, not per-band.
        self.set_notch(rtl, freq_hz).await?;

        // Selecting the mux reasserts the tracking filter, so the HF bypass has to be
        // reapplied after it rather than only when the band changes.
        if band == Band::Hf {
            self.write_mask(rtl, 0x1A, 0x40, 0xC3).await?;
            self.write(rtl, 0x1B, 0x00).await?;
        }

        self.set_pll(rtl, lo_hz).await?;
        rtl.set_i2c_repeater(false).await?;

        self.tuned_hz = freq_hz;
        Ok(())
    }

    /// Routes the antenna connector to the right tuner input.
    ///
    /// The V4 triplexes one connector into all three of the tuner's inputs, and on later
    /// boards also powers the upconverter down when it is not in use.
    async fn set_mux(&mut self, rtl: &mut Rtl2832<T>, band: Band) -> Result<(), TransportError> {
        let cable_2 = if band == Band::Hf { 0x08 } else { 0x00 };
        self.write_mask(rtl, 0x06, cable_2, 0x08).await?;

        let cable_1 = if band == Band::Vhf { 0x40 } else { 0x00 };
        self.write_mask(rtl, 0x05, cable_1, 0x40).await?;

        // Inverted sense: clearing this bit is what selects the antenna input.
        let air_in = if band == Band::Uhf { 0x00 } else { 0x20 };
        self.write_mask(rtl, 0x05, air_in, 0x20).await?;

