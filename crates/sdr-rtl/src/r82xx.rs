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

        // Pin 5 enables the upconverter on later boards, low for HF. Earlier boards leave
        // it running and ignore this, so driving it is harmless either way.
        rtl.set_i2c_repeater(false).await?;
        rtl.set_gpio_output(5).await?;
        rtl.write_gpio(5, band != Band::Hf).await?;
        rtl.set_i2c_repeater(true).await?;

        Ok(())
    }

    /// Engages or bypasses the broadcast notch filters.
    async fn set_notch(
        &mut self,
        rtl: &mut Rtl2832<T>,
        freq_hz: u32,
    ) -> Result<(), TransportError> {
        let value = if regs::notch_engaged(freq_hz) {
            0x08
        } else {
            0x00
        };
        self.write_mask(rtl, 0x17, value, 0x08).await
    }

    /// Programs the synthesiser to `lo_hz`.
    async fn set_pll(&mut self, rtl: &mut Rtl2832<T>, lo_hz: u32) -> Result<(), TransportError> {
        // Reference divider straight through, and autotune at its finest step.
        self.write_mask(rtl, 0x10, 0x00, 0x10).await?;
        self.write_mask(rtl, 0x1A, 0x00, 0x0C).await?;
        self.write_mask(rtl, 0x12, 0x80, 0xE0).await?;

        // Register 4 reports whether the oscillator is running above or below centre for
        // its current divider; the calculation shifts the divider to bring it back.
        let mut status = [0u8; 5];
        rtl.tuner_read(self.i2c_addr, &mut status).await?;
        let vco_fine_tune = (status[4] & 0x30) >> 4;

        let pll = regs::compute_pll(lo_hz, self.xtal, vco_fine_tune, VCO_POWER_REF)
            .map_err(|e| TransportError::Io(format!("cannot tune to {lo_hz} Hz: {e:?}")))?;

        self.write_mask(rtl, 0x10, pll.div_num << 5, 0xE0).await?;
        self.write(rtl, 0x14, pll.reg_14).await?;
        // The fractional path can be powered down when the division comes out exact,
        // which removes its spurs.
        self.write_mask(
            rtl,
            0x12,
            if pll.power_down_sdm { 0x08 } else { 0x00 },
            0x08,
        )
        .await?;
        self.write(rtl, 0x16, pll.reg_16).await?;
        self.write(rtl, 0x15, pll.reg_15).await?;

        Ok(())
    }

    /// Reads back whether the synthesiser has locked.
    pub async fn is_locked(&mut self, rtl: &mut Rtl2832<T>) -> Result<bool, TransportError> {
        rtl.set_i2c_repeater(true).await?;
        let mut status = [0u8; 3];
        let result = rtl.tuner_read(self.i2c_addr, &mut status).await;
        rtl.set_i2c_repeater(false).await?;
        result?;
        Ok(status[2] & 0x40 != 0)
    }

    /// Sets the gain, either fixed or under the tuner's own control.
    pub async fn set_gain(
        &mut self,
        rtl: &mut Rtl2832<T>,
        mode: GainMode,
    ) -> Result<(), TransportError> {
        rtl.set_i2c_repeater(true).await?;

        match mode {
            GainMode::Automatic => {
                self.write_mask(rtl, 0x05, 0x00, 0x10).await?;
                self.write_mask(rtl, 0x07, 0x10, 0x10).await?;
                // A fixed 26.5 dB on the variable stage, which is what the tuner's loops
                // are tuned around.
                self.write_mask(rtl, 0x0C, 0x0B, 0x9F).await?;
            }
            GainMode::Manual(tenths) => {
                // Both automatic loops off before touching the manual indices, or they
                // will move again immediately.
                self.write_mask(rtl, 0x05, 0x10, 0x10).await?;
                self.write_mask(rtl, 0x07, 0x00, 0x10).await?;
                self.write_mask(rtl, 0x0C, 0x08, 0x9F).await?;

                let setting = regs::distribute_gain(tenths);
                self.write_mask(rtl, 0x05, setting.lna_index, 0x0F).await?;
                self.write_mask(rtl, 0x07, setting.mixer_index, 0x0F)
                    .await?;
            }
        }

        rtl.set_i2c_repeater(false).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;

    fn block_on<F: core::future::Future>(mut future: F) -> F::Output {
        use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        static VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VTABLE),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: every vtable entry ignores its data pointer and does nothing.
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
        let mut cx = Context::from_waker(&waker);
        // SAFETY: the future is owned here and never moved after pinning.
        let mut future = unsafe { core::pin::Pin::new_unchecked(&mut future) };
        loop {
            if let Poll::Ready(v) = future.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    fn rig() -> (Rtl2832<MockTransport>, R82xx<MockTransport>) {
        let mut rtl = Rtl2832::new(MockTransport::new());
        rtl.set_verify_demod_writes(false);
        (rtl, R82xx::new_v4())
    }

    /// Every tuner write recorded, as `(register, value)`.
    fn tuner_writes(log: &MockTransport) -> Vec<(u8, u8)> {
        let (v, i) = regs::i2c_write(R828D_I2C_ADDR);
        log.writes()
            .into_iter()
            .filter(|(wv, wi, data)| *wv == v && *wi == i && data.len() == 2)
            .map(|(_, _, data)| (data[0], data[1]))
            .collect()
    }

    #[test]
    fn the_v4_clocks_its_tuner_from_the_shared_reference() {
        // This is the whole difference between a V4 and a conventional R828D board, and
        // getting it wrong scales every synthesiser calculation by 1.8.
        assert_eq!(R82xx::<MockTransport>::new_v4().xtal(), 28_800_000);
        assert_eq!(
            R82xx::<MockTransport>::new_legacy_r828d().xtal(),
            16_000_000
        );
    }

    #[test]
    fn init_writes_every_register_in_the_set() {
        let (mut rtl, mut tuner) = rig();
        block_on(tuner.init(&mut rtl)).unwrap();
        let log = rtl.into_transport();

        // The run is split across several messages by the bus limit, so reassemble it
        // rather than looking for one transfer.
        let (v, i) = regs::i2c_write(R828D_I2C_ADDR);
        let mut rebuilt = Vec::new();
        for (wv, wi, data) in log.writes() {
            if wv == v && wi == i && data[0] == FIRST_REG + rebuilt.len() as u8 {
                rebuilt.extend_from_slice(&data[1..]);
            }
        }
        assert_eq!(rebuilt, INIT_REGS.to_vec(), "init register set incomplete");
    }

    #[test]
    fn masked_writes_preserve_neighbouring_bits() {
        let (mut rtl, mut tuner) = rig();
        // Register 0x05 starts at 0x83. Setting only the low nibble must leave 0x80 alone.
        block_on(tuner.write_mask(&mut rtl, 0x05, 0x0A, 0x0F)).unwrap();
        assert_eq!(tuner.shadow(0x05), 0x8A);

        // And touching the automatic-gain bit must not disturb the index just written.
        block_on(tuner.write_mask(&mut rtl, 0x05, 0x10, 0x10)).unwrap();
        assert_eq!(tuner.shadow(0x05), 0x9A);
    }

    #[test]
    fn hf_selects_the_second_cable_input() {
        let (mut rtl, mut tuner) = rig();
        block_on(tuner.set_frequency(&mut rtl, 7_100_000)).unwrap();

        assert_eq!(tuner.band(), Some(Band::Hf));
        // Cable 2 on, cable 1 off, antenna input deselected.
        assert_eq!(tuner.shadow(0x06) & 0x08, 0x08, "HF port not selected");
        assert_eq!(tuner.shadow(0x05) & 0x40, 0x00, "VHF port still selected");
        assert_eq!(
            tuner.shadow(0x05) & 0x20,
            0x20,
            "antenna input still selected"
        );
    }

    #[test]
    fn vhf_and_uhf_select_their_own_inputs() {
        let (mut rtl, mut tuner) = rig();
        block_on(tuner.set_frequency(&mut rtl, 100_000_000)).unwrap();
        assert_eq!(tuner.band(), Some(Band::Vhf));
        assert_eq!(tuner.shadow(0x05) & 0x40, 0x40, "VHF port not selected");
        assert_eq!(tuner.shadow(0x06) & 0x08, 0x00, "HF port still selected");

        block_on(tuner.set_frequency(&mut rtl, 433_000_000)).unwrap();
        assert_eq!(tuner.band(), Some(Band::Uhf));
        // Inverted sense: cleared means selected.
        assert_eq!(
            tuner.shadow(0x05) & 0x20,
            0x00,
            "antenna input not selected"
        );
        assert_eq!(tuner.shadow(0x05) & 0x40, 0x00, "VHF port still selected");
    }

    #[test]
    fn hf_bypasses_the_tracking_filter_on_every_tune() {
        // Selecting the mux reasserts the tracking filter, so a bypass applied only on a
        // band change is silently undone by the next tune within HF.
        let (mut rtl, mut tuner) = rig();
        block_on(tuner.set_frequency(&mut rtl, 7_100_000)).unwrap();
        let first = rtl.transport_mut().log.len();
        rtl.transport_mut().clear();

        block_on(tuner.set_frequency(&mut rtl, 14_200_000)).unwrap();
        let log = rtl.into_transport();
        let writes = tuner_writes(&log);

        assert!(first > 0);
        assert!(
            writes.iter().any(|(reg, _)| *reg == 0x1B),
            "tracking filter bypass not reapplied on a second HF tune"
        );
    }

    #[test]
    fn band_switching_is_skipped_when_the_band_is_unchanged() {
        let (mut rtl, mut tuner) = rig();
        block_on(tuner.set_frequency(&mut rtl, 100_000_000)).unwrap();
        rtl.transport_mut().clear();

        // A second VHF frequency should not rewrite the mux.
        block_on(tuner.set_frequency(&mut rtl, 105_000_000)).unwrap();
        let log = rtl.into_transport();
        let writes = tuner_writes(&log);
        assert!(
            !writes.iter().any(|(reg, _)| *reg == 0x06),
            "rewrote the mux without changing band"
        );
    }

    #[test]
    fn notch_state_follows_the_frequency_not_the_band() {
        let (mut rtl, mut tuner) = rig();

        // Inside broadcast FM the notch is bypassed.
        block_on(tuner.set_frequency(&mut rtl, 98_000_000)).unwrap();
        assert_eq!(
            tuner.shadow(0x17) & 0x08,
            0x00,
            "notch engaged inside FM broadcast"
        );

        // Still VHF, but outside the broadcast band, so it engages.
        block_on(tuner.set_frequency(&mut rtl, 145_000_000)).unwrap();
        assert_eq!(
            tuner.shadow(0x17) & 0x08,
            0x08,
            "notch not engaged outside FM broadcast"
        );
    }

    #[test]
    fn hf_tuning_targets_the_upconverted_frequency() {
        let (mut rtl, mut tuner) = rig();
        block_on(tuner.set_frequency(&mut rtl, 7_100_000)).unwrap();
        let log = rtl.into_transport();
        let writes = tuner_writes(&log);

        // Recover the divider settings and check they describe the upconverted frequency
        // plus the intermediate frequency, not the requested one.
        let reg_14 = writes
            .iter()
            .rev()
            .find(|(r, _)| *r == 0x14)
            .expect("no divider write")
            .1;
        let reg_16 = writes
            .iter()
            .rev()
            .find(|(r, _)| *r == 0x16)
            .expect("no fraction high")
            .1;
        let reg_15 = writes
            .iter()
            .rev()
            .find(|(r, _)| *r == 0x15)
            .expect("no fraction low")
            .1;

        let expected_lo = regs::tuner_frequency(7_100_000) + crate::rtl2832::DEFAULT_IF_HZ;
        let expected = regs::compute_pll(expected_lo, regs::RTL_XTAL, 0, VCO_POWER_REF).unwrap();
        assert_eq!(reg_14, expected.reg_14);
        assert_eq!(reg_16, expected.reg_16);
        assert_eq!(reg_15, expected.reg_15);
    }

    #[test]
    fn manual_gain_disables_both_automatic_loops_first() {
        let (mut rtl, mut tuner) = rig();
        block_on(tuner.set_gain(&mut rtl, GainMode::Manual(300))).unwrap();

        // Automatic low-noise-amplifier control off, automatic mixer control off.
        assert_eq!(tuner.shadow(0x05) & 0x10, 0x10, "LNA loop still running");
        assert_eq!(tuner.shadow(0x07) & 0x10, 0x00, "mixer loop still running");

        let expected = regs::distribute_gain(300);
        assert_eq!(tuner.shadow(0x05) & 0x0F, expected.lna_index);
        assert_eq!(tuner.shadow(0x07) & 0x0F, expected.mixer_index);
    }

    #[test]
    fn automatic_gain_hands_control_back_to_the_tuner() {
        let (mut rtl, mut tuner) = rig();
        block_on(tuner.set_gain(&mut rtl, GainMode::Manual(300))).unwrap();
        block_on(tuner.set_gain(&mut rtl, GainMode::Automatic)).unwrap();

        assert_eq!(tuner.shadow(0x05) & 0x10, 0x00, "LNA loop not re-enabled");
        assert_eq!(tuner.shadow(0x07) & 0x10, 0x10, "mixer loop not re-enabled");
    }

    #[test]
    fn tuning_is_bracketed_by_the_i2c_repeater() {
        let (mut rtl, mut tuner) = rig();
        block_on(tuner.set_frequency(&mut rtl, 100_000_000)).unwrap();
        let log = rtl.into_transport();

        let (v, i) = regs::demod_write(1, 0x01);
        let toggles: Vec<u8> = log
            .writes()
            .into_iter()
            .filter(|(wv, wi, _)| *wv == v && *wi == i)
            .map(|(_, _, data)| data[0])
            .collect();

        assert_eq!(toggles.first(), Some(&0x18), "repeater not opened first");
        assert_eq!(toggles.last(), Some(&0x10), "repeater left open");
    }

    #[test]
    fn lock_status_is_read_from_the_third_register() {
        let (mut rtl, mut tuner) = rig();
        // Bit 6 of register 2 is the lock indicator.
        rtl.transport_mut().push_response(vec![0u8, 0, 0x40, 0, 0]);
        assert!(block_on(tuner.is_locked(&mut rtl)).unwrap());

        let (mut rtl2, mut tuner2) = rig();
        rtl2.transport_mut().push_response(vec![0u8; 5]);
        assert!(!block_on(tuner2.is_locked(&mut rtl2)).unwrap());
    }

    #[test]
    fn an_unreachable_frequency_is_reported_rather_than_programmed() {
        let (mut rtl, mut tuner) = rig();
        // Above what any divider can reach, and not low enough to be upconverted.
        let err = block_on(tuner.set_frequency(&mut rtl, 2_500_000_000)).unwrap_err();
        assert!(
            matches!(err, TransportError::Io(_)),
            "expected a reported failure, got {err:?}"
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn writing_outside_the_register_window_is_rejected() {
        let (mut rtl, mut tuner) = rig();
        block_on(tuner.write(&mut rtl, 0x02, 0x00)).unwrap();
    }
}
