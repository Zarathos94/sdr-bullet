//! The RTL2832U demodulator: register access, initialisation, and the sample stream.

use crate::regs::{self, encode_value, Block};
use crate::transport::{ControlRequest, Transport, TransportError};

/// I2C address of the R828D tuner fitted to the Blog V4.
///
/// This is the eight-bit form, which is what the demodulator's I2C master expects. The
/// R820T on earlier boards sits at 0x34, and probing the wrong one is the first thing that
/// goes wrong when a V3 driver meets V4 hardware.
pub const R828D_I2C_ADDR: u8 = 0x74;

/// Register 0 of the R82xx family reads back as this on a working part.
pub const R82XX_CHECK_VALUE: u8 = 0x69;

/// Default intermediate frequency for the R82xx tuners.
pub const DEFAULT_IF_HZ: u32 = 3_570_000;

/// Longest I2C message the demodulator's bus master will carry, register address included.
pub const MAX_I2C_MSG_LEN: usize = 8;

/// Decimating filter the demodulator applies ahead of the resampler.
///
/// The first eight are plain signed bytes; the remaining eight are twelve-bit values
/// packed three bytes to every two coefficients.
const FIR_COEFFICIENTS: [i32; 16] = [
    -54, -36, -41, -40, -32, -14, 14, 53, 101, 156, 215, 273, 327, 372, 404, 421,
];

/// Packs the filter coefficients into the twenty bytes the register block expects.
///
/// The mixed widths are not a quirk worth abstracting away — the first half genuinely is
/// eight-bit and the second half genuinely is twelve, and the packing is what the hardware
/// reads.
pub fn pack_fir(coefficients: &[i32; 16]) -> [u8; 20] {
    let mut out = [0u8; 20];
    for i in 0..8 {
        out[i] = coefficients[i] as i8 as u8;
    }
    // Two twelve-bit values share three bytes.
    for pair in 0..4 {
        let first = coefficients[8 + pair * 2];
        let second = coefficients[8 + pair * 2 + 1];
        let base = 8 + pair * 3;
        out[base] = (first >> 4) as u8;
        out[base + 1] = (((first << 4) & 0xF0) | ((second >> 8) & 0x0F)) as u8;
        out[base + 2] = (second & 0xFF) as u8;
    }
    out
}

/// Driver for the demodulator half of the device.
#[derive(Debug)]
pub struct Rtl2832<T> {
    transport: T,
    xtal: u32,
    /// The reference driver reads a demodulator register back after every write. It is
    /// undocumented why, and it doubles the number of round trips on a bus where round
    /// trips are the bottleneck — so it stays switchable and is measured rather than
    /// assumed. On by default, because matching the reference is the safe starting point.
    verify_demod_writes: bool,
}

impl<T: Transport> Rtl2832<T> {
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            xtal: regs::RTL_XTAL,
            verify_demod_writes: true,
        }
    }

    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    pub fn into_transport(self) -> T {
        self.transport
    }

    pub fn xtal(&self) -> u32 {
        self.xtal
    }

    pub fn set_verify_demod_writes(&mut self, verify: bool) {
        self.verify_demod_writes = verify;
    }

    // -- Raw register access ------------------------------------------------

    /// Writes to one of the flat address blocks.
    pub async fn write_reg(
        &mut self,
        block: Block,
        addr: u16,
        value: u16,
        len: usize,
    ) -> Result<(), TransportError> {
        let (v, i) = regs::block_write(addr, block);
        let bytes = encode_value(value, len);
        self.transport
            .control_out(ControlRequest::write(v, i), bytes.as_slice())
            .await
    }

    /// Reads from one of the flat address blocks.
    pub async fn read_reg(
        &mut self,
        block: Block,
        addr: u16,
        len: usize,
    ) -> Result<u16, TransportError> {
        let (v, i) = regs::block_read(addr, block);
        let mut buf = [0u8; 2];
        self.transport
            .control_in(ControlRequest::read(v, i), &mut buf[..len])
            .await?;
        Ok(if len == 1 {
            buf[0] as u16
        } else {
            ((buf[0] as u16) << 8) | buf[1] as u16
        })
    }

    /// Writes a demodulator register on the given page.
    pub async fn demod_write(
        &mut self,
        page: u8,
        addr: u16,
        value: u16,
        len: usize,
    ) -> Result<(), TransportError> {
        let (v, i) = regs::demod_write(page, addr);
        let bytes = encode_value(value, len);
        self.transport
            .control_out(ControlRequest::write(v, i), bytes.as_slice())
            .await?;
        if self.verify_demod_writes {
            let (rv, ri) = regs::demod_read(0x0A, 0x01);
            let mut scratch = [0u8; 1];
            self.transport
                .control_in(ControlRequest::read(rv, ri), &mut scratch)
                .await?;
        }
        Ok(())
    }

    /// Reads a demodulator register on the given page.
    pub async fn demod_read(&mut self, page: u8, addr: u16) -> Result<u8, TransportError> {
        let (v, i) = regs::demod_read(page, addr);
        let mut buf = [0u8; 1];
        self.transport
            .control_in(ControlRequest::read(v, i), &mut buf)
            .await?;
        Ok(buf[0])
    }

    // -- Tuner bus ----------------------------------------------------------

    /// Opens or closes the path between the USB host and the tuner's I2C bus.
    ///
    /// The tuner is unreachable while this is closed, so every tuner access has to be
    /// bracketed by it.
    pub async fn set_i2c_repeater(&mut self, enabled: bool) -> Result<(), TransportError> {
        self.demod_write(1, 0x01, if enabled { 0x18 } else { 0x10 }, 1)
            .await
    }

    /// Writes `value` to `reg` on the tuner.
    pub async fn tuner_write(
        &mut self,
        i2c_addr: u8,
        reg: u8,
        value: u8,
    ) -> Result<(), TransportError> {
        let (v, i) = regs::i2c_write(i2c_addr);
        self.transport
            .control_out(ControlRequest::write(v, i), &[reg, value])
            .await
    }

    /// Writes a run of consecutive tuner registers.
    ///
    /// The demodulator's I2C master will not carry a message longer than
    /// [`MAX_I2C_MSG_LEN`], so a long run is split across several transfers with the
    /// register address advanced each time. Sending the whole run in one transfer stalls
    /// the endpoint — which is what happens if the limit is missed, since nothing in the
    /// interface hints that it exists.
    pub async fn tuner_write_burst(
        &mut self,
        i2c_addr: u8,
        start_reg: u8,
        values: &[u8],
    ) -> Result<(), TransportError> {
        let (v, i) = regs::i2c_write(i2c_addr);
        let per_message = MAX_I2C_MSG_LEN - 1;

        for (chunk_index, chunk) in values.chunks(per_message).enumerate() {
            let mut payload = Vec::with_capacity(chunk.len() + 1);
            payload.push(start_reg + (chunk_index * per_message) as u8);
            payload.extend_from_slice(chunk);
            self.transport
                .control_out(ControlRequest::write(v, i), &payload)
                .await?;
        }
        Ok(())
    }

    /// Reads `len` bytes from the tuner, starting at register zero.
    ///
    /// The R82xx has no read pointer — a read always begins at register zero and streams
    /// forward, so getting at register 4 means reading five bytes and discarding four.
    ///
    /// Note that the bytes arrive the right way round. Several reference implementations
    /// pass every byte of an R82xx read through a bit-reversal, and doing the same here
    /// turns the identifying value in register 0 from 0x69 into 0x96 — a working tuner
    /// then looks absent. Measured against a Blog V4 the data needs no reversal, so
    /// whatever those implementations are compensating for is not present on this path.
    pub async fn tuner_read(&mut self, i2c_addr: u8, buf: &mut [u8]) -> Result<(), TransportError> {
        let (v, i) = regs::i2c_read(i2c_addr);
        self.transport
            .control_in(ControlRequest::read(v, i), buf)
            .await?;
        Ok(())
    }

    // -- General purpose pins -----------------------------------------------

    /// Configures a pin as an output.
    pub async fn set_gpio_output(&mut self, gpio: u8) -> Result<(), TransportError> {
        let mask = 1u16 << gpio;
        let direction = self.read_reg(Block::Sys, regs::sys::GPD, 1).await?;
        self.write_reg(Block::Sys, regs::sys::GPD, direction & !mask, 1)
            .await?;
        let enable = self.read_reg(Block::Sys, regs::sys::GPOE, 1).await?;
        self.write_reg(Block::Sys, regs::sys::GPOE, enable | mask, 1)
            .await
    }

    /// Drives a pin high or low.
    pub async fn write_gpio(&mut self, gpio: u8, high: bool) -> Result<(), TransportError> {
        let mask = 1u16 << gpio;
        let current = self.read_reg(Block::Sys, regs::sys::GPO, 1).await?;
        let updated = if high {
            current | mask
        } else {
            current & !mask
        };
        self.write_reg(Block::Sys, regs::sys::GPO, updated, 1).await
    }

    /// Powers the antenna feed. Leave it off unless something upstream needs it — it puts
    /// 4.5 V on the connector, which not every antenna appreciates.
    pub async fn set_bias_tee(&mut self, enabled: bool) -> Result<(), TransportError> {
        self.set_gpio_output(0).await?;
        self.write_gpio(0, enabled).await
    }

    // -- Initialisation -----------------------------------------------------

    /// Brings the demodulator up into software-defined-radio mode.
    ///
    /// The order matters throughout. Powering the demodulator before the USB endpoint is
    /// configured, or programming the filter before the soft reset, leaves the part in a
    /// state that only shows up much later as a stream that never starts.
    pub async fn init_baseband(&mut self) -> Result<(), TransportError> {
        // USB endpoint first, so the bulk path exists before anything can drive it.
        self.write_reg(Block::Usb, regs::usb::SYSCTL, 0x09, 1)
            .await?;
        self.write_reg(Block::Usb, regs::usb::EPA_MAXPKT, 0x0002, 2)
            .await?;
        self.write_reg(Block::Usb, regs::usb::EPA_CTL, 0x1002, 2)
            .await?;

        // Power up the demodulator.
        self.write_reg(Block::Sys, regs::sys::DEMOD_CTL_1, 0x22, 1)
            .await?;
        self.write_reg(Block::Sys, regs::sys::DEMOD_CTL, 0xE8, 1)
            .await?;

        // Soft reset: assert then release.
        self.demod_write(1, 0x01, 0x14, 1).await?;
        self.demod_write(1, 0x01, 0x10, 1).await?;

        // Clear the spectrum-inversion and offset registers before they are set properly.
        self.demod_write(1, 0x15, 0x00, 1).await?;
        self.demod_write(1, 0x16, 0x0000, 2).await?;
        for offset in 0..6u16 {
            self.demod_write(1, 0x16 + offset, 0x00, 1).await?;
        }

        // Decimating filter, twenty bytes across consecutive registers.
        let fir = pack_fir(&FIR_COEFFICIENTS);
        for (offset, byte) in fir.iter().enumerate() {
            self.demod_write(1, 0x1C + offset as u16, *byte as u16, 1)
                .await?;
        }

        // Software-defined-radio mode, with the digital gain control off.
        self.demod_write(0, 0x19, 0x05, 1).await?;

        // State machine thresholds.
        self.demod_write(1, 0x93, 0xF0, 1).await?;
        self.demod_write(1, 0x94, 0x0F, 1).await?;

        // Disable the broadcast television automatic gain loops, which would otherwise
        // fight whatever gain the tuner is set to.
        self.demod_write(1, 0x11, 0x00, 1).await?;
        self.demod_write(1, 0x04, 0x00, 1).await?;

        // No transport-stream filtering, and the default converter data path.
        self.demod_write(0, 0x61, 0x60, 1).await?;
        self.demod_write(0, 0x06, 0x80, 1).await?;

        // Zero intermediate frequency with offset cancellation and quadrature correction.
        self.demod_write(1, 0xB1, 0x1B, 1).await?;

        // Stop driving the 4.096 MHz test clock onto a pin.
        self.demod_write(0, 0x0D, 0x83, 1).await?;

        Ok(())
    }

    /// Switches the data path to suit an R82xx tuner.
    ///
    /// These parts present a real intermediate frequency rather than quadrature baseband,
    /// so the converter runs single-ended and the demodulator undoes both the offset and
    /// the spectral flip the tuner's high-side mixing introduces.
    pub async fn configure_for_r82xx(&mut self) -> Result<(), TransportError> {
        self.demod_write(1, 0xB1, 0x1A, 1).await?;
        self.demod_write(0, 0x08, 0x4D, 1).await?;
        self.set_if_frequency(DEFAULT_IF_HZ).await?;
        // The tuner mixes from above, which inverts the spectrum; this puts it back.
        self.demod_write(1, 0x15, 0x01, 1).await?;
        Ok(())
    }

    /// Programs the demodulator's intermediate frequency offset.
    pub async fn set_if_frequency(&mut self, if_hz: u32) -> Result<(), TransportError> {
        let [hi, mid, lo] = regs::if_frequency(if_hz, self.xtal);
        self.demod_write(1, 0x19, hi as u16, 1).await?;
        self.demod_write(1, 0x1A, mid as u16, 1).await?;
        self.demod_write(1, 0x1B, lo as u16, 1).await?;
        Ok(())
    }

    /// Programs the sample rate, returning the rate actually achieved.
    ///
    /// The requested rate is a ratio against the crystal and rarely lands exactly, so the
    /// caller has to use the returned value for anything rate-dependent downstream.
    pub async fn set_sample_rate(&mut self, rate: u32) -> Result<f64, TransportError> {
        let setting = regs::sample_rate(rate, self.xtal)
            .map_err(|e| TransportError::Io(format!("unsupported sample rate: {e:?}")))?;

        self.demod_write(1, 0x9F, setting.high, 2).await?;
        self.demod_write(1, 0xA1, setting.low, 2).await?;
        self.set_frequency_correction(0).await?;

        // The resampler needs a reset before the new ratio takes effect.
        self.demod_write(1, 0x01, 0x14, 1).await?;
        self.demod_write(1, 0x01, 0x10, 1).await?;

        Ok(regs::achieved_rate(setting.ratio, self.xtal))
    }

    /// Trims the sample clock, in parts per million.
    pub async fn set_frequency_correction(&mut self, ppm: i32) -> Result<(), TransportError> {
        let [hi, lo] = regs::frequency_correction(ppm);
        self.demod_write(1, 0x3E, hi as u16, 1).await?;
        self.demod_write(1, 0x3F, lo as u16, 1).await?;
        Ok(())
    }

    /// Enables the demodulator's own digital gain control.
    ///
    /// Independent of the tuner's gain, and usually best left off — it operates after the
    /// converter, so it cannot recover anything the tuner already clipped.
    pub async fn set_agc(&mut self, enabled: bool) -> Result<(), TransportError> {
        self.demod_write(0, 0x19, if enabled { 0x25 } else { 0x05 }, 1)
            .await
    }

    // -- Streaming ----------------------------------------------------------

    /// Clears whatever the endpoint has buffered.
    ///
    /// Without this the first read returns samples captured before the current tuning,
    /// which shows up as a fraction of a second of the previous frequency at every retune.
    pub async fn reset_buffer(&mut self) -> Result<(), TransportError> {
        self.write_reg(Block::Usb, regs::usb::EPA_CTL, 0x1002, 2)
            .await?;
        self.write_reg(Block::Usb, regs::usb::EPA_CTL, 0x0000, 2)
            .await
    }

    /// Reads raw interleaved samples, returning how many bytes arrived.
    pub async fn read_samples(&mut self, buf: &mut [u8]) -> Result<usize, TransportError> {
        self.transport.bulk_in(buf).await
    }

    /// Checks that a tuner is present and answering at the given address.
    pub async fn probe_tuner(&mut self, i2c_addr: u8) -> Result<bool, TransportError> {
        self.set_i2c_repeater(true).await?;
        let mut buf = [0u8; 1];
        let result = self.tuner_read(i2c_addr, &mut buf).await;
        self.set_i2c_repeater(false).await?;
        result?;
        Ok(buf[0] == R82XX_CHECK_VALUE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;

    /// Minimal executor. Nothing in the driver suspends against a mock transport.
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

    fn driver() -> Rtl2832<MockTransport> {
        let mut d = Rtl2832::new(MockTransport::new());
        // The read-back after each write triples the log length and obscures what the
        // ordering assertions are actually about.
        d.set_verify_demod_writes(false);
        d
    }

    #[test]
    fn fir_packing_matches_the_register_layout() {
        let packed = pack_fir(&FIR_COEFFICIENTS);
        assert_eq!(packed.len(), 20);

        // First eight are plain signed bytes.
        assert_eq!(packed[0], (-54i32) as i8 as u8);
        assert_eq!(packed[7], 53);

        // The next eight are twelve-bit, three bytes per pair. Unpack and compare.
        for pair in 0..4 {
            let base = 8 + pair * 3;
            let first = ((packed[base] as u32) << 4) | ((packed[base + 1] as u32) >> 4);
            let second = (((packed[base + 1] as u32) & 0x0F) << 8) | packed[base + 2] as u32;
            assert_eq!(
                first as i32,
                FIR_COEFFICIENTS[8 + pair * 2],
                "pair {pair} first"
            );
            assert_eq!(
                second as i32,
                FIR_COEFFICIENTS[8 + pair * 2 + 1],
                "pair {pair} second"
            );
        }
    }

    #[test]
    fn init_configures_usb_before_powering_the_demodulator() {
        let mut d = driver();
        block_on(d.init_baseband()).unwrap();
        let t = d.into_transport();

        let usb = t
            .position_of(regs::usb::SYSCTL, 0x0110)
            .expect("USB SYSCTL never written");
        let power = t
            .position_of(regs::sys::DEMOD_CTL, 0x0210)
            .expect("demodulator never powered");
        assert!(
            usb < power,
            "powered the demodulator before configuring the endpoint"
        );
    }

    #[test]
    fn init_writes_the_documented_startup_values() {
        let mut d = driver();
        block_on(d.init_baseband()).unwrap();
        let t = d.into_transport();

        assert!(t.wrote(regs::usb::SYSCTL, 0x0110, &[0x09]));
        assert!(t.wrote(regs::usb::EPA_MAXPKT, 0x0110, &[0x00, 0x02]));
        assert!(t.wrote(regs::usb::EPA_CTL, 0x0110, &[0x10, 0x02]));
        assert!(t.wrote(regs::sys::DEMOD_CTL_1, 0x0210, &[0x22]));
        assert!(t.wrote(regs::sys::DEMOD_CTL, 0x0210, &[0xE8]));

        // Software-defined-radio mode, and the test clock switched off.
        let (v, i) = regs::demod_write(0, 0x19);
        assert!(t.wrote(v, i, &[0x05]), "never entered SDR mode");
        let (v, i) = regs::demod_write(0, 0x0D);
        assert!(t.wrote(v, i, &[0x83]));
    }

    #[test]
    fn soft_reset_asserts_then_releases() {
        let mut d = driver();
        block_on(d.init_baseband()).unwrap();
        let t = d.into_transport();

        let (v, i) = regs::demod_write(1, 0x01);
        let resets: Vec<_> = t
            .writes()
            .into_iter()
            .filter(|(wv, wi, _)| *wv == v && *wi == i)
            .map(|(_, _, data)| data[0])
            .collect();
        assert_eq!(
            resets,
            vec![0x14, 0x10],
            "reset was not asserted then released"
        );
    }

    #[test]
    fn init_programs_all_twenty_filter_bytes() {
        let mut d = driver();
        block_on(d.init_baseband()).unwrap();
        let t = d.into_transport();

        let expected = pack_fir(&FIR_COEFFICIENTS);
        for (offset, byte) in expected.iter().enumerate() {
            let (v, i) = regs::demod_write(1, 0x1C + offset as u16);
            assert!(
                t.wrote(v, i, &[*byte]),
                "filter byte {offset} (0x{byte:02X}) never written"
            );
        }
    }

    #[test]
    fn r82xx_configuration_inverts_the_spectrum() {
        // High-side mixing in the tuner flips the spectrum; leaving this unset puts every
        // signal on the wrong side of centre.
        let mut d = driver();
        block_on(d.configure_for_r82xx()).unwrap();
        let t = d.into_transport();

        let (v, i) = regs::demod_write(1, 0x15);
        assert!(t.wrote(v, i, &[0x01]), "spectrum inversion never enabled");

        let (v, i) = regs::demod_write(1, 0xB1);
        assert!(t.wrote(v, i, &[0x1A]), "zero-IF path not disabled");

        let (v, i) = regs::demod_write(0, 0x08);
        assert!(
            t.wrote(v, i, &[0x4D]),
            "converter not switched to single-ended"
        );
    }

    #[test]
    fn sample_rate_writes_both_halves_and_resets() {
        let mut d = driver();
        let actual = block_on(d.set_sample_rate(2_400_000)).unwrap();
        assert!((actual - 2_400_000.0).abs() < 1.0, "achieved rate {actual}");

        let t = d.into_transport();
        let (v, i) = regs::demod_write(1, 0x9F);
        assert!(t.wrote(v, i, &[0x03, 0x00]), "high half of the ratio wrong");
        let (v, i) = regs::demod_write(1, 0xA1);
        assert!(t.wrote(v, i, &[0x00, 0x00]), "low half of the ratio wrong");

        // The ratio only takes effect after a reset.
        let (rv, ri) = regs::demod_write(1, 0x01);
        assert!(t.wrote(rv, ri, &[0x14]));
    }

    #[test]
    fn an_unsupported_sample_rate_is_refused_before_any_transfer() {
        let mut d = driver();
        let err = block_on(d.set_sample_rate(500_000)).unwrap_err();
        assert!(matches!(err, TransportError::Io(_)));
        assert!(
            d.into_transport().log.is_empty(),
            "touched the device before validating the rate"
        );
    }

    #[test]
    fn if_frequency_writes_three_registers() {
        let mut d = driver();
        block_on(d.set_if_frequency(DEFAULT_IF_HZ)).unwrap();
        let t = d.into_transport();

        let expected = regs::if_frequency(DEFAULT_IF_HZ, regs::RTL_XTAL);
        for (offset, byte) in expected.iter().enumerate() {
            let (v, i) = regs::demod_write(1, 0x19 + offset as u16);
            assert!(t.wrote(v, i, &[*byte]), "IF byte {offset} wrong");
        }
    }

    #[test]
    fn i2c_repeater_toggles_the_documented_values() {
        let mut d = driver();
        block_on(async {
            d.set_i2c_repeater(true).await.unwrap();
            d.set_i2c_repeater(false).await.unwrap();
        });
        let t = d.into_transport();
        let (v, i) = regs::demod_write(1, 0x01);
        let values: Vec<_> = t
            .writes()
            .into_iter()
            .filter(|(wv, wi, _)| *wv == v && *wi == i)
            .map(|(_, _, data)| data[0])
            .collect();
        assert_eq!(values, vec![0x18, 0x10]);
    }

    #[test]
    fn tuner_reads_are_passed_through_unaltered() {
        // Measured against a Blog V4: register 0 arrives as 0x69, the value that
        // identifies the part. Reversing the bits on the way through — as several
        // reference implementations do — would turn it into 0x96 and make the tuner
        // look absent.
        let mut d = driver();
        d.transport_mut().push_response([R82XX_CHECK_VALUE]);

        let mut buf = [0u8; 1];
        block_on(d.tuner_read(R828D_I2C_ADDR, &mut buf)).unwrap();
        assert_eq!(buf[0], R82XX_CHECK_VALUE);
    }

    #[test]
    fn probing_finds_a_tuner_that_answers_correctly() {
        let mut d = driver();
        d.transport_mut().push_response([R82XX_CHECK_VALUE]);
        assert!(block_on(d.probe_tuner(R828D_I2C_ADDR)).unwrap());

        let mut absent = driver();
        absent.transport_mut().push_response([0x00u8]);
        assert!(!block_on(absent.probe_tuner(R828D_I2C_ADDR)).unwrap());
    }

    #[test]
    fn probing_brackets_the_read_with_the_repeater() {
        let mut d = driver();
        d.transport_mut().push_response([R82XX_CHECK_VALUE]);
        block_on(d.probe_tuner(R828D_I2C_ADDR)).unwrap();

        let t = d.into_transport();
        let (v, i) = regs::demod_write(1, 0x01);
        let toggles: Vec<_> = t
            .writes()
            .into_iter()
            .filter(|(wv, wi, _)| *wv == v && *wi == i)
            .map(|(_, _, data)| data[0])
            .collect();
        assert_eq!(
            toggles,
            vec![0x18, 0x10],
            "repeater not opened and closed around the read"
        );
    }

    #[test]
    fn tuner_writes_carry_register_then_value() {
        let mut d = driver();
        block_on(d.tuner_write(R828D_I2C_ADDR, 0x1A, 0x40)).unwrap();
        let t = d.into_transport();
        let (v, i) = regs::i2c_write(R828D_I2C_ADDR);
        assert!(t.wrote(v, i, &[0x1A, 0x40]));
    }

    #[test]
    fn burst_writes_prefix_the_starting_register() {
        let mut d = driver();
        block_on(d.tuner_write_burst(R828D_I2C_ADDR, 0x05, &[0x83, 0x30, 0x75])).unwrap();
        let t = d.into_transport();
        let (v, i) = regs::i2c_write(R828D_I2C_ADDR);
        assert!(t.wrote(v, i, &[0x05, 0x83, 0x30, 0x75]));
    }

    #[test]
    fn long_bursts_are_split_and_the_register_advances() {
        // The bus master stalls on anything longer than eight bytes including the address,
        // so a twenty-seven register run has to become several messages.
        let mut d = driver();
        let values: Vec<u8> = (0..27).collect();
        block_on(d.tuner_write_burst(R828D_I2C_ADDR, 0x05, &values)).unwrap();

        let t = d.into_transport();
        let (v, i) = regs::i2c_write(R828D_I2C_ADDR);
        let messages: Vec<Vec<u8>> = t
            .writes()
            .into_iter()
            .filter(|(wv, wi, _)| *wv == v && *wi == i)
            .map(|(_, _, data)| data)
            .collect();

        assert!(messages.len() > 1, "a 27-byte run was not split");
        for message in &messages {
            assert!(
                message.len() <= MAX_I2C_MSG_LEN,
                "message of {} bytes exceeds the bus limit",
                message.len()
            );
        }

        // Reassembling the payloads must reproduce the original run, with each message
        // addressed to where its slice belongs.
        let mut rebuilt = Vec::new();
        for message in &messages {
            assert_eq!(
                message[0],
                0x05 + rebuilt.len() as u8,
                "message addressed to the wrong register"
            );
            rebuilt.extend_from_slice(&message[1..]);
        }
        assert_eq!(rebuilt, values);
    }

    #[test]
    fn buffer_reset_cycles_the_endpoint() {
        let mut d = driver();
        block_on(d.reset_buffer()).unwrap();
        let t = d.into_transport();
        let writes: Vec<_> = t
            .writes()
            .into_iter()
            .filter(|(v, i, _)| *v == regs::usb::EPA_CTL && *i == 0x0110)
            .map(|(_, _, data)| data)
            .collect();
        assert_eq!(writes, vec![vec![0x10, 0x02], vec![0x00, 0x00]]);
    }

    #[test]
    fn verified_writes_add_a_read_back() {
        let mut plain = Rtl2832::new(MockTransport::new());
        plain.set_verify_demod_writes(false);
        block_on(plain.demod_write(1, 0x01, 0x10, 1)).unwrap();
        assert_eq!(plain.into_transport().log.len(), 1);

        let mut verified = Rtl2832::new(MockTransport::new());
        verified.set_verify_demod_writes(true);
        block_on(verified.demod_write(1, 0x01, 0x10, 1)).unwrap();
        assert_eq!(
            verified.into_transport().log.len(),
            2,
            "read-back missing, which is what doubles the round trips"
        );
    }

    #[test]
    fn gpio_configuration_sets_direction_before_driving() {
        let mut d = driver();
        block_on(d.set_bias_tee(true)).unwrap();
        let t = d.into_transport();

        let direction = t
            .position_of(regs::sys::GPD, 0x0210)
            .expect("direction never set");
        let output = t
            .position_of(regs::sys::GPO, 0x0210)
            .expect("pin never driven");
        assert!(
            direction < output,
            "drove the pin before making it an output"
        );
    }

    #[test]
    fn transport_failures_propagate() {
        let mut d = driver();
        d.transport_mut().fail_after = Some(0);
        assert!(block_on(d.init_baseband()).is_err());
    }
}
