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

