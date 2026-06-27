//! libusb-backed transport, so the driver can be exercised without a browser.
//!
//! This is the whole point of keeping the driver generic. The register sequences that will
//! eventually run inside a worker, behind WebUSB, behind cross-origin isolation, run here
//! first against the same hardware from a terminal — where a failure produces a stack
//! trace instead of a silent stream of zeroes.

use std::time::Duration;

use nusb::transfer::{Bulk, ControlIn, ControlOut, ControlType, In, Recipient};
use nusb::{Device, Interface, MaybeFuture};
use sdr_rtl::transport::{ControlRequest, Direction, Transport, TransportError};

/// Vendor and product of every RTL2832U dongle, including the Blog V4.
pub const VID: u16 = 0x0BDA;
pub const PID: u16 = 0x2838;

/// Descriptor strings the Blog V4 reports.
///
/// The only way to tell a V4 from any other RTL2832U board: the identifiers are shared,
/// and the difference that matters — the tuner's reference clock — is not discoverable
/// over the bus.
pub const V4_MANUFACTURER: &str = "RTLSDRBlog";
pub const V4_PRODUCT: &str = "Blog V4";

/// The reference driver keeps 15 transfers of 256 KiB in flight. Fewer than about four and
/// the stream stalls between transfers at 2.4 million samples a second.
const TRANSFERS_IN_FLIGHT: usize = 8;
const BULK_CHUNK: usize = 128 * 1024;

pub struct NusbTransport {
    device: Device,
    reader: Option<Box<dyn std::io::Read>>,
    interface: Interface,
    timeout: Duration,
}

impl std::fmt::Debug for NusbTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NusbTransport")
            .field("streaming", &self.reader.is_some())
            .finish_non_exhaustive()
    }
}

/// What was found on the bus, before anything is opened.
#[derive(Debug, Clone)]
pub struct Found {
    pub manufacturer: Option<String>,
    pub product: Option<String>,
    pub serial: Option<String>,
    pub bus: u8,
    pub address: u8,
}

impl Found {
    /// Whether the descriptors identify this as a Blog V4.
    pub fn is_v4(&self) -> bool {
        self.manufacturer.as_deref() == Some(V4_MANUFACTURER)
            && self.product.as_deref() == Some(V4_PRODUCT)
    }
}

/// Lists every attached RTL2832U device.
pub fn list() -> Result<Vec<Found>, String> {
    let devices = nusb::list_devices()
        .wait()
        .map_err(|e| format!("cannot enumerate USB devices: {e}"))?;

    Ok(devices
        .filter(|d| d.vendor_id() == VID && d.product_id() == PID)
        .map(|d| Found {
            manufacturer: d.manufacturer_string().map(str::to_owned),
            product: d.product_string().map(str::to_owned),
            serial: d.serial_number().map(str::to_owned),
            bus: d.busnum(),
            address: d.device_address(),
        })
        .collect())
}

impl NusbTransport {
    /// Opens the first attached dongle and claims its interface.
    pub fn open() -> Result<(Self, Found), String> {
        let info = nusb::list_devices()
            .wait()
            .map_err(|e| format!("cannot enumerate USB devices: {e}"))?
            .find(|d| d.vendor_id() == VID && d.product_id() == PID)
            .ok_or_else(|| "no RTL2832U device found. Is it plugged in?".to_string())?;

        let found = Found {
            manufacturer: info.manufacturer_string().map(str::to_owned),
            product: info.product_string().map(str::to_owned),
            serial: info.serial_number().map(str::to_owned),
            bus: info.busnum(),
            address: info.device_address(),
        };

        let device = info.open().wait().map_err(|e| {
            format!(
                "cannot open the device: {e}\n\
                 If this is a permission error, the udev rule granting your user access is \
                 probably missing. Installing the rtl-sdr package provides one."
            )
        })?;

        let interface = device.claim_interface(0).wait().map_err(|e| {
            format!(
                "cannot claim the interface: {e}\n\
                 This usually means the kernel's DVB driver has it. Blacklist \
                 dvb_usb_rtl28xxu and unplug and replug the device."
            )
        })?;

        Ok((
            Self {
                device,
                interface,
                reader: None,
                timeout: Duration::from_millis(500),
            },
            found,
        ))
    }

    /// Opens the bulk endpoint and starts the transfer pipeline.
    ///
    /// Kept separate from opening the device because the endpoint has to be started after
    /// the demodulator is configured, not before — a reader running against an
    /// unconfigured device returns whatever the endpoint had buffered from last time.
    pub fn start_stream(&mut self) -> Result<(), String> {
        let endpoint = self
            .interface
            .endpoint::<Bulk, In>(0x81)
            .map_err(|e| format!("cannot open the sample endpoint: {e}"))?;
        let reader = endpoint
            .reader(BULK_CHUNK)
            .with_num_transfers(TRANSFERS_IN_FLIGHT);
        self.reader = Some(Box::new(reader));
        Ok(())
    }

    pub fn stop_stream(&mut self) {
        self.reader = None;
    }
}

fn map_error(e: nusb::transfer::TransferError) -> TransportError {
    use nusb::transfer::TransferError;
    match e {
        TransferError::Disconnected => TransportError::Disconnected,
        TransferError::Stall => TransportError::Stalled,
        other => TransportError::Io(other.to_string()),
    }
}

impl Transport for NusbTransport {
    async fn control_out(
        &mut self,
        request: ControlRequest,
        data: &[u8],
    ) -> Result<(), TransportError> {
        debug_assert_eq!(request.direction, Direction::Out);
        self.device
            .control_out(
                ControlOut {
                    control_type: ControlType::Vendor,
                    recipient: Recipient::Device,
                    request: 0,
                    value: request.value,
                    index: request.index,
                    data,
                },
                self.timeout,
            )
            .wait()
            .map_err(map_error)
    }

    async fn control_in(
        &mut self,
        request: ControlRequest,
        data: &mut [u8],
    ) -> Result<usize, TransportError> {
        debug_assert_eq!(request.direction, Direction::In);
        let response = self
            .device
            .control_in(
                ControlIn {
                    control_type: ControlType::Vendor,
                    recipient: Recipient::Device,
                    request: 0,
                    value: request.value,
                    index: request.index,
                    length: data.len() as u16,
                },
                self.timeout,
            )
            .wait()
            .map_err(map_error)?;

        let n = response.len().min(data.len());
        data[..n].copy_from_slice(&response[..n]);
        Ok(n)
    }

    async fn bulk_in(&mut self, data: &mut [u8]) -> Result<usize, TransportError> {
        use std::io::Read;
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| TransportError::Io("stream not started".into()))?;
        reader
            .read_exact(data)
            .map(|_| data.len())
            .map_err(|e| TransportError::Io(e.to_string()))
    }
}
