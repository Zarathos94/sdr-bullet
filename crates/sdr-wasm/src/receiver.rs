//! The hardware driver, wired to a JavaScript transport.
//!
//! Nothing about the register sequences is repeated here. The same `sdr-rtl` code that was
//! validated against a physical device from a terminal is what runs in the browser, with
//! only the bytes-on-the-wire step swapped out. Reimplementing the sequences in JavaScript
//! would have meant maintaining two copies of the one part of a USB driver that is
//! genuinely difficult, and debugging the browser copy through a sandbox.

use js_sys::{Function, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use sdr_rtl::r82xx::{GainMode, R82xx};
use sdr_rtl::regs;
use sdr_rtl::rtl2832::{Rtl2832, R828D_I2C_ADDR};
use sdr_rtl::transport::{ControlRequest, Transport, TransportError};

/// Bridges the driver's transport onto a JavaScript object.
///
/// Methods are looked up by name on the supplied object rather than going through
/// `web-sys`. That keeps this crate independent of which USB bindings a given `web-sys`
/// release happens to gate behind a feature, and means the same bridge works against a
/// test double as against a real device.
struct JsTransport {
    object: JsValue,
    control_out: Function,
    control_in: Function,
    read_samples: Function,
}

impl core::fmt::Debug for JsTransport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("JsTransport").finish_non_exhaustive()
    }
}

fn method(object: &JsValue, name: &str) -> Result<Function, TransportError> {
    Reflect::get(object, &JsValue::from_str(name))
        .map_err(|_| TransportError::Io(format!("transport has no '{name}'")))?
        .dyn_into::<Function>()
        .map_err(|_| TransportError::Io(format!("transport's '{name}' is not callable")))
}

fn js_error(context: &str, value: JsValue) -> TransportError {
    let detail = value
        .as_string()
        .or_else(|| {
            Reflect::get(&value, &JsValue::from_str("message"))
                .ok()
                .and_then(|m| m.as_string())
        })
        .unwrap_or_else(|| format!("{value:?}"));

    // Disconnection has to be distinguishable, because it is the one failure the pipeline
    // should stop for rather than retry through.
    if detail.contains("device was disconnected") || detail.contains("No device selected") {
        TransportError::Disconnected
    } else if detail.contains("stall") {
        TransportError::Stalled
    } else {
        TransportError::Io(format!("{context}: {detail}"))
    }
}

impl JsTransport {
    fn new(object: JsValue) -> Result<Self, TransportError> {
        Ok(Self {
            control_out: method(&object, "controlOut")?,
            control_in: method(&object, "controlIn")?,
            read_samples: method(&object, "readSamples")?,
            object,
        })
    }
}

impl Transport for JsTransport {
    async fn control_out(
        &mut self,
        request: ControlRequest,
        data: &[u8],
    ) -> Result<(), TransportError> {
        let payload = Uint8Array::new_with_length(data.len() as u32);
        payload.copy_from(data);

        let promise = self
            .control_out
            .call3(
                &self.object,
                &JsValue::from_f64(request.value as f64),
                &JsValue::from_f64(request.index as f64),
                &payload,
            )
            .map_err(|e| js_error("control write", e))?;

        JsFuture::from(js_sys::Promise::from(promise))
            .await
            .map_err(|e| js_error("control write", e))?;
        Ok(())
    }

    async fn control_in(
        &mut self,
        request: ControlRequest,
        data: &mut [u8],
    ) -> Result<usize, TransportError> {
        let promise = self
            .control_in
            .call3(
                &self.object,
                &JsValue::from_f64(request.value as f64),
                &JsValue::from_f64(request.index as f64),
                &JsValue::from_f64(data.len() as f64),
            )
            .map_err(|e| js_error("control read", e))?;

        let result = JsFuture::from(js_sys::Promise::from(promise))
            .await
            .map_err(|e| js_error("control read", e))?;

        let bytes = Uint8Array::new(&result);
        let n = (bytes.length() as usize).min(data.len());
        bytes.slice(0, n as u32).copy_to(&mut data[..n]);
        Ok(n)
    }

    async fn bulk_in(&mut self, data: &mut [u8]) -> Result<usize, TransportError> {
        let promise = self
            .read_samples
            .call0(&self.object)
            .map_err(|e| js_error("sample read", e))?;

        let result = JsFuture::from(js_sys::Promise::from(promise))
            .await
            .map_err(|e| js_error("sample read", e))?;

        let bytes = Uint8Array::new(&result);
        let n = (bytes.length() as usize).min(data.len());
        bytes.slice(0, n as u32).copy_to(&mut data[..n]);
        Ok(n)
    }
}

/// The tuner and demodulator, driven from JavaScript.
#[wasm_bindgen]
#[derive(Debug)]
pub struct Receiver {
    rtl: Rtl2832<JsTransport>,
    tuner: R82xx<JsTransport>,
    tuned_hz: u32,
    sample_rate: f64,
}

