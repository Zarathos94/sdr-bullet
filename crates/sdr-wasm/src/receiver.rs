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

fn to_js(error: TransportError) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[wasm_bindgen]
impl Receiver {
    /// Brings a device up from cold.
    ///
    /// `transport` is an object exposing `controlOut`, `controlIn` and `readSamples`.
    /// `is_v4` comes from the USB descriptor strings on the JavaScript side — it is not
    /// discoverable over the bus, and it decides which reference clock every synthesiser
    /// calculation uses.
    #[wasm_bindgen(js_name = open)]
    pub async fn open(transport: JsValue, is_v4: bool) -> Result<Receiver, JsValue> {
        let bridge = JsTransport::new(transport).map_err(to_js)?;
        let mut rtl = Rtl2832::new(bridge);

        rtl.init_baseband().await.map_err(to_js)?;

        let present = rtl.probe_tuner(R828D_I2C_ADDR).await.map_err(to_js)?;
        if !present {
            return Err(JsValue::from_str(
                "No R828D tuner answered at 0x74. An older V3 board answers at 0x34 and is \
                 not supported by this driver.",
            ));
        }

        let mut tuner = if is_v4 {
            R82xx::new_v4()
        } else {
            R82xx::new_legacy_r828d()
        };
        tuner.init(&mut rtl).await.map_err(to_js)?;
        rtl.configure_for_r82xx().await.map_err(to_js)?;

        Ok(Receiver {
            rtl,
            tuner,
            tuned_hz: 0,
            sample_rate: 0.0,
        })
    }

    /// Programs the sample rate, returning the rate actually achieved.
    ///
    /// The rate is a ratio against a crystal and rarely lands exactly where it was asked,
    /// so everything downstream has to use the returned value.
    #[wasm_bindgen(js_name = setSampleRate)]
    pub async fn set_sample_rate(&mut self, rate: u32) -> Result<f64, JsValue> {
        let actual = self.rtl.set_sample_rate(rate).await.map_err(to_js)?;
        self.sample_rate = actual;
        Ok(actual)
    }

    #[wasm_bindgen(js_name = setFrequency)]
    pub async fn set_frequency(&mut self, hz: u32) -> Result<(), JsValue> {
        self.tuner
            .set_frequency(&mut self.rtl, hz)
            .await
            .map_err(to_js)?;
        self.tuned_hz = hz;
        Ok(())
    }

    /// Fixed gain in tenths of a decibel, or automatic when `tenths` is negative.
    #[wasm_bindgen(js_name = setGain)]
    pub async fn set_gain(&mut self, tenths: i32) -> Result<(), JsValue> {
        let mode = if tenths < 0 {
            GainMode::Automatic
        } else {
            GainMode::Manual(regs::nearest_gain(tenths))
        };
        self.tuner
            .set_gain(&mut self.rtl, mode)
            .await
            .map_err(to_js)
    }

    /// Enables the demodulator's own digital gain control, which is independent of the
    /// tuner's and usually best left off — it runs after the converter, so it cannot
    /// recover anything the tuner has already clipped.
    #[wasm_bindgen(js_name = setDigitalAgc)]
    pub async fn set_digital_agc(&mut self, enabled: bool) -> Result<(), JsValue> {
        self.rtl.set_agc(enabled).await.map_err(to_js)
    }

    #[wasm_bindgen(js_name = setBiasTee)]
    pub async fn set_bias_tee(&mut self, enabled: bool) -> Result<(), JsValue> {
        self.rtl.set_bias_tee(enabled).await.map_err(to_js)
    }

    #[wasm_bindgen(js_name = setFrequencyCorrection)]
    pub async fn set_frequency_correction(&mut self, ppm: i32) -> Result<(), JsValue> {
        self.rtl.set_frequency_correction(ppm).await.map_err(to_js)
    }

    /// Discards whatever the endpoint has buffered.
    ///
    /// Without this the first samples after a retune are still from the previous
    /// frequency, which sounds like a fraction of a second of the old station.
    #[wasm_bindgen(js_name = resetBuffer)]
    pub async fn reset_buffer(&mut self) -> Result<(), JsValue> {
        self.rtl.reset_buffer().await.map_err(to_js)
    }

    /// Whether the tuner's synthesiser has acquired. A false reading after tuning means
    /// the requested frequency is outside what the dividers can reach.
    #[wasm_bindgen(js_name = isLocked)]
    pub async fn is_locked(&mut self) -> Result<bool, JsValue> {
        self.tuner.is_locked(&mut self.rtl).await.map_err(to_js)
    }

    #[wasm_bindgen(getter, js_name = tunedHz)]
    pub fn tuned_hz(&self) -> u32 {
        self.tuned_hz
    }

    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Which port of the triplexer the current frequency arrives on, for the display.
    #[wasm_bindgen(getter, js_name = band)]
    pub fn band(&self) -> String {
        match regs::band_for(self.tuned_hz) {
            regs::Band::Hf => "HF",
            regs::Band::Vhf => "VHF",
            regs::Band::Uhf => "UHF",
        }
        .to_string()
    }

    /// What the synthesiser was actually asked for.
    ///
    /// Below 28.8 MHz this differs from the tuned frequency, because the signal reaches
    /// the tuner through the board's upconverter rather than directly.
    #[wasm_bindgen(getter, js_name = tunerFrequencyHz)]
    pub fn tuner_frequency_hz(&self) -> u32 {
        regs::tuner_frequency(self.tuned_hz)
    }
}

/// Gain settings the tuner can actually reach, in tenths of a decibel.
#[wasm_bindgen(js_name = supportedGains)]
pub fn supported_gains() -> Vec<i32> {
    regs::GAIN_VALUES.to_vec()
}
