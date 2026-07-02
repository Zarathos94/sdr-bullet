//! WebAssembly entry points for the receiver pipeline.
//!
//! The pipeline is split across workers, and each worker instantiates this module
//! separately with its own linear memory. That is deliberate rather than a workaround:
//! shared-memory threading needs an atomics-enabled standard library, which needs a
//! nightly toolchain and `build-std`, and produces a module that fails in ways that are
//! very hard to see. Independent instances joined by ring buffers give real parallelism on
//! stable Rust, and the pipeline is a chain of stages anyway — the natural split is by
//! stage, not by data.
//!
//! # Buffer lifetime
//!
//! Callers get raw pointers into this module's memory and build typed-array views over
//! them. Growing the WebAssembly heap **detaches every such view**, silently turning them
//! into zero-length arrays. So every buffer here is allocated once, at construction, from
//! sizes fixed then — nothing in a `process` call may allocate. That constraint is why the
//! stages take a maximum block size up front rather than sizing to whatever arrives.

use wasm_bindgen::prelude::*;

use sdr_dsp::agc::{Agc, Deemphasis, Emphasis};
use sdr_dsp::demod::{AmDemod, FmDemod, Mode, RdsDecoder, Sideband, SsbDemod, StereoDecoder};
use sdr_dsp::fft::{self, Fft};
use sdr_dsp::fir::Decimator;
use sdr_dsp::iq::IqConverter;
use sdr_dsp::nco::Nco;
use sdr_dsp::resample::RationalResampler;
use sdr_dsp::window::Window;

/// Which vector backend the module was built with, for the diagnostics panel.
///
/// WebAssembly has no runtime feature detection, so a module either requires vector support
/// or does not. Surfacing it makes a scalar fallback visible rather than merely slow.
#[wasm_bindgen]
pub fn simd_backend() -> String {
    sdr_dsp::simd::backend().to_string()
}

/// Demodulation modes, mirrored for JavaScript.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemodMode {
    Nfm = 0,
    Wfm = 1,
    Am = 2,
    Usb = 3,
    Lsb = 4,
    Cw = 5,
}

impl From<DemodMode> for Mode {
    fn from(m: DemodMode) -> Self {
        match m {
            DemodMode::Nfm => Mode::Nfm,
            DemodMode::Wfm => Mode::Wfm,
            DemodMode::Am => Mode::Am,
            DemodMode::Usb => Mode::Usb,
            DemodMode::Lsb => Mode::Lsb,
            DemodMode::Cw => Mode::Cw,
        }
    }
}

// ---------------------------------------------------------------------------
// Capture stage
// ---------------------------------------------------------------------------

/// Unpacks receiver bytes into corrected, deinterleaved baseband.
///
/// Runs in the worker that owns the USB endpoint, so the expensive part of a transfer is
/// already done by the time the samples reach a ring buffer.
#[wasm_bindgen]
#[derive(Debug)]
pub struct CaptureStage {
    converter: IqConverter,
    raw: Vec<u8>,
    i: Vec<f32>,
    q: Vec<f32>,
}

#[wasm_bindgen]
impl CaptureStage {
    /// `max_bytes` bounds a single transfer and fixes every buffer size for good.
    #[wasm_bindgen(constructor)]
    pub fn new(max_bytes: usize) -> Self {
        let complex = max_bytes / 2;
        Self {
            converter: IqConverter::new(),
            raw: vec![0; max_bytes],
            i: vec![0.0; complex],
            q: vec![0.0; complex],
        }
    }

    /// Where to write the raw transfer before calling [`CaptureStage::process`].
    pub fn input_ptr(&self) -> *const u8 {
        self.raw.as_ptr()
    }

    pub fn input_capacity(&self) -> usize {
        self.raw.len()
    }

    pub fn i_ptr(&self) -> *const f32 {
        self.i.as_ptr()
    }

    pub fn q_ptr(&self) -> *const f32 {
        self.q.as_ptr()
    }

    /// Converts the first `bytes` of the input buffer, returning the sample count.
    pub fn process(&mut self, bytes: usize) -> usize {
        let bytes = bytes.min(self.raw.len()) & !1;
        // Split the borrow so the converter can hold the input while writing the outputs.
        let raw = core::mem::take(&mut self.raw);
        let n = self
            .converter
            .process(&raw[..bytes], &mut self.i, &mut self.q);
        self.raw = raw;
        n
    }

    /// Clears the adaptive corrections. Call after retuning — the offsets are
    /// frequency-dependent and a stale estimate is worse than none.
    pub fn reset(&mut self) {
        self.converter.reset();
    }

    pub fn dc_i(&self) -> f32 {
        self.converter.corrections().dc_i
    }

    pub fn dc_q(&self) -> f32 {
        self.converter.corrections().dc_q
    }
}

// ---------------------------------------------------------------------------
// Channel stage
// ---------------------------------------------------------------------------

/// Selects a channel out of the captured bandwidth and demodulates it.
#[wasm_bindgen]
#[derive(Debug)]
pub struct ChannelStage {
    capture_rate: f32,
    channel_rate: f32,
    audio_rate: f32,
    mode: Mode,

    nco: Nco,
    decimate_i: Decimator,
    decimate_q: Decimator,

    fm: FmDemod,
    am: AmDemod,
    ssb: SsbDemod,
    stereo: StereoDecoder,
    rds: RdsDecoder,
    deemphasis_left: Deemphasis,
    deemphasis_right: Deemphasis,
    resample_left: RationalResampler,
    resample_right: RationalResampler,
    agc: Agc,

    // Every buffer below is sized once and never resized. See the module docs.
    in_i: Vec<f32>,
    in_q: Vec<f32>,
    channel_i: Vec<f32>,
    channel_q: Vec<f32>,
    demodulated: Vec<f32>,
    left: Vec<f32>,
    right: Vec<f32>,
    pilot_phase: Vec<f32>,
    audio_left: Vec<f32>,
    audio_right: Vec<f32>,

    stereo_detected: bool,
    squelch_open: bool,
    audio_frames: usize,
}

#[wasm_bindgen]
impl ChannelStage {
    /// Builds the chain for a capture rate and a maximum input block.
    ///
    /// The channel rate is chosen so wideband FM has room for its subcarriers while
    /// narrower modes are not carrying bandwidth they will only throw away.
    #[wasm_bindgen(constructor)]
    pub fn new(capture_rate: f32, max_samples: usize, mode: DemodMode) -> Self {
        let mode: Mode = mode.into();
        let audio_rate = 48_000.0;

        // Wideband FM needs the stereo pilot at 19 kHz and the data subcarrier at 57 kHz,
        // so its channel rate cannot go below about 150 kHz whatever the audio needs.
        let target_channel_rate = if mode.is_stereo_capable() {
            240_000.0
        } else {
            48_000.0
        };
        let decimation = ((capture_rate / target_channel_rate).round() as usize).max(1);
        let channel_rate = capture_rate / decimation as f32;

        let channel_max = max_samples / decimation + 2;
        let audio_max = (channel_max as f32 * audio_rate / channel_rate) as usize + 64;

        Self {
            capture_rate,
            channel_rate,
            audio_rate,
            mode,

            nco: Nco::new(capture_rate as f64),
            decimate_i: Decimator::lowpass(decimation, 63),
            decimate_q: Decimator::lowpass(decimation, 63),

            fm: FmDemod::new(
                channel_rate,
                if mode == Mode::Wfm { 75_000.0 } else { 5_000.0 },
            ),
            am: AmDemod::new(channel_rate),
            ssb: SsbDemod::new(
                channel_rate,
                if mode == Mode::Lsb {
                    Sideband::Lower
                } else {
                    Sideband::Upper
                },
                mode.bandwidth().min(channel_rate * 0.4),
            ),
            stereo: StereoDecoder::new(channel_rate.max(90_001.0)),
            rds: RdsDecoder::new(channel_rate.max(122_001.0)),
            deemphasis_left: Deemphasis::new(channel_rate, Emphasis::Us50),
            deemphasis_right: Deemphasis::new(channel_rate, Emphasis::Us50),
            resample_left: RationalResampler::for_rates(channel_rate as f64, audio_rate as f64, 16),
            resample_right: RationalResampler::for_rates(
                channel_rate as f64,
                audio_rate as f64,
                16,
            ),
            agc: Agc::new(audio_rate),

            in_i: vec![0.0; max_samples],
            in_q: vec![0.0; max_samples],
            channel_i: vec![0.0; channel_max],
            channel_q: vec![0.0; channel_max],
            demodulated: vec![0.0; channel_max],
            left: vec![0.0; channel_max],
            right: vec![0.0; channel_max],
            pilot_phase: vec![0.0; channel_max],
            audio_left: vec![0.0; audio_max],
            audio_right: vec![0.0; audio_max],

            stereo_detected: false,
            squelch_open: true,
            audio_frames: 0,
        }
    }

    pub fn i_ptr(&self) -> *const f32 {
        self.in_i.as_ptr()
    }

    pub fn q_ptr(&self) -> *const f32 {
        self.in_q.as_ptr()
    }

    pub fn input_capacity(&self) -> usize {
        self.in_i.len()
    }

    pub fn audio_left_ptr(&self) -> *const f32 {
        self.audio_left.as_ptr()
    }

    pub fn audio_right_ptr(&self) -> *const f32 {
        self.audio_right.as_ptr()
    }

    pub fn capture_rate(&self) -> f32 {
        self.capture_rate
    }

    pub fn channel_rate(&self) -> f32 {
        self.channel_rate
    }

    pub fn audio_rate(&self) -> f32 {
        self.audio_rate
    }

    pub fn stereo(&self) -> bool {
        self.stereo_detected
    }

    pub fn squelch_open(&self) -> bool {
        self.squelch_open
    }

    /// Offset of the wanted channel from the tuned centre, in hertz.
    ///
    /// Shifting in software rather than retuning the hardware means the display stays
    /// still while the channel moves, and avoids a retune's settling time per step.
    pub fn set_channel_offset(&mut self, hz: f32) {
        self.nco.set_frequency(-hz as f64);
    }

    pub fn set_squelch(&mut self, enabled: bool, threshold: f32) {
        let squelch = self.fm.squelch_mut();
        squelch.set_enabled(enabled);
        if threshold > 0.0 {
            squelch.set_threshold(threshold);
        }
    }

    pub fn set_deemphasis_us(&mut self, microseconds: u32) {
        let emphasis = match microseconds {
            75 => Emphasis::Us75,
            50 => Emphasis::Us50,
            _ => Emphasis::None,
        };
        self.deemphasis_left = Deemphasis::new(self.channel_rate, emphasis);
        self.deemphasis_right = Deemphasis::new(self.channel_rate, emphasis);
    }

    pub fn set_forced_mono(&mut self, mono: bool) {
        self.stereo.set_forced_mono(mono);
    }

    pub fn set_agc_enabled(&mut self, enabled: bool) {
        self.agc.set_enabled(enabled);
    }

    /// Demodulates `samples` of input, returning the number of audio frames produced.
    pub fn process(&mut self, samples: usize) -> usize {
        let n = samples.min(self.in_i.len());
        if n == 0 {
            self.audio_frames = 0;
            return 0;
        }

        // Shift the wanted channel to zero, then decimate to the channel rate.
        self.nco.mix_down(&mut self.in_i[..n], &mut self.in_q[..n]);
        let channel_n = self
            .decimate_i
            .process(&self.in_i[..n], &mut self.channel_i);
        self.decimate_q
            .process(&self.in_q[..n], &mut self.channel_q);

        let i = &self.channel_i[..channel_n];
        let q = &self.channel_q[..channel_n];

        match self.mode {
            Mode::Wfm => {
                self.squelch_open = self.fm.process(i, q, &mut self.demodulated);
                let mpx = &self.demodulated[..channel_n];
                self.stereo_detected = self.stereo.process(mpx, &mut self.left, &mut self.right);

                // The data subcarrier is suppressed, so its only phase reference is three
                // times the stereo pilot. That makes data decoding conditional on the
                // pilot loop having acquired — with no pilot there is nothing to lock to.
                if self.stereo.is_locked() {
                    let phase = self.stereo.pilot_phase();
                    let usable = channel_n.min(phase.len());
                    self.pilot_phase[..usable].copy_from_slice(&phase[..usable]);
                    self.rds
                        .process(&mpx[..usable], &self.pilot_phase[..usable]);
                }
            }
            Mode::Nfm => {
                self.squelch_open = self.fm.process(i, q, &mut self.demodulated);
                self.mono(channel_n);
            }
            Mode::Am => {
                self.am.process(i, q, &mut self.demodulated);
                self.squelch_open = true;
                self.mono(channel_n);
            }
            Mode::Usb | Mode::Lsb | Mode::Cw => {
                self.ssb.process(i, q, &mut self.demodulated);
                self.squelch_open = true;
                self.mono(channel_n);
            }
        }

        self.deemphasis_left.process(&mut self.left[..channel_n]);
        self.deemphasis_right.process(&mut self.right[..channel_n]);

        let frames = self
            .resample_left
            .process(&self.left[..channel_n], &mut self.audio_left);
        self.resample_right
            .process(&self.right[..channel_n], &mut self.audio_right);

        self.agc.process(&mut self.audio_left[..frames]);
        // Both channels share one gain, or the stereo image would wander with the
        // difference between two independently adapting controls.
        let gain = self.agc.gain();
        for sample in self.audio_right[..frames].iter_mut() {
            *sample *= gain;
        }

        self.audio_frames = frames;
        frames
    }

    /// Copies the single demodulated channel into both outputs.
    fn mono(&mut self, n: usize) {
        self.left[..n].copy_from_slice(&self.demodulated[..n]);
        self.right[..n].copy_from_slice(&self.demodulated[..n]);
        self.stereo_detected = false;
    }

    /// Clears every filter and adaptive state. Call after retuning or changing mode.
    pub fn reset(&mut self) {
        self.nco.reset();
        self.decimate_i.reset();
        self.decimate_q.reset();
        self.fm.reset();
        self.am.reset();
        self.ssb.reset();
        self.stereo.reset();
        self.rds.reset();
        self.deemphasis_left.reset();
        self.deemphasis_right.reset();
        self.resample_left.reset();
        self.resample_right.reset();
        self.agc.reset();
    }

    pub fn pilot_level(&self) -> f32 {
        self.stereo.pilot_level()
    }

    // -- Radio data ---------------------------------------------------------

    pub fn rds_station_name(&self) -> String {
        self.rds.state().station_name.clone()
    }

    pub fn rds_radio_text(&self) -> String {
        self.rds.state().radio_text.clone()
    }

    pub fn rds_program_id(&self) -> u32 {
        self.rds.state().program_id.map(u32::from).unwrap_or(0)
    }

    pub fn rds_block_error_rate(&self) -> f32 {
        self.rds.state().block_error_rate()
    }

    pub fn rds_synchronised(&self) -> bool {
        self.rds.is_synchronised()
    }
}

// ---------------------------------------------------------------------------
// Spectrum stage
// ---------------------------------------------------------------------------

/// Turns baseband into calibrated spectrum rows for the display.
#[wasm_bindgen]
#[derive(Debug)]
pub struct SpectrumStage {
    fft: Fft,
    window: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    bins: Vec<f32>,
    /// Exponential average across frames, which quiets the display without hiding
    /// short-lived signals the way a long average would.
    averaged: Vec<f32>,
    smoothing: f32,
    primed: bool,
}

#[wasm_bindgen]
impl SpectrumStage {
    /// # Panics
    /// If `size` is not a power of two.
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> Self {
        Self {
            fft: Fft::new(size),
            // Sidelobes below -92 dB, so a strong carrier does not smear across the
            // display and hide its quieter neighbours.
            window: Window::BlackmanHarris.periodic(size),
            re: vec![0.0; size],
            im: vec![0.0; size],
            bins: vec![0.0; size],
            averaged: vec![-200.0; size],
            smoothing: 0.5,
            primed: false,
        }
    }

    pub fn size(&self) -> usize {
        self.fft.len()
    }

    pub fn i_ptr(&self) -> *const f32 {
        self.re.as_ptr()
    }

    pub fn q_ptr(&self) -> *const f32 {
        self.im.as_ptr()
    }

    /// Averaged bins in display order: lowest frequency first, centre in the middle.
    pub fn bins_ptr(&self) -> *const f32 {
        self.averaged.as_ptr()
    }

    /// How much of each new frame is taken, from 0 to 1. Lower is smoother and slower.
    pub fn set_smoothing(&mut self, alpha: f32) {
        self.smoothing = alpha.clamp(0.01, 1.0);
    }

    /// Windows and transforms whatever is in the input buffers.
    ///
    /// The input pointers are consumed in place, so a caller must rewrite them before
    /// every call rather than assuming they survive.
    pub fn process(&mut self) {
        let n = self.fft.len();
        for k in 0..n {
            self.re[k] *= self.window[k];
            self.im[k] *= self.window[k];
        }

        self.fft.forward(&mut self.re, &mut self.im);
        self.fft
            .power_db(&self.re, &self.im, &self.window, &mut self.bins);
        // Negative frequencies occupy the upper half of the transform; the display wants
        // them on the left.
        fft::shift(&mut self.bins);

        if self.primed {
            for k in 0..n {
                self.averaged[k] += (self.bins[k] - self.averaged[k]) * self.smoothing;
            }
        } else {
            self.averaged.copy_from_slice(&self.bins);
            self.primed = true;
        }
    }

    pub fn reset(&mut self) {
        self.primed = false;
    }
}
