//! Drives a physically attached dongle so the driver can be validated off-browser.
//!
//! Everything here runs the same `sdr-rtl` register sequences and the same `sdr-dsp` chain
//! that the browser build uses. If a station comes out of `probe fm` as audio, the parts
//! that are hard to debug through three layers of browser sandbox are already known good.

mod transport;

use std::io::Write;

use sdr_dsp::demod::{FmDemod, StereoDecoder};
use sdr_dsp::{Agc, Decimator, Deemphasis, Emphasis, Fft, RationalResampler};
use sdr_rtl::r82xx::{GainMode, R82xx};
use sdr_rtl::regs;
use sdr_rtl::rtl2832::{Rtl2832, R828D_I2C_ADDR};

use transport::NusbTransport;

/// Capture rate. Divides by ten to a 240 kHz multiplex and then by five to 48 kHz of
/// audio, so the whole chain is integer decimation with no fractional resampling.
const CAPTURE_RATE: u32 = 2_400_000;
const MPX_RATE: f32 = 240_000.0;
const AUDIO_RATE: f32 = 48_000.0;

/// Runs a future to completion. Every transport operation blocks internally, so nothing
/// here ever actually suspends and a real executor would earn nothing.
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
    // SAFETY: the future is owned by this frame and never moved after pinning.
    let mut future = unsafe { core::pin::Pin::new_unchecked(&mut future) };
    loop {
        if let Poll::Ready(v) = future.as_mut().poll(&mut cx) {
            return v;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("info");

    let result = match command {
        "info" => cmd_info(),
        "debug" => cmd_debug(),
        "spectrum" => {
            let freq = args.get(2).and_then(|s| parse_frequency(s));
            match freq {
                Some(f) => cmd_spectrum(f),
                None => Err("usage: probe spectrum <frequency>".into()),
            }
        }
        "scan" => cmd_scan(),
        "fm" => {
            let freq = args.get(2).and_then(|s| parse_frequency(s));
            let seconds = args
                .get(3)
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(5.0);
            match freq {
                Some(f) => cmd_fm(f, seconds),
                None => Err("usage: probe fm <frequency, e.g. 98.5M> [seconds]".into()),
            }
        }
        "capture" => {
            let freq = args.get(2).and_then(|s| parse_frequency(s));
            let seconds = args
                .get(3)
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.0);
            match freq {
                Some(f) => cmd_capture(f, seconds),
                None => Err("usage: probe capture <frequency> [seconds]".into()),
            }
        }
        other => Err(format!(
            "unknown command '{other}'. Try info, scan, fm or capture."
        )),
    };

    if let Err(message) = result {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

/// Accepts plain hertz or a suffixed form such as `98.5M` or `7100k`.
fn parse_frequency(text: &str) -> Option<u32> {
    let text = text.trim();
    let (number, scale) = match text.chars().last()? {
        'k' | 'K' => (&text[..text.len() - 1], 1_000.0),
        'M' | 'm' => (&text[..text.len() - 1], 1_000_000.0),
        'G' | 'g' => (&text[..text.len() - 1], 1_000_000_000.0),
        _ => (text, 1.0),
    };
    let value: f64 = number.parse().ok()?;
    Some((value * scale) as u32)
}

fn cmd_info() -> Result<(), String> {
    let devices = transport::list()?;
    if devices.is_empty() {
        return Err("no RTL2832U device found. Is it plugged in?".into());
    }

    for d in &devices {
        println!("bus {} address {}", d.bus, d.address);
        println!(
            "  manufacturer  {}",
            d.manufacturer.as_deref().unwrap_or("(none)")
        );
        println!(
            "  product       {}",
            d.product.as_deref().unwrap_or("(none)")
        );
        println!(
            "  serial        {}",
            d.serial.as_deref().unwrap_or("(none)")
        );
        println!(
            "  identified as {}",
            if d.is_v4() {
                "RTL-SDR Blog V4 (tuner clocked at 28.8 MHz)"
            } else {
                "a generic RTL2832U board (tuner assumed at 16 MHz)"
            }
        );
    }

    let (mut rtl, mut tuner, found) = open_receiver()?;
    println!("\ninitialised.");
    println!("  tuner reference {} Hz", tuner.xtal());

    let locked = block_on(tuner.is_locked(&mut rtl)).map_err(|e| e.to_string())?;
    println!(
        "  synthesiser     {}",
        if locked { "locked" } else { "unlocked" }
    );
    println!(
        "  detection       {}",
        if found.is_v4() {
            "V4 path"
        } else {
            "legacy path"
        }
    );

    Ok(())
}

/// Dumps raw register reads, for working out why a probe is not answering.
fn cmd_debug() -> Result<(), String> {
    use sdr_rtl::transport::{ControlRequest, Transport};

    let (mut usb, found) = NusbTransport::open()?;
    println!(
        "device: {} / {}",
        found.manufacturer.as_deref().unwrap_or("?"),
        found.product.as_deref().unwrap_or("?")
    );

    // Read the USB control register before touching anything, to confirm that plain
    // control transfers work at all.
    let (v, i) = regs::block_read(regs::usb::SYSCTL, regs::Block::Usb);
    let mut buf = [0u8; 1];
    block_on(usb.control_in(ControlRequest::read(v, i), &mut buf))
        .map_err(|e| format!("cannot read USB SYSCTL: {e}"))?;
    println!("USB SYSCTL before init: 0x{:02X}", buf[0]);

    let mut rtl = Rtl2832::new(usb);
    block_on(rtl.init_baseband()).map_err(|e| format!("baseband init failed: {e}"))?;
    println!("baseband initialised");

    // Confirm the demodulator is actually accepting writes by reading one back.
    let readback = block_on(rtl.demod_read(1, 0x01)).map_err(|e| e.to_string())?;
    println!("demod page 1 reg 0x01 (repeater): 0x{readback:02X}");

    block_on(rtl.set_i2c_repeater(true)).map_err(|e| e.to_string())?;
    let readback = block_on(rtl.demod_read(1, 0x01)).map_err(|e| e.to_string())?;
    println!("after opening the repeater:      0x{readback:02X}");

    // The reference implementations all read at least eight bytes even when they want
    // one, without saying why. Try both and compare.
    for len in [1usize, 8] {
        for addr in [0x74u8, 0x34] {
            let (v, i) = regs::i2c_read(addr);
            let mut raw = vec![0u8; len];
            match block_on(
                rtl.transport_mut()
                    .control_in(ControlRequest::read(v, i), &mut raw),
            ) {
                Ok(n) => {
                    let reversed: Vec<u8> = raw.iter().map(|b| b.reverse_bits()).collect();
                    println!(
                        "i2c 0x{addr:02X} len {len}: got {n} bytes  raw {:02X?}  reversed {:02X?}",
                        &raw[..n.min(raw.len())],
                        &reversed[..n.min(reversed.len())]
                    );
                }
                Err(e) => println!("i2c 0x{addr:02X} len {len}: {e}"),
            }
        }
    }

    block_on(rtl.set_i2c_repeater(false)).map_err(|e| e.to_string())?;
    Ok(())
}

/// Opens the device and brings it up to the point where it can stream.
fn open_receiver() -> Result<
    (
        Rtl2832<NusbTransport>,
        R82xx<NusbTransport>,
        transport::Found,
    ),
    String,
> {
    let (usb, found) = NusbTransport::open()?;
    let mut rtl = Rtl2832::new(usb);

    block_on(rtl.init_baseband()).map_err(|e| format!("baseband init failed: {e}"))?;

    let present = block_on(rtl.probe_tuner(R828D_I2C_ADDR))
        .map_err(|e| format!("tuner probe failed: {e}"))?;
    if !present {
        return Err(
            "no R828D tuner answered at 0x74. A V3 board would answer at 0x34 instead.".to_string(),
        );
    }

    // The reference clock is the one thing that cannot be read off the device, so it comes
    // from the descriptor strings.
    let mut tuner = if found.is_v4() {
        R82xx::new_v4()
    } else {
        R82xx::new_legacy_r828d()
    };

    block_on(tuner.init(&mut rtl)).map_err(|e| format!("tuner init failed: {e}"))?;
    block_on(rtl.configure_for_r82xx()).map_err(|e| format!("data path setup failed: {e}"))?;

    Ok((rtl, tuner, found))
}

/// Tunes, streams, and returns raw interleaved bytes.
fn capture(
    rtl: &mut Rtl2832<NusbTransport>,
    tuner: &mut R82xx<NusbTransport>,
    freq_hz: u32,
    bytes: usize,
) -> Result<Vec<u8>, String> {
    block_on(tuner.set_frequency(rtl, freq_hz)).map_err(|e| format!("tuning failed: {e}"))?;

    let mut buf = vec![0u8; bytes];
    // Discard whatever the endpoint had buffered from before this tuning.
    block_on(rtl.reset_buffer()).map_err(|e| format!("buffer reset failed: {e}"))?;
    rtl.transport_mut().start_stream()?;

    // The first transfer after retuning still contains the tail of the previous one.
    let mut warmup = vec![0u8; 65536];
    let _ = block_on(rtl.read_samples(&mut warmup));

    block_on(rtl.read_samples(&mut buf)).map_err(|e| format!("capture failed: {e}"))?;
    rtl.transport_mut().stop_stream();
    Ok(buf)
}

/// Mean power of a capture, in decibels relative to full scale.
fn power_dbfs(bytes: &[u8]) -> f32 {
    let mut sum = 0.0f64;
    for pair in bytes.chunks_exact(2) {
        let i = (pair[0] as f32 - 127.5) / 127.5;
        let q = (pair[1] as f32 - 127.5) / 127.5;
        sum += (i * i + q * q) as f64;
    }
    let mean = sum / (bytes.len() / 2) as f64;
    10.0 * (mean.max(1e-20)).log10() as f32
}

/// Prints the spectrum around a frequency as text, averaged over several captures.
///
/// A picture of the whole captured bandwidth answers a question that a single power
/// reading cannot: whether there are signals at all. A live band shows humps standing
/// well clear of the floor; a disconnected antenna shows a flat line.
fn cmd_spectrum(freq_hz: u32) -> Result<(), String> {
    const SIZE: usize = 1024;
    const AVERAGES: usize = 24;

