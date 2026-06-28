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

    let (mut rtl, mut tuner, _) = open_receiver()?;
    let rate = block_on(rtl.set_sample_rate(CAPTURE_RATE))
        .map_err(|e| format!("cannot set the sample rate: {e}"))?;
    block_on(tuner.set_gain(&mut rtl, GainMode::Automatic))
        .map_err(|e| format!("cannot set gain: {e}"))?;

    let raw = capture(&mut rtl, &mut tuner, freq_hz, SIZE * 2 * AVERAGES)?;

    let fft = Fft::new(SIZE);
    let window = sdr_dsp::window::Window::BlackmanHarris.periodic(SIZE);
    let mut accumulated = vec![0.0f32; SIZE];

    for block in raw.chunks_exact(SIZE * 2) {
        let mut re = vec![0.0f32; SIZE];
        let mut im = vec![0.0f32; SIZE];
        for (k, pair) in block.chunks_exact(2).enumerate() {
            re[k] = (pair[0] as f32 - 127.5) / 127.5 * window[k];
            im[k] = (pair[1] as f32 - 127.5) / 127.5 * window[k];
        }
        fft.forward(&mut re, &mut im);
        let mut db = vec![0.0f32; SIZE];
        fft.power_db(&re, &im, &window, &mut db);
        for k in 0..SIZE {
            accumulated[k] += db[k] / AVERAGES as f32;
        }
    }

    sdr_dsp::fft::shift(&mut accumulated);

    let floor = accumulated.iter().fold(f32::MAX, |a, b| a.min(*b));
    let peak = accumulated.iter().fold(f32::MIN, |a, b| a.max(*b));
    println!(
        "centre {:.3} MHz, span {:.1} MHz",
        freq_hz as f32 / 1e6,
        rate as f32 / 1e6
    );
    println!(
        "floor {floor:.1} dBFS, peak {peak:.1} dBFS, spread {:.1} dB\n",
        peak - floor
    );

    // One row per 16 bins keeps the plot to a screenful.
    for row in 0..SIZE / 16 {
        let bin = row * 16;
        let level = accumulated[bin..bin + 16]
            .iter()
            .fold(f32::MIN, |a, b| a.max(*b));
        let offset_hz = (bin as f32 - SIZE as f32 / 2.0) * rate as f32 / SIZE as f32;
        let normalised = ((level - floor) / (peak - floor).max(1.0) * 50.0) as usize;
        println!(
            "{:>+8.2} MHz {:>7.1} |{}",
            offset_hz / 1e6,
            level,
            "#".repeat(normalised.min(50))
        );
    }

    println!(
        "\n{}",
        if peak - floor > 15.0 {
            "Signals present: the spread is well above the noise floor."
        } else {
            "Flat. This is what a disconnected or badly matched antenna looks like."
        }
    );
    Ok(())
}

fn cmd_scan() -> Result<(), String> {
    let (mut rtl, mut tuner, _) = open_receiver()?;
    let actual = block_on(rtl.set_sample_rate(CAPTURE_RATE))
        .map_err(|e| format!("cannot set the sample rate: {e}"))?;
    println!("sample rate {actual:.0} Hz\n");

    block_on(tuner.set_gain(&mut rtl, GainMode::Automatic))
        .map_err(|e| format!("cannot set gain: {e}"))?;

    // Sweep the broadcast band in steps comfortably inside the captured bandwidth, and
    // measure each channel with a transform rather than by total power, so a strong
    // neighbour does not count as a station here.
    let fft = Fft::new(4096);
    let window = sdr_dsp::window::Window::BlackmanHarris.periodic(4096);

    println!("  frequency     level");
    let mut peaks: Vec<(u32, f32)> = Vec::new();

    let mut freq = 87_500_000u32;
    while freq <= 108_000_000 {
        let bytes = capture(&mut rtl, &mut tuner, freq, 4096 * 2)?;

        let mut re = vec![0.0f32; 4096];
        let mut im = vec![0.0f32; 4096];
        for (k, pair) in bytes.chunks_exact(2).enumerate().take(4096) {
            re[k] = (pair[0] as f32 - 127.5) / 127.5 * window[k];
            im[k] = (pair[1] as f32 - 127.5) / 127.5 * window[k];
        }
        fft.forward(&mut re, &mut im);

        let mut db = vec![0.0f32; 4096];
        fft.power_db(&re, &im, &window, &mut db);

        // Only the bins nearest the centre, which is where the tuned channel sits.
        let centre_level = db[4090..]
            .iter()
            .chain(db[..6].iter())
            .fold(f32::MIN, |a, b| a.max(*b));
        peaks.push((freq, centre_level));

        freq += 200_000;
    }

    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (freq, level) in peaks.iter().take(12) {
        println!("  {:>6.1} MHz   {level:>6.1} dBFS", *freq as f32 / 1e6);
    }

    if let Some((best, _)) = peaks.first() {
        println!("\nstrongest: {:.1} MHz", *best as f32 / 1e6);
        println!(
            "try:  cargo run -p sdr-probe -- fm {:.1}M",
            *best as f32 / 1e6
        );
    }

    Ok(())
}

fn cmd_fm(freq_hz: u32, seconds: f32) -> Result<(), String> {
    let (mut rtl, mut tuner, found) = open_receiver()?;
    let actual = block_on(rtl.set_sample_rate(CAPTURE_RATE))
        .map_err(|e| format!("cannot set the sample rate: {e}"))?;

    block_on(tuner.set_gain(&mut rtl, GainMode::Automatic))
        .map_err(|e| format!("cannot set gain: {e}"))?;

    println!(
        "device      {}",
        if found.is_v4() {
            "Blog V4"
        } else {
            "generic RTL2832U"
        }
    );
    println!("tuned       {:.4} MHz", freq_hz as f32 / 1e6);
    println!("band        {:?}", regs::band_for(freq_hz));
    println!(
        "tuner asked {:.4} MHz",
        regs::tuner_frequency(freq_hz) as f32 / 1e6
    );
    println!("rate        {actual:.0} Hz");

    let bytes_wanted = (seconds * CAPTURE_RATE as f32) as usize * 2;
    // Round up to whole transfers so the decimation chain sees a clean block boundary.
    let bytes_wanted = bytes_wanted.next_multiple_of(1 << 18);
    let raw = capture(&mut rtl, &mut tuner, freq_hz, bytes_wanted)?;

    let locked = block_on(tuner.is_locked(&mut rtl)).map_err(|e| e.to_string())?;
    println!("synth       {}", if locked { "locked" } else { "UNLOCKED" });
    println!("level       {:.1} dBFS", power_dbfs(&raw));

    // Unpack, correct, and drop to the multiplex rate.
    let mut converter = sdr_dsp::IqConverter::new();
    let complex = raw.len() / 2;
    let mut i = vec![0.0f32; complex];
    let mut q = vec![0.0f32; complex];
    converter.process(&raw, &mut i, &mut q);

    let mut dec_i = Decimator::lowpass(10, 63);
    let mut dec_q = Decimator::lowpass(10, 63);
    let mut mpx_i = vec![0.0f32; dec_i.output_len(complex)];
    let mut mpx_q = vec![0.0f32; dec_q.output_len(complex)];
    let n_mpx = dec_i.process(&i, &mut mpx_i);
    dec_q.process(&q, &mut mpx_q);

    // Discriminate, then split the multiplex into two channels.
    let mut demod = FmDemod::new(MPX_RATE, 75_000.0);
    let mut mpx = vec![0.0f32; n_mpx];
    demod.process(&mpx_i[..n_mpx], &mpx_q[..n_mpx], &mut mpx);

    let mut stereo = StereoDecoder::new(MPX_RATE);
    let mut left = vec![0.0f32; n_mpx];
    let mut right = vec![0.0f32; n_mpx];
    let is_stereo = stereo.process(&mpx, &mut left, &mut right);
    println!(
        "stereo      {}",
        if is_stereo {
            "yes"
        } else {
            "no (mono or no pilot)"
        }
    );
    println!("pilot       {:.4}", stereo.pilot_level());

    // De-emphasise, drop to the audio rate, and level.
    let mut audio = Vec::new();
    for channel in [&mut left, &mut right] {
        let mut de = Deemphasis::new(MPX_RATE, Emphasis::Us50);
        de.process(&mut channel[..n_mpx]);

        let mut rs = RationalResampler::for_rates(MPX_RATE as f64, AUDIO_RATE as f64, 16);
        let mut out = vec![0.0f32; rs.output_len(n_mpx)];
        let n = rs.process(&channel[..n_mpx], &mut out);
        out.truncate(n);

        let mut agc = Agc::new(AUDIO_RATE);
        agc.set_target(0.4);
        agc.process(&mut out);
        audio.push(out);
    }

    let path = "captures/fm.wav";
    std::fs::create_dir_all("captures").map_err(|e| e.to_string())?;
    write_wav(path, &audio[0], &audio[1], AUDIO_RATE as u32)?;
    println!(
        "\nwrote {path} ({:.1} s of stereo audio)",
        audio[0].len() as f32 / AUDIO_RATE
    );

    Ok(())
}

fn cmd_capture(freq_hz: u32, seconds: f32) -> Result<(), String> {
    let (mut rtl, mut tuner, _) = open_receiver()?;
    let actual = block_on(rtl.set_sample_rate(CAPTURE_RATE))
        .map_err(|e| format!("cannot set the sample rate: {e}"))?;
    block_on(tuner.set_gain(&mut rtl, GainMode::Automatic))
        .map_err(|e| format!("cannot set gain: {e}"))?;

    let bytes = ((seconds * CAPTURE_RATE as f32) as usize * 2).next_multiple_of(1 << 18);
    let raw = capture(&mut rtl, &mut tuner, freq_hz, bytes)?;

    std::fs::create_dir_all("captures").map_err(|e| e.to_string())?;
    let path = format!("captures/{}Hz-{}sps.cs8", freq_hz, actual as u32);
    std::fs::write(&path, &raw).map_err(|e| e.to_string())?;

    println!("wrote {path}");
    println!("  {} complex samples at {actual:.0} Hz", raw.len() / 2);
    println!("  level {:.1} dBFS", power_dbfs(&raw));
    Ok(())
}

/// Writes 16-bit stereo PCM.
fn write_wav(path: &str, left: &[f32], right: &[f32], rate: u32) -> Result<(), String> {
    let frames = left.len().min(right.len());
    let data_bytes = frames * 4;

    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut header = Vec::with_capacity(44);
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&(36 + data_bytes as u32).to_le_bytes());
    header.extend_from_slice(b"WAVEfmt ");
    header.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    header.extend_from_slice(&1u16.to_le_bytes()); // PCM
    header.extend_from_slice(&2u16.to_le_bytes()); // channels
    header.extend_from_slice(&rate.to_le_bytes());
    header.extend_from_slice(&(rate * 4).to_le_bytes()); // bytes per second
    header.extend_from_slice(&4u16.to_le_bytes()); // block align
    header.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    header.extend_from_slice(b"data");
    header.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    file.write_all(&header).map_err(|e| e.to_string())?;

    let mut pcm = Vec::with_capacity(data_bytes);
    for k in 0..frames {
        for sample in [left[k], right[k]] {
            let clamped = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            pcm.extend_from_slice(&clamped.to_le_bytes());
        }
    }
    file.write_all(&pcm).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_parsing_accepts_the_usual_forms() {
        assert_eq!(parse_frequency("98500000"), Some(98_500_000));
        assert_eq!(parse_frequency("98.5M"), Some(98_500_000));
        assert_eq!(parse_frequency("7100k"), Some(7_100_000));
        assert_eq!(parse_frequency("1.2G"), Some(1_200_000_000));
        assert_eq!(parse_frequency("nonsense"), None);
    }

    #[test]
    fn power_of_a_midscale_capture_is_near_silence() {
        // 127 and 128 straddle zero, so an alternating pattern is as close to silent as
        // offset binary gets.
        let quiet: Vec<u8> = (0..2048)
            .map(|k| if k % 2 == 0 { 127 } else { 128 })
            .collect();
        assert!(power_dbfs(&quiet) < -40.0, "got {}", power_dbfs(&quiet));

        let loud = vec![255u8; 2048];
        assert!(power_dbfs(&loud) > -6.0, "got {}", power_dbfs(&loud));
    }
}
