# sdr-bullet

A software defined radio receiver that runs entirely in the browser. WebUSB drives an
RTL-SDR dongle, the whole signal chain — decimation, FFT, and FM/AM/SSB demodulation with
RDS — runs in Rust compiled to WebAssembly with SIMD, and the waterfall and constellation
are drawn on the GPU with WebGPU.

Built and tested against an [RTL-SDR Blog V4](https://www.rtl-sdr.com/v4/).

## What makes this more than a toy

- **A real demodulator pipeline.** NFM, wideband FM with 19 kHz-pilot stereo decoding and
  50/75 µs de-emphasis, AM, SSB by Weaver's method, CW, and RDS decoded down to the station
  name and radio text. Not one mode wired up end to end, but a full chain.
- **One driver, three transports.** The RTL2832U and R828D register logic is written once,
  in Rust, generic over a transport. The same code drives real hardware over libusb from a
  terminal, drives it over WebUSB in the browser, and runs against a recording mock in the
  tests — so the register sequences are validated against silicon before anything depends on
  them inside a worker.
- **Genuine parallelism on stable Rust.** Shared-memory WebAssembly threading needs a
  nightly toolchain and `build-std`; this instead runs each pipeline stage in its own worker
  with its own WebAssembly instance, joined by lock-free SharedArrayBuffer ring buffers. That
  is the natural shape for a streaming signal chain anyway.
- **The GPU does the drawing.** The waterfall is a ring texture scrolled by a sampling
  offset rather than copied each frame; the constellation accumulates density into an atomic
  storage buffer, because WebGPU has no storage-texture atomics.

## Layout

```
crates/
  sdr-dsp     the signal chain — filters, FFT, demodulators — with no I/O, fully unit-tested
  sdr-rtl     the transport-agnostic RTL2832U + R828D driver
  sdr-wasm    the WebAssembly bindings: pipeline stages and the WebUSB-backed receiver
  sdr-probe   a native harness that drives a real dongle from the command line
web/          the browser application: workers, audio, WebGPU, UI
docs/         architecture, DSP notes, the register protocol, and Linux setup
```

## Building

The DSP and driver are ordinary Rust:

```sh
cargo test --workspace          # the whole test suite, no hardware required
cargo run -p sdr-probe -- info  # talk to an attached dongle
```

The browser build needs Node 20.19+ or 22.12+ and a pinned `wasm-bindgen`:

```sh
cargo install wasm-bindgen-cli --version 0.2.100   # must match the crate exactly
./scripts/build-wasm.sh                            # compile the WebAssembly and bindings
cd web && npm install && npm run dev
```

The dev server sets the cross-origin isolation headers the pipeline needs. Develop in a
Chromium-based browser: WebUSB is Chromium-only, and WebGPU on Linux is furthest along
there.

## Running against real hardware on Linux

Two prerequisites, both outside the app's control:

The kernel's DVB driver claims the dongle on sight, and WebUSB cannot detach a kernel
driver. Blacklist it:

```sh
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/blacklist-rtlsdr.conf
sudo modprobe -r dvb_usb_rtl28xxu   # then unplug and replug
```

And your user needs access to the device node — a udev rule with `TAG+="uaccess"` grants it
to whoever is logged in at the seat. Installing your distribution's `rtl-sdr` package
provides both the blacklist and the rule.

If the app reports *Access denied* the udev rule is missing; if it reports *could not claim
the interface* the kernel driver is still attached.

See [docs/linux-setup.md](docs/linux-setup.md) for the details, and
[docs/architecture.md](docs/architecture.md) for how the pieces fit together.

## Licence

MIT.
