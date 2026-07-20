# The RTL2832U / R828D register protocol

Notes on driving the demodulator and tuner, with emphasis on the parts that are easy to get
wrong and the two that the published references get wrong for this hardware.

## Control transfers

Every register access is a vendor control transfer with `bRequest = 0`. The target is
encoded in `wValue` and `wIndex`, and there are two different encodings.

**Flat blocks** (USB, system, I2C) put the address in `wValue` and the block number in the
high byte of `wIndex`, with `0x10` set for a write:

```
read:   wValue = addr,  wIndex = block << 8
write:  wValue = addr,  wIndex = (block << 8) | 0x10
```

**The demodulator** does not. Its address moves into the high byte of `wValue` and the page
number takes over `wIndex`:

```
read:   wValue = (addr << 8) | 0x20,  wIndex = page
write:  wValue = (addr << 8) | 0x20,  wIndex = 0x10 | page
```

Applying the flat-block layout to a demodulator register reads a plausible-looking wrong
register, which is the sort of bug that costs an afternoon. Multi-byte values go on the wire
most-significant byte first.

## The tuner bus

The tuner is reached over an I2C bus behind the demodulator. That bus is gated by a
repeater bit — demodulator page 1, register `0x01`, value `0x18` to open and `0x10` to
close — and every tuner access has to be bracketed by opening and closing it.

Two hardware behaviours here were established by driving a real RTL-SDR Blog V4, and both
contradict what several reference implementations do:

**Reads are not bit-reversed.** librtlsdr passes every byte of an R82xx read through a
nibble-reversing lookup table. On this device register 0 returns `0x69` — the value that
identifies the part — directly. Applying the reversal turns it into `0x96`, and a working
tuner then looks absent. Reads here are passed through untouched.

**Writes must be split to eight bytes.** The demodulator's I2C master stalls the endpoint on
any message longer than eight bytes, register address included. The tuner's 27-register
initialisation therefore cannot go out in one transfer; it is split into chunks with the
register address advanced for each. The stall is the only symptom, and nothing in the
interface hints that the limit exists.

## Sample rate

The rate is a ratio against the 28.8 MHz crystal, programmed into a 28-bit resampler
register:

```
ratio = (crystal * 2^22) / rate,  low two bits cleared
```

The multiply is done in floating point, not integer arithmetic — the truncation point
differs between the two, and an integer version lands on a neighbouring ratio for some
rates. The result rarely divides exactly, so the driver computes and returns the rate
actually achieved, and everything downstream uses that rather than the requested value.

## Tuning

The tuner mixes from above, so the synthesiser is programmed to the wanted frequency plus
the intermediate frequency (3.57 MHz by default). The demodulator's spectrum-inversion bit
is set to undo the flip that high-side mixing introduces.

The synthesiser divider is chosen so the oscillator lands inside its 1770–3540 MHz window,
then nudged by one step according to a fine-tune reading from the tuner. The fractional part
is exact 64-bit arithmetic rather than the reference's iterative halving, which reaches the
same answer without accumulating rounding.

## The V4 front end

The Blog V4 differs from a conventional R828D board in three ways that all have to be handled
together:

- **Its tuner runs from the shared 28.8 MHz reference,** not a separate 16 MHz crystal.
  Assume 16 MHz and every synthesiser calculation is off by a factor of 1.8 — the single
  most common way to end up tuned to the wrong frequency. This cannot be read off the device;
  it is inferred from the USB descriptor strings (`RTLSDRBlog` / `Blog V4`).
- **One antenna connector is triplexed into the tuner's three inputs.** HF below 28.8 MHz
  goes through the second cable input, VHF up to 250 MHz through the first, UHF above that
  through the antenna input. The switching is rewritten only on a band change.
- **Everything below 28.8 MHz arrives through a built-in upconverter,** which shifts it up by
  the crystal frequency. The tuner cannot reach HF directly — its oscillator has no solution
  below about 27.7 MHz — so this is not an optimisation but the only way to receive it.
  Switchable notch filters attenuate the broadcast AM, FM and DAB bands, and are bypassed
  when the receiver is tuned into one of them.

There is a subtle off-by-one in the reference: upconversion uses a strict comparison against
28.8 MHz while band selection uses a non-strict one, so at exactly the crossover the HF path
is taken without upconverting and the tuner looks 28.8 MHz away from the signal. This driver
uses the same comparison on both sides.
