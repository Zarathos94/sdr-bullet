/**
 * WebUSB transport for the RTL2832U.
 *
 * The register sequences live in Rust and are shared with the native harness; this file
 * only moves bytes. That split is deliberate — the sequences are the part that is hard to
 * get right, and debugging them through a browser sandbox is considerably worse than
 * debugging them from a terminal.
 *
 * Two platform facts shape the design:
 *
 * `requestDevice()` is exposed only to `Window`, so the chooser has to be opened from the
 * page. Everything else — `getDevices()`, and every transfer — is available to a dedicated
 * worker, so the worker reacquires the already-permitted device rather than being handed
 * one, because a `USBDevice` cannot be structured-cloned across the boundary.
 *
 * WebUSB cannot detach a kernel driver, and Chromium's automatic detach is both disabled
 * on Linux and restricted to an allowlist that does not include the DVB driver. There is
 * no workaround in code; see the diagnostics in `describeClaimFailure`.
 */

const VENDOR_ID = 0x0bda
const PRODUCT_ID = 0x2838

/** Descriptor strings that identify the Blog V4. */
const V4_MANUFACTURER = 'RTLSDRBlog'
const V4_PRODUCT = 'Blog V4'

/** Interface carrying the sample endpoint. */
const INTERFACE = 0
/** Bulk IN endpoint number. */
const ENDPOINT = 1

/**
 * Transfers kept in flight.
 *
 * WebUSB follows a host-driven request/response model: with no transfer outstanding the
 * device's data is simply dropped, so a single awaited transfer loses everything that
 * arrives between one completing and the next being submitted. At 2.4 million samples a
 * second that gap is not small. Keeping several queued means there is always somewhere for
 * the data to go.
 */
const TRANSFERS_IN_FLIGHT = 6
const TRANSFER_BYTES = 128 * 1024

export interface DeviceIdentity {
  manufacturer: string | undefined
  product: string | undefined
  serial: string | undefined
  /**
   * Whether this is a Blog V4, which is only discoverable from the descriptor strings.
   *
   * It matters more than it looks: the V4 clocks its tuner from the shared 28.8 MHz
   * reference where a conventional board uses a separate 16 MHz crystal, and every
   * synthesiser calculation depends on which.
   */
  isV4: boolean
}

function identify(device: USBDevice): DeviceIdentity {
  return {
    // The descriptor accessors are nullable; normalise to undefined so the identity type
    // stays optional rather than dragging null through every consumer.
    manufacturer: device.manufacturerName ?? undefined,
    product: device.productName ?? undefined,
    serial: device.serialNumber ?? undefined,
    isV4:
      device.manufacturerName === V4_MANUFACTURER && device.productName === V4_PRODUCT,
  }
}

/** Whether this browser can talk to USB devices at all. */
export function isSupported(): boolean {
  return typeof navigator !== 'undefined' && 'usb' in navigator
}

/**
 * Opens the device chooser. Must be called from the page, inside a user gesture.
 *
 * Returns the identity of whatever was granted; the worker picks the device up afterwards
 * via `getDevices()`.
 */
export async function requestDevice(): Promise<DeviceIdentity> {
  if (!isSupported()) {
    throw new Error(
      'This browser has no WebUSB. Chromium-based browsers support it; Firefox and Safari ' +
        'have both declined to implement it.',
    )
  }
  const device = await navigator.usb.requestDevice({
    filters: [{ vendorId: VENDOR_ID, productId: PRODUCT_ID }],
  })
  return identify(device)
}

/** Devices already permitted, which is what a worker can reach. */
export async function grantedDevices(): Promise<USBDevice[]> {
  if (!isSupported()) return []
  const all = await navigator.usb.getDevices()
  return all.filter((d) => d.vendorId === VENDOR_ID && d.productId === PRODUCT_ID)
}

/**
 * Turns a claim failure into something actionable.
 *
 * The two failure modes are cleanly separable and have completely different fixes, but
 * both surface as an opaque `DOMException` — so telling them apart here saves the far
 * longer diagnosis of why a device that "connects fine" produces silence.
 */
export function describeClaimFailure(error: unknown): string {
  const name = error instanceof DOMException ? error.name : ''
  if (name === 'SecurityError') {
    return (
      'Access denied opening the device. The operating system is not letting this browser ' +
      'reach it — on Linux that means a udev rule granting your user access is missing. ' +
      'Installing the rtl-sdr package provides one.'
    )
  }
  if (name === 'NetworkError') {
    return (
      'Could not claim the interface, which means a kernel driver already holds it. ' +
      'WebUSB cannot detach kernel drivers and Chromium will not do it automatically for ' +
      'this device. Blacklist dvb_usb_rtl28xxu, then unplug and replug:\n\n' +
      "  echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/blacklist-rtlsdr.conf\n" +
      '  sudo modprobe -r dvb_usb_rtl28xxu'
    )
  }
  return error instanceof Error ? error.message : String(error)
}

/**
 * An open device, ready for the driver to drive.
 *
 * Shaped to match the `Transport` trait on the Rust side so the two stay in step.
 */
export class UsbTransport {
  private streaming = false
  private queue: Promise<USBInTransferResult>[] = []

  private constructor(
    private readonly device: USBDevice,
    readonly identity: DeviceIdentity,
  ) {}

  /** Opens the first already-permitted device. Safe to call from a worker. */
  static async open(): Promise<UsbTransport> {
    const devices = await grantedDevices()
    const device = devices[0]
    if (!device) {
      throw new Error('no permitted device found — request access from the page first')
    }

    await device.open()
    // Most devices are already in configuration 1, but selecting it is cheap and makes the
    // starting state explicit rather than inherited.
    if (device.configuration === null) {
      await device.selectConfiguration(1)
    }

    try {
      await device.claimInterface(INTERFACE)
    } catch (error) {
      await device.close().catch(() => {})
      throw new Error(describeClaimFailure(error))
    }

    return new UsbTransport(device, identify(device))
  }

  /** Vendor control transfer, host to device. */
  async controlOut(value: number, index: number, data: Uint8Array): Promise<void> {
    // Copy into a fresh buffer: the source is usually a view into WebAssembly memory,
    // which the transfer must not alias while it is in flight.
    const payload = new Uint8Array(data.length)
    payload.set(data)

    const result = await this.device.controlTransferOut(
      {
        requestType: 'vendor',
        recipient: 'device',
        request: 0,
        value,
        index,
      },
      payload,
    )
    if (result.status !== 'ok') {
      throw new Error(`control write to 0x${value.toString(16)} failed: ${result.status}`)
    }
  }

  /** Vendor control transfer, device to host. */
  async controlIn(value: number, index: number, length: number): Promise<Uint8Array> {
    const result = await this.device.controlTransferIn(
      {
        requestType: 'vendor',
        recipient: 'device',
        request: 0,
        value,
        index,
      },
      length,
    )
    if (result.status !== 'ok' || !result.data) {
      throw new Error(`control read from 0x${value.toString(16)} failed: ${result.status}`)
    }
    return new Uint8Array(result.data.buffer, result.data.byteOffset, result.data.byteLength)
  }

  /** Begins the sample stream, filling the transfer queue. */
  startStream(): void {
    if (this.streaming) return
    this.streaming = true
    this.queue = []
    for (let k = 0; k < TRANSFERS_IN_FLIGHT; k++) {
      this.queue.push(this.device.transferIn(ENDPOINT, TRANSFER_BYTES))
    }
  }

  /**
   * Takes the next completed transfer and immediately submits a replacement.
   *
   * Resubmitting before returning is what keeps the queue depth constant; doing it after
   * the caller has processed the block would leave a gap for exactly as long as processing
   * takes.
   */
  async readSamples(): Promise<Uint8Array> {
    if (!this.streaming) throw new Error('stream not started')

    const pending = this.queue.shift()
    if (!pending) throw new Error('transfer queue is empty')
    this.queue.push(this.device.transferIn(ENDPOINT, TRANSFER_BYTES))

    const result = await pending
    if (result.status === 'stall') {
      // Clearing the halt recovers the endpoint, but the data for this transfer is gone.
      // Reporting that honestly matters: returning silent zeroes here — which is what the
      // established JavaScript implementations do — turns a recoverable fault into an
      // unexplained gap in the audio.
      await this.device.clearHalt('in', ENDPOINT)
      throw new Error('sample endpoint stalled')
    }
    if (result.status !== 'ok' || !result.data) {
      throw new Error(`sample transfer failed: ${result.status}`)
    }
    return new Uint8Array(result.data.buffer, result.data.byteOffset, result.data.byteLength)
  }

  async stopStream(): Promise<void> {
    this.streaming = false
    // Let the outstanding transfers settle rather than tearing the endpoint down beneath
    // them, which surfaces as spurious errors on the next open.
    await Promise.allSettled(this.queue)
    this.queue = []
  }

  async close(): Promise<void> {
    await this.stopStream()
    try {
      await this.device.releaseInterface(INTERFACE)
    } catch {
      // Already released, or the device is gone. Nothing useful to do either way.
    }
    await this.device.close().catch(() => {})
  }
}
