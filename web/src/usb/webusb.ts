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

