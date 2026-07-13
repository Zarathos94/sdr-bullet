/**
 * WebGPU device acquisition, kept in one place so every renderer shares one device and one
 * failure story.
 *
 * The guiding decision is that a missing GPU is not an error. WebGPU is absent on older
 * browsers and behind a flag on a few current ones, and when it is absent `navigator.gpu`
 * is simply undefined or `requestAdapter()` returns null. The receiver's audio path does not
 * touch the GPU at all, so the right response is to return null and let the caller carry on
 * with the displays dark — not to throw and take the whole app down with the pictures.
 *
 * Note there is deliberately no `forceFallbackAdapter` path. No shipping browser ships a
 * software fallback adapter, so requesting one does not rescue the null case; it just returns
 * null again a second time. The honest signal is the first null.
 */

/** Adapter identity for a diagnostics panel. Every field is the empty string when the driver withholds it. */
export interface AdapterDescription {
  readonly vendor: string
  readonly architecture: string
  readonly device: string
  readonly description: string
  /** True only for a software adapter, which as noted above no shipping browser provides. */
  readonly fallback: boolean
}

/** Optional hooks for {@link acquireDevice}. */
export interface AcquireOptions {
  /**
   * Called if the device is lost after it was handed back — a GPU reset, a driver update, or
   * the tab being evicted from the GPU process. Not called for an intentional `destroy()`,
   * which resolves the same promise but is not a fault worth surfacing.
   */
  onLost?: (info: GPUDeviceLostInfo) => void
}

/**
 * Remembers which adapter produced each device. `GPUDevice` does not reference its adapter,
 * yet the diagnostics panel wants adapter details, so the link is kept here rather than
 * forcing callers to thread the adapter through by hand. Weak so a disposed device is
 * collectable.
 */
const adapters = new WeakMap<GPUDevice, GPUAdapter>()

/** The adapter a device came from, if it was created here. */
export function adapterOf(device: GPUDevice): GPUAdapter | undefined {
  return adapters.get(device)
}

/** Reads the stable identity fields off an adapter for display. */
export function describeAdapter(adapter: GPUAdapter): AdapterDescription {
  const info = adapter.info
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    fallback: info.isFallbackAdapter,
  }
}

/**
 * Acquires a device, or null when WebGPU is unavailable.
 *
 * `requestDevice()` is the subtle step: by spec it does not reject when creation fails, it
 * returns a device that is already lost. So the lost promise is inspected before the device
 * is trusted, and a device lost at birth is reported as unavailable rather than handed back
 * as a corpse the renderers would fail against one call later.
 */
export async function acquireDevice(options?: AcquireOptions): Promise<GPUDevice | null> {
  const gpu: GPU | undefined = typeof navigator === 'undefined' ? undefined : navigator.gpu
  if (!gpu) return null

  // High performance because a waterfall wants the discrete GPU on a laptop that has one;
  // the power cost is irrelevant next to a receiver that is already driving a USB radio.
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) return null

  let device: GPUDevice
  try {
    // No required features or limits: everything the renderers use — r32float storage
    // textures, storage-buffer atomics, compute — is core, so asking for nothing keeps the
    // widest device compatibility.
    device = await adapter.requestDevice()
  } catch {
    return null
  }

  // Race the lost promise against a resolved sentinel: if the device died during creation
  // the lost promise is already settled and this observes it synchronously-enough to reject
  // the device before returning.
  const bornLost = await Promise.race([
    device.lost.then((info) => info),
    Promise.resolve(null),
  ])
  if (bornLost && bornLost.reason !== 'destroyed') return null

  adapters.set(device, adapter)

  // Surface a later, genuine loss. `destroyed` is the caller's own dispose and is not a fault.
  void device.lost.then((info) => {
    if (info.reason === 'destroyed') return
    if (options?.onLost) options.onLost(info)
    else console.warn(`WebGPU device lost: ${info.message || info.reason}`)
  })

  return device
}
