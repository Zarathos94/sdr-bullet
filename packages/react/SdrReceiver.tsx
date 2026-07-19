/**
 * React mount for the receiver, for embedding as a showcase on a host site.
 *
 * The receiver's interface is written against the DOM directly rather than in React, so
 * this wrapper is thin by design: mount the component into a ref on mount, tear it down on
 * unmount. Keeping the UI framework-free is what lets the same code serve both the
 * standalone app and this embed without dragging a second rendering library into a page
 * that already has one.
 *
 * The receiver needs the page to be cross-origin isolated. On a Next.js host that means a
 * scoped `headers()` entry for this route and self-hosted fonts; see docs/deployment.md.
 * This component does not — and cannot — set those headers itself.
 */

'use client'

import { useEffect, useRef } from 'react'
import { ReceiverUI } from '../../web/src/ui.js'
import '../../web/src/styles.css'

export interface SdrReceiverProps {
  /** Frequency to open on, in hertz. Defaults to a strong FM broadcast frequency. */
  initialFrequencyHz?: number
  className?: string
}

export default function SdrReceiver({ className }: SdrReceiverProps) {
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const host = mountRef.current
    if (!host) return

    const ui = new ReceiverUI(host)
    // The receiver holds a USB device, an audio graph and several workers, none of which
    // survive a component unmount cleanly on their own. Tearing them down explicitly is
    // what stops a device staying claimed until the page is reloaded.
    return () => {
      void ui.destroy()
    }
  }, [])

  return <div ref={mountRef} className={className} />
}
