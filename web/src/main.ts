/**
 * Standalone entry point.
 *
 * The same `ReceiverUI` is what the site embed mounts, so this file is only the shell:
 * find the mount point, attach the styles, and hand over.
 */

import './styles.css'
import { ReceiverUI } from './ui.js'

const root = document.getElementById('app')
if (!root) {
  throw new Error('missing #app mount point')
}

new ReceiverUI(root)
