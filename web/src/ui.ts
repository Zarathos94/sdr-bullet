/**
 * The receiver's user interface, built as one self-contained component.
 *
 * Written against the DOM directly rather than through a framework so it can be dropped
 * into any host — the standalone page and the React wrapper both mount the same class. The
 * host site it targets is deliberately square-cornered and token-themed, and building
 * against the DOM keeps this in step with that without pulling a rendering library into a
 * page that already has one.
 */

import { Pipeline, type PipelineStatus } from './pipeline.js'
import { startRendering, type RenderHandle } from './render.js'
import {
  DEFAULT_DEEMPHASIS_US,
  MAX_FREQUENCY_HZ,
  MIN_FREQUENCY_HZ,
  PRESETS,
  defaultsFor,
  formatFrequency,
  parseFrequency,
} from './bandplan.js'
import type { DemodModeName } from './workers/protocol.js'

const MODES: { name: DemodModeName; label: string }[] = [
  { name: 'wfm', label: 'WFM' },
  { name: 'nfm', label: 'NFM' },
  { name: 'am', label: 'AM' },
  { name: 'usb', label: 'USB' },
  { name: 'lsb', label: 'LSB' },
  { name: 'cw', label: 'CW' },
]

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

export class ReceiverUI {
  private readonly pipeline = new Pipeline()
  private render: RenderHandle | undefined

  private frequencyHz = 98_000_000
  private mode: DemodModeName = 'wfm'
  private running = false

  // Rebuilt on start, so switching mode restarts the pipeline rather than mutating a chain
  // whose filters and rates are all decided at construction.
  private canvases: {
    spectrum: HTMLCanvasElement
    waterfall: HTMLCanvasElement
    constellation: HTMLCanvasElement
  }

  private readonly frequencyDisplay = el('div', 'frequency')
  private readonly rdsName = el('div', 'rds-name')
  private readonly rdsText = el('div', 'rds-text')
  private readonly stats = el('div', 'stat-grid')
  private readonly diagnostics = el('div', 'diagnostics')
  private readonly startButton = el('button', 'primary', 'Connect receiver')
  private readonly noticeArea = el('div')
  private readonly modeButtons = new Map<DemodModeName, HTMLButtonElement>()

  constructor(private readonly root: HTMLElement) {
    this.canvases = {
      spectrum: el('canvas'),
      waterfall: el('canvas'),
      constellation: el('canvas'),
    }
    this.build()
    this.pipeline.onStatus((status) => this.onStatus(status))
    this.pipeline.onError((message, fatal) => this.onError(message, fatal))
  }

  private build() {
    this.root.replaceChildren()

    const header = el('div', 'header')
    const title = el('div')
    title.append(el('h1', undefined, 'sdr-bullet'))
    title.append(
      el(
        'div',
        'tagline',
        'WebUSB to the dongle · Rust and WebAssembly for the DSP · the waterfall on the GPU',
      ),
    )
    header.append(title, this.diagnostics)
    this.root.append(header)

    // Capability gate. Without cross-origin isolation or WebUSB the pipeline cannot run,
    // and saying exactly why beats a blank page or an opaque failure deep in a worker.
    const capability = Pipeline.capabilities()
    if (!capability.ok) {
      const notice = el('div', 'notice error')
      notice.append(el('strong', undefined, 'This browser cannot run the receiver.'))
      notice.append(el('pre', undefined, capability.reason ?? ''))
      this.root.append(notice)
      return
    }

    this.root.append(this.noticeArea)
    this.root.append(this.buildDisplays())
    this.root.append(this.buildControls())

    this.updateFrequencyDisplay()
    this.startButton.addEventListener('click', () => void this.toggle())
  }

  private buildDisplays(): HTMLElement {
    const displays = el('div', 'displays')

    const stack = el('div', 'display-stack')
    for (const [key, label] of [
      ['spectrum', 'Spectrum'],
      ['waterfall', 'Waterfall'],
    ] as const) {
      const wrap = el('div', `canvas-wrap ${key}`)
      wrap.append(this.canvases[key])
      wrap.append(el('div', 'canvas-label', label))
      stack.append(wrap)
    }
    displays.append(stack)

    const constellationWrap = el('div', 'canvas-wrap constellation-wrap')
    constellationWrap.append(this.canvases.constellation)
    constellationWrap.append(el('div', 'canvas-label', 'Constellation'))
    displays.append(constellationWrap)

    return displays
  }

  private buildControls(): HTMLElement {
    const side = el('div', 'side')

    // Frequency and tuning.
    const tuning = el('div', 'panel')
    tuning.append(el('h2', undefined, 'Tuning'))
    tuning.append(this.frequencyDisplay)

    const input = el('input')
    input.type = 'text'
    input.placeholder = 'e.g. 98.5M, 7100k, 145000000'
    input.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return
      const hz = parseFrequency(input.value)
      if (hz !== null) this.setFrequency(hz)
      input.value = ''
    })
    tuning.append(input)

    const presetSelect = el('select')
    presetSelect.append(el('option', undefined, 'Presets…'))
    for (const [index, preset] of PRESETS.entries()) {
      const option = el('option', undefined, preset.label)
      option.value = String(index)
      presetSelect.append(option)
    }
    presetSelect.addEventListener('change', () => {
      const preset = PRESETS[Number(presetSelect.value)]
      if (!preset) return
      this.mode = preset.mode
      this.setFrequency(preset.frequencyHz)
      this.syncModeButtons()
      if (preset.deemphasisUs) this.pipeline.setDeemphasis(preset.deemphasisUs)
    })
    tuning.append(presetSelect)
    tuning.append(this.startButton)
    side.append(tuning)

    // Mode.
    const modePanel = el('div', 'panel')
    modePanel.append(el('h2', undefined, 'Mode'))
    const modeGroup = el('div', 'mode-group')
    for (const { name, label } of MODES) {
      const button = el('button', undefined, label)
      button.addEventListener('click', () => void this.setMode(name))
      this.modeButtons.set(name, button)
      modeGroup.append(button)
    }
    modePanel.append(modeGroup)
    this.syncModeButtons()
    side.append(modePanel)

    side.append(this.buildAudioPanel())
    side.append(this.buildRdsPanel())

    const statusPanel = el('div', 'panel')
    statusPanel.append(el('h2', undefined, 'Signal'))
    statusPanel.append(this.stats)
    side.append(statusPanel)

    return side
  }

  private buildAudioPanel(): HTMLElement {
    const panel = el('div', 'panel')
    panel.append(el('h2', undefined, 'Audio'))

    const volumeRow = el('div', 'control-row')
    volumeRow.append(el('label', undefined, 'Volume'))
    const volume = el('input')
    volume.type = 'range'
    volume.min = '0'
    volume.max = '100'
    volume.value = '70'
    volume.addEventListener('input', () => {
      this.pipeline.setVolume(Number(volume.value) / 100)
    })
    volumeRow.append(volume)
    panel.append(volumeRow)

    const gainRow = el('div', 'control-row')
    gainRow.append(el('label', undefined, 'Tuner gain'))
    const gain = el('select')
    const autoOption = el('option', undefined, 'Automatic')
    autoOption.value = '-1'
    gain.append(autoOption)
    // Whole-decibel steps are plenty for a control; the underlying table is finer.
    for (let db = 0; db <= 49; db += 3) {
      const option = el('option', undefined, `${db} dB`)
      option.value = String(db * 10)
      gain.append(option)
    }
    gain.addEventListener('change', () => this.pipeline.setGain(Number(gain.value)))
    gainRow.append(gain)
    panel.append(gainRow)

    const squelchRow = el('div', 'control-row')
    squelchRow.append(el('label', undefined, 'Squelch'))
    const squelch = el('input')
    squelch.type = 'checkbox'
    squelch.addEventListener('change', () => {
      this.pipeline.setSquelch(squelch.checked, 0.08)
    })
    squelchRow.append(squelch)
    panel.append(squelchRow)

    return panel
  }

  private buildRdsPanel(): HTMLElement {
    const panel = el('div', 'panel rds-panel')
    panel.append(el('h2', undefined, 'Radio data'))
    panel.append(this.rdsName)
    panel.append(this.rdsText)
    return panel
  }

  private syncModeButtons() {
    for (const [name, button] of this.modeButtons) {
      button.classList.toggle('active', name === this.mode)
    }
  }

  private updateFrequencyDisplay() {
    const { value, unit } = formatFrequency(this.frequencyHz)
    this.frequencyDisplay.replaceChildren(
      document.createTextNode(value),
      el('span', 'unit', unit),
    )
  }

