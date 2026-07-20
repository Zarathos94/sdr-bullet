/**
 * The receiver's user interface, built as one self-contained component.
 *
 * Written against the DOM directly rather than through a framework so it can be dropped
 * into any host — the standalone page and the React wrapper both mount the same class. The
 * host site it targets is deliberately square-cornered and token-themed, and building
 * against the DOM keeps this in step with that without pulling a rendering library into a
 * page that already has one.
 *
 * Two views over the same pipeline: a **radio** mode that behaves like a consumer AM/FM set
 * — tune, seek, presets, scan the band into a station list — and a **spectrum** mode with
 * the demodulator controls and the raw displays. Before a device is connected, a synthetic
 * demo animation runs so the displays are alive and it is clear the app initialised.
 */

import { DemoDisplay } from './demo.js'
import { Pipeline, type PipelineStatus } from './pipeline.js'
import { startRendering, type RenderHandle } from './render.js'
import { Scanner, SCAN_BANDS, type FoundStation, type ScanBand } from './scan.js'
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

type ViewMode = 'radio' | 'spectrum'
type BandName = 'fm' | 'am'

const MODES: { name: DemodModeName; label: string }[] = [
  { name: 'wfm', label: 'WFM' },
  { name: 'nfm', label: 'NFM' },
  { name: 'am', label: 'AM' },
  { name: 'usb', label: 'USB' },
  { name: 'lsb', label: 'LSB' },
  { name: 'cw', label: 'CW' },
]

const PRESET_SLOTS = 6
const PRESET_KEY = 'sdr-bullet.presets.v1'

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

// Inline SVG icons. The app runs cross-origin-isolated (COEP: require-corp) inside the
// site's iframe, which blocks the Google Fonts icon stylesheet — so icons are drawn as
// self-contained SVG with no network dependency, rather than as an icon font whose glyphs
// would otherwise fall back to their raw ligature text ("chevron_left", "radar", …).
const ICON_GLYPHS: Record<string, string> = {
  chevron_left: '<polyline points="15 5 8 12 15 19"/>',
  chevron_right: '<polyline points="9 5 16 12 9 19"/>',
  keyboard_double_arrow_left: '<polyline points="18 6 12 12 18 18"/><polyline points="11 6 5 12 11 18"/>',
  keyboard_double_arrow_right: '<polyline points="6 6 12 12 6 18"/><polyline points="13 6 19 12 13 18"/>',
  radar:
    '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.2"/>' +
    '<line x1="12" y1="12" x2="19" y2="8"/>' +
    '<circle cx="12" cy="12" r="1.15" fill="currentColor" stroke="none"/>',
  stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="1" fill="currentColor" stroke="none"/>',
}

function icon(name: string): HTMLElement {
  const span = el('span', 'material-symbols-outlined')
  span.setAttribute('aria-hidden', 'true')
  const glyph = ICON_GLYPHS[name] ?? ''
  span.innerHTML =
    '<svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    glyph +
    '</svg>'
  return span
}

interface PresetSlot {
  frequencyHz: number
  band: BandName
  label: string
}

export class ReceiverUI {
  private readonly pipeline = new Pipeline()
  private readonly scanner = new Scanner(this.pipeline)
  private render: RenderHandle | undefined
  private demo: DemoDisplay | undefined
  private demoResizeObserver: ResizeObserver | undefined

  private frequencyHz = 98_000_000
  private mode: DemodModeName = 'wfm'
  private band: BandName = 'fm'
  private view: ViewMode = 'radio'
  private running = false
  private stations: FoundStation[] = []
  private presets: (PresetSlot | null)[] = new Array(PRESET_SLOTS).fill(null)

  private readonly canvases: {
    spectrum: HTMLCanvasElement
    waterfall: HTMLCanvasElement
    constellation: HTMLCanvasElement
  }
  private readonly demoCanvas = el('canvas', 'demo-canvas')

  // Shared DOM references.
  private readonly diagnostics = el('div', 'diagnostics')
  private readonly connectButton = el('button', 'primary', 'Connect receiver')
  private readonly noticeArea = el('div')
  private readonly radioView = el('div', 'view radio-view')
  private readonly spectrumView = el('div', 'view spectrum-view')
  private readonly viewToggle = el('div', 'view-toggle')
  private readonly gpuCanvasWrap = el('div', 'gpu-displays')

  // Radio view references.
  private readonly radioFreq = el('div', 'radio-freq')
  private readonly radioBandLabel = el('span', 'radio-band')
  private readonly radioStereo = el('span', 'badge off', 'MONO')
  private readonly radioName = el('div', 'radio-name')
  private readonly radioText = el('div', 'radio-text')
  private readonly signalBar = el('div', 'meter-fill')
  private readonly stationList = el('div', 'station-list')
  private readonly scanButton = el('button', undefined)
  private readonly bandTabs = new Map<BandName, HTMLButtonElement>()
  private readonly presetButtons: HTMLButtonElement[] = []

  // Spectrum view references.
  private readonly frequencyDisplay = el('div', 'frequency')
  private readonly rdsName = el('div', 'rds-name')
  private readonly rdsText = el('div', 'rds-text')
  private readonly stats = el('div', 'stat-grid')
  private readonly modeButtons = new Map<DemodModeName, HTMLButtonElement>()

  constructor(private readonly root: HTMLElement) {
    this.canvases = {
      spectrum: el('canvas'),
      waterfall: el('canvas'),
      constellation: el('canvas'),
    }
    this.loadPresets()
    this.build()
    this.pipeline.onStatus((status) => this.onStatus(status))
    this.pipeline.onError((message, fatal) => this.onError(message, fatal))
  }

  // -- Layout -------------------------------------------------------------

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

    const capability = Pipeline.capabilities()
    if (!capability.ok) {
      const notice = el('div', 'notice error')
      notice.append(el('strong', undefined, 'This browser cannot run the receiver.'))
      notice.append(el('pre', undefined, capability.reason ?? ''))
      this.root.append(notice)
      return
    }

    this.root.append(this.buildViewToggle())
    this.root.append(this.noticeArea)
    this.root.append(this.buildDisplays())
    this.buildRadioView()
    this.buildSpectrumView()
    this.root.append(this.radioView)
    this.root.append(this.spectrumView)

    this.setView('radio')
    this.updateRadioDisplay()
    this.updateFrequencyDisplay()
    this.startDemo()

    this.connectButton.addEventListener('click', () => void this.toggle())
  }

  private buildViewToggle(): HTMLElement {
    for (const [mode, label] of [
      ['radio', 'Radio'],
      ['spectrum', 'Spectrum'],
    ] as const) {
      const button = el('button', undefined, label)
      button.addEventListener('click', () => this.setView(mode))
      button.dataset['view'] = mode
      this.viewToggle.append(button)
    }
    return this.viewToggle
  }

  private buildDisplays(): HTMLElement {
    const displays = el('div', 'displays')

    // The WebGPU displays, hidden behind the demo until a device is connected.
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
    this.gpuCanvasWrap.append(stack)

    const constellationWrap = el('div', 'canvas-wrap constellation-wrap')
    constellationWrap.append(this.canvases.constellation)
    constellationWrap.append(el('div', 'canvas-label', 'Constellation'))
    this.gpuCanvasWrap.append(constellationWrap)

    // The demo overlay sits on top and covers the displays until we go live.
    const demoWrap = el('div', 'canvas-wrap demo-wrap')
    demoWrap.append(this.demoCanvas)
    demoWrap.append(el('div', 'canvas-label', 'Demo signal — connect a receiver for live data'))

    displays.append(this.gpuCanvasWrap)
    displays.append(demoWrap)
    return displays
  }

  // -- Radio view ---------------------------------------------------------

  private buildRadioView() {
    const dial = el('div', 'panel radio-dial')

    const top = el('div', 'radio-top')
    const bandRow = el('div', 'radio-bandrow')
    for (const name of ['fm', 'am'] as const) {
      const button = el('button', undefined, SCAN_BANDS[name].label)
      button.addEventListener('click', () => void this.setBand(name))
      this.bandTabs.set(name, button)
      bandRow.append(button)
    }
    top.append(bandRow)
    top.append(this.radioStereo)
    dial.append(top)

    // The frequency readout doubles as a tuning surface: scroll to step channels.
    this.radioFreq.tabIndex = 0
    this.radioFreq.addEventListener('wheel', (event) => {
      event.preventDefault()
      this.tuneStep(event.deltaY < 0 ? 1 : -1)
    }, { passive: false })
    this.radioFreq.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowUp' || event.key === 'ArrowRight') this.tuneStep(1)
      if (event.key === 'ArrowDown' || event.key === 'ArrowLeft') this.tuneStep(-1)
    })
    const freqWrap = el('div', 'radio-freq-wrap')
    freqWrap.append(this.radioFreq, this.radioBandLabel)
    dial.append(freqWrap)

    // Seek / tune controls.
    const transport = el('div', 'radio-transport')
    const mk = (title: string, glyph: string, onClick: () => void) => {
      const b = el('button', 'radio-btn')
      b.title = title
      b.setAttribute('aria-label', title)
      b.append(icon(glyph))
      b.addEventListener('click', onClick)
      return b
    }
    transport.append(
      mk('Seek down', 'keyboard_double_arrow_left', () => void this.seek(-1)),
      mk('Tune down', 'chevron_left', () => this.tuneStep(-1)),
      mk('Tune up', 'chevron_right', () => this.tuneStep(1)),
      mk('Seek up', 'keyboard_double_arrow_right', () => void this.seek(1)),
    )
    dial.append(transport)

    // Signal strength meter + station name / radio text.
    const meter = el('div', 'meter')
    meter.append(this.signalBar)
    dial.append(meter)
    const rds = el('div', 'radio-rds')
    this.radioName.textContent = '—'
    rds.append(this.radioName, this.radioText)
    dial.append(rds)

    const volumeRow = el('div', 'control-row')
    volumeRow.append(el('label', undefined, 'Volume'))
    const volume = el('input')
    volume.type = 'range'
    volume.min = '0'
    volume.max = '100'
    volume.value = '70'
    volume.addEventListener('input', () => this.pipeline.setVolume(Number(volume.value) / 100))
    volumeRow.append(volume)
    dial.append(volumeRow)

    dial.append(this.connectButton)
    this.radioView.append(dial)

    // Presets.
    const presetPanel = el('div', 'panel')
    presetPanel.append(el('h2', undefined, 'Presets'))
    const presetGrid = el('div', 'preset-grid')
    for (let slot = 0; slot < PRESET_SLOTS; slot++) {
      const button = el('button', 'preset-slot')
      button.addEventListener('click', () => this.recallPreset(slot))
      button.addEventListener('contextmenu', (event) => {
        event.preventDefault()
        this.savePreset(slot)
      })
      this.presetButtons.push(button)
      presetGrid.append(button)
    }
    presetPanel.append(presetGrid)
    presetPanel.append(
      el('p', 'hint', 'Click to recall · right-click to save the current station'),
    )
    this.radioView.append(presetPanel)

    // Station catalogue + scan.
    const scanPanel = el('div', 'panel')
    const scanHeader = el('div', 'panel-header')
    scanHeader.append(el('h2', undefined, 'Stations'))
    this.scanButton.append(icon('radar'), document.createTextNode('Scan band'))
    this.scanButton.addEventListener('click', () => void this.scan())
    scanHeader.append(this.scanButton)
    scanPanel.append(scanHeader)
    scanPanel.append(this.stationList)
    this.radioView.append(scanPanel)

    this.renderStations()
    this.syncBandTabs()
    this.renderPresets()
  }

  // -- Spectrum (advanced) view ------------------------------------------

  private buildSpectrumView() {
    const side = el('div', 'side')

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
    presetSelect.append(el('option', undefined, 'Band presets…'))
    for (const [index, preset] of PRESETS.entries()) {
      const option = el('option', undefined, preset.label)
      option.value = String(index)
      presetSelect.append(option)
    }
    presetSelect.addEventListener('change', () => {
      const preset = PRESETS[Number(presetSelect.value)]
      if (!preset) return
      void this.applyModeAndTune(preset.mode, preset.frequencyHz, preset.deemphasisUs)
    })
    tuning.append(presetSelect)
    side.append(tuning)

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

    const rdsPanel = el('div', 'panel rds-panel')
    rdsPanel.append(el('h2', undefined, 'Radio data'))
    rdsPanel.append(this.rdsName, this.rdsText)
    side.append(rdsPanel)

    const statusPanel = el('div', 'panel')
    statusPanel.append(el('h2', undefined, 'Signal'))
    statusPanel.append(this.stats)
    side.append(statusPanel)

    this.spectrumView.append(side)
  }

  private buildAudioPanel(): HTMLElement {
    const panel = el('div', 'panel')
    panel.append(el('h2', undefined, 'Audio'))

    const gainRow = el('div', 'control-row')
    gainRow.append(el('label', undefined, 'Tuner gain'))
    const gain = el('select')
    const autoOption = el('option', undefined, 'Automatic')
    autoOption.value = '-1'
    gain.append(autoOption)
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
    squelch.addEventListener('change', () => this.pipeline.setSquelch(squelch.checked, 0.08))
    squelchRow.append(squelch)
    panel.append(squelchRow)

    return panel
  }

  private setView(view: ViewMode) {
    this.view = view
    this.radioView.classList.toggle('hidden', view !== 'radio')
    this.spectrumView.classList.toggle('hidden', view !== 'spectrum')
    for (const button of this.viewToggle.querySelectorAll('button')) {
      button.classList.toggle('active', button.dataset['view'] === view)
    }
  }

  // -- Demo ---------------------------------------------------------------

  private startDemo() {
    if (this.running) return
    try {
      this.demo = new DemoDisplay(this.demoCanvas)
      this.demo.start()
      this.demoResizeObserver = new ResizeObserver(() => this.demo?.resize())
      this.demoResizeObserver.observe(this.demoCanvas)
    } catch {
      // A missing 2D context is not worth failing the whole app over.
    }
    document.querySelector('.demo-wrap')?.classList.remove('hidden')
    this.gpuCanvasWrap.classList.add('hidden')
  }

  private stopDemo() {
    this.demo?.stop()
    this.demo = undefined
    this.demoResizeObserver?.disconnect()
    this.demoResizeObserver = undefined
    document.querySelector('.demo-wrap')?.classList.add('hidden')
    this.gpuCanvasWrap.classList.remove('hidden')
  }

  // -- Tuning -------------------------------------------------------------

  private setFrequency(hz: number) {
    this.frequencyHz = Math.max(MIN_FREQUENCY_HZ, Math.min(MAX_FREQUENCY_HZ, hz))
    this.updateFrequencyDisplay()
    this.updateRadioDisplay()
    if (this.running) this.pipeline.tune(this.frequencyHz)
  }

  private tuneStep(direction: 1 | -1) {
    const band = SCAN_BANDS[this.band]
    let next = this.frequencyHz + direction * band.channelSpacingHz
    // Wrap within the band so the dial never dead-ends.
    if (next > band.endHz) next = band.startHz
    if (next < band.startHz) next = band.endHz
    this.setFrequency(next)
  }

  private async seek(direction: 1 | -1) {
    if (!this.running) {
      this.tuneStep(direction)
      return
    }
    this.setTransportBusy(true)
    try {
      const found = await this.scanner.seek(direction, SCAN_BANDS[this.band])
      if (found !== null) {
        this.frequencyHz = found
        this.updateRadioDisplay()
        this.updateFrequencyDisplay()
      }
    } finally {
      this.setTransportBusy(false)
    }
  }

  private async setBand(band: BandName) {
    if (band === this.band) return
    this.band = band
    this.syncBandTabs()
    const scanBand = SCAN_BANDS[band]
    const target = Math.min(Math.max(this.frequencyHz, scanBand.startHz), scanBand.endHz)
    await this.applyModeAndTune(scanBand.mode, band === 'fm' ? 98_000_000 : target)
  }

  /** Applies a demod mode and frequency together, restarting the pipeline if it changed. */
  private async applyModeAndTune(mode: DemodModeName, hz: number, deemphasisUs?: number) {
    const modeChanged = mode !== this.mode
    this.mode = mode
    this.frequencyHz = Math.max(MIN_FREQUENCY_HZ, Math.min(MAX_FREQUENCY_HZ, hz))
    this.band = hz < 30_000_000 ? 'am' : 'fm'
    this.syncModeButtons()
    this.syncBandTabs()
    this.updateFrequencyDisplay()
    this.updateRadioDisplay()

    if (this.running) {
      if (modeChanged) await this.restart()
      else this.pipeline.tune(this.frequencyHz)
    }
    if (deemphasisUs && this.running) this.pipeline.setDeemphasis(deemphasisUs)
  }

  private async setMode(mode: DemodModeName) {
    if (mode === this.mode) return
    this.mode = mode
    this.syncModeButtons()
    if (this.running) await this.restart()
  }

  // -- Scanning -----------------------------------------------------------

  private async scan() {
    if (!this.running) {
      this.showNotice('Connect a receiver first, then scan the band.', false)
      return
    }
    if (this.scanner.isScanning) {
      this.scanner.cancel()
      return
    }
    this.stations = []
    this.renderStations()
    this.scanButton.classList.add('scanning')
    this.scanButton.replaceChildren(icon('stop'), document.createTextNode('Stop'))

    const band = SCAN_BANDS[this.band]
    const returnTo = this.frequencyHz
    try {
      await this.scanner.scan(band, {
        onStation: (station) => {
          this.stations.push(station)
          this.stations.sort((a, b) => a.frequencyHz - b.frequencyHz)
          this.renderStations()
        },
        onProgress: (fraction) => {
          this.scanButton.style.setProperty('--scan-progress', `${Math.round(fraction * 100)}%`)
        },
      })
    } finally {
      this.scanButton.classList.remove('scanning')
      this.scanButton.style.removeProperty('--scan-progress')
      this.scanButton.replaceChildren(icon('radar'), document.createTextNode('Scan band'))
      // Return to where the user was before scanning.
      if (this.running) {
        this.frequencyHz = returnTo
        this.pipeline.tune(returnTo)
        this.updateRadioDisplay()
      }
    }
  }

  private renderStations() {
    this.stationList.replaceChildren()
    if (this.stations.length === 0) {
      const empty = el(
        'p',
        'hint',
        this.scanner.isScanning ? 'Scanning…' : 'Scan the band to catalogue stations.',
      )
      this.stationList.append(empty)
      return
    }
    for (const station of this.stations) {
      const row = el('button', 'station-row')
      const freq = formatFrequency(station.frequencyHz)
      row.append(el('span', 'station-freq', `${freq.value} ${freq.unit}`))
      const bars = Math.max(1, Math.min(5, Math.round(station.strengthDb / 8)))
      const strength = el('span', 'station-strength')
      strength.append(el('span', undefined, '▂▃▅▆▇'.slice(0, bars)))
      row.append(strength)
      row.addEventListener('click', () => this.setFrequency(station.frequencyHz))
      this.stationList.append(row)
    }
  }

  private setTransportBusy(busy: boolean) {
    for (const button of this.radioView.querySelectorAll<HTMLButtonElement>('.radio-btn')) {
      button.disabled = busy
    }
  }

  // -- Presets ------------------------------------------------------------

  private loadPresets() {
    try {
      const raw = localStorage.getItem(PRESET_KEY)
      if (raw) {
        const parsed = JSON.parse(raw) as (PresetSlot | null)[]
        if (Array.isArray(parsed)) {
          this.presets = parsed.slice(0, PRESET_SLOTS)
          while (this.presets.length < PRESET_SLOTS) this.presets.push(null)
        }
      }
    } catch {
      // A corrupt or unavailable store just means no saved presets.
    }
  }

  private savePreset(slot: number) {
    const label =
      this.band === 'am'
        ? `${Math.round(this.frequencyHz / 1000)}k`
        : `${(this.frequencyHz / 1e6).toFixed(1)}`
    this.presets[slot] = {
      frequencyHz: this.frequencyHz,
      band: this.band,
      label,
    }
    try {
      localStorage.setItem(PRESET_KEY, JSON.stringify(this.presets))
    } catch {
      // Non-fatal; the preset still holds for this session.
    }
    this.renderPresets()
  }

  private async recallPreset(slot: number) {
    const preset = this.presets[slot]
    if (!preset) {
      this.savePreset(slot)
      return
    }
    if (preset.band !== this.band) {
      await this.setBand(preset.band)
    }
    this.setFrequency(preset.frequencyHz)
  }

  private renderPresets() {
    this.presetButtons.forEach((button, slot) => {
      const preset = this.presets[slot]
      button.replaceChildren()
      button.classList.toggle('empty', !preset)
      if (preset) {
        button.append(el('span', 'preset-num', String(slot + 1)))
        button.append(el('span', 'preset-freq', preset.label))
      } else {
        button.append(el('span', 'preset-num', String(slot + 1)))
        button.append(el('span', 'preset-empty', 'empty'))
      }
    })
  }

  // -- Lifecycle ----------------------------------------------------------

  private async toggle() {
    if (this.running) await this.stop()
    else await this.startPipeline()
  }

  private async startPipeline() {
    this.clearNotice()
    this.connectButton.disabled = true
    this.connectButton.textContent = 'Connecting…'

    try {
      await this.pipeline.requestDevice()
      await this.pipeline.start(this.frequencyHz, this.mode)

      const defaults = defaultsFor(this.mode)
      this.pipeline.setSquelch(defaults.squelch, defaults.squelchThreshold)
      this.pipeline.setAgc(defaults.agc)
      if (this.mode === 'wfm') this.pipeline.setDeemphasis(DEFAULT_DEEMPHASIS_US)

      this.stopDemo()

      // The displays are optional and must never take the receiver down with them. WebGPU
      // can be absent or, embedded in a cross-origin-isolated iframe, throw while setting up
      // a context — and until now that threw straight into the catch below, which stops the
      // pipeline and, mid-initialisation, cancels the device's own control transfers ("the
      // transfer was cancelled"), leaving no audio. Audio and capture are already running by
      // this point, so a rendering failure is isolated here: the waterfall goes dark, the
      // radio keeps playing.
      try {
        this.render = await startRendering(this.pipeline, this.canvases)
        if (!this.render.usingGpu) this.showNotice(this.render.adapterInfo, false)
      } catch (error) {
        this.render = undefined
        console.warn('sdr-bullet: displays unavailable', error)
        this.showNotice('The GPU displays could not start here — audio is unaffected.', false)
      }

      this.running = true
      this.connectButton.textContent = 'Disconnect'
    } catch (error) {
      this.onError(error instanceof Error ? error.message : String(error), true)
      await this.stop()
    } finally {
      this.connectButton.disabled = false
    }
  }

  private async restart() {
    const wasRunning = this.running
    await this.stop({ keepDemoOff: true })
    if (wasRunning) await this.startPipeline()
  }

  private async stop(options: { keepDemoOff?: boolean } = {}) {
    this.scanner.cancel()
    this.render?.stop()
    this.render = undefined
    await this.pipeline.stop()
    this.running = false
    this.connectButton.textContent = 'Connect receiver'
    if (!options.keepDemoOff) this.startDemo()
  }

  // -- Status -------------------------------------------------------------

  private onStatus(status: PipelineStatus) {
    if (status.dsp) {
      const stereo = status.dsp.stereo
      this.radioStereo.textContent = stereo ? 'STEREO' : 'MONO'
      this.radioStereo.className = `badge ${stereo ? 'on' : 'off'}`
      this.radioName.textContent = status.dsp.rds.stationName || '—'
      this.radioText.textContent = status.dsp.rds.radioText || ''
      this.rdsName.textContent = status.dsp.rds.stationName || '—'
      this.rdsText.textContent = status.dsp.rds.radioText || ''
      // A rough strength indicator from the pilot level and ring health.
      const level = Math.min(1, status.dsp.pilotLevel * 40 + (status.dsp.squelchOpen ? 0.25 : 0))
      this.signalBar.style.width = `${Math.round(level * 100)}%`
    }

    const rows: [string, string][] = []
    if (status.capture) {
      rows.push(['Tuned', formatFrequency(status.capture.tunedHz).value + ' MHz'])
      rows.push(['Band', status.capture.band])
      rows.push(['PLL', status.capture.locked ? 'locked' : 'unlocked'])
      rows.push(['Throughput', `${(status.capture.bytesPerSecond / 1e6).toFixed(1)} MB/s`])
      if (status.capture.dropped > 0) rows.push(['Dropped', String(status.capture.dropped)])
    }
    if (status.dsp) {
      rows.push(['Stereo', status.dsp.stereo ? 'yes' : 'no'])
      rows.push(['Squelch', status.dsp.squelchOpen ? 'open' : 'closed'])
      rows.push(['Ring fill', `${Math.round(status.dsp.ringFill * 100)}%`])
    }
    rows.push(['Audio underruns', String(status.audio.underruns)])
    rows.push(['Latency', `${Math.round(status.audioLatency * 1000)} ms`])

    this.stats.replaceChildren()
    for (const [key, value] of rows) {
      this.stats.append(el('span', 'key', key), el('span', 'value', value))
    }

    this.diagnostics.replaceChildren()
    this.diagnostics.append(el('span', undefined, `SIMD: ${status.simdBackend}`))
    if (this.render) this.diagnostics.append(el('span', undefined, `GPU: ${this.render.adapterInfo}`))
    if (status.device) {
      this.diagnostics.append(
        el('span', undefined, status.device.isV4 ? 'RTL-SDR Blog V4' : 'RTL2832U'),
      )
    }
  }

  private updateRadioDisplay() {
    // A radio-style readout: whole kilohertz on AM, two decimals of megahertz on FM, rather
    // than the spectrum view's full precision.
    const readout =
      this.band === 'am'
        ? { value: Math.round(this.frequencyHz / 1000).toString(), unit: 'kHz' }
        : { value: (this.frequencyHz / 1e6).toFixed(2), unit: 'MHz' }
    this.radioFreq.replaceChildren(
      document.createTextNode(readout.value),
      el('span', 'radio-unit', readout.unit),
    )
    this.radioBandLabel.textContent = `${SCAN_BANDS[this.band].label} · ${this.mode.toUpperCase()}`
  }

  private updateFrequencyDisplay() {
    const { value, unit } = formatFrequency(this.frequencyHz)
    this.frequencyDisplay.replaceChildren(
      document.createTextNode(value),
      el('span', 'unit', unit),
    )
  }

  private syncModeButtons() {
    for (const [name, button] of this.modeButtons) {
      button.classList.toggle('active', name === this.mode)
    }
  }

  private syncBandTabs() {
    for (const [name, button] of this.bandTabs) {
      button.classList.toggle('active', name === this.band)
    }
  }

  private onError(message: string, fatal: boolean) {
    this.showNotice(message, fatal)
    if (fatal) void this.stop()
  }

  private showNotice(message: string, error: boolean) {
    const notice = el('div', error ? 'notice error' : 'notice')
    if (message.includes('\n')) {
      const [first, ...rest] = message.split('\n')
      notice.append(el('strong', undefined, first))
      notice.append(el('pre', undefined, rest.join('\n').trim()))
    } else {
      notice.textContent = message
    }
    this.noticeArea.replaceChildren(notice)
  }

  private clearNotice() {
    this.noticeArea.replaceChildren()
  }

  /** Tears everything down. The React wrapper calls this on unmount. */
  async destroy() {
    this.stopDemo()
    await this.stop({ keepDemoOff: true })
    this.root.replaceChildren()
  }
}
