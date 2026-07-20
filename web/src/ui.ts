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
import { ScopeDisplay } from './scope.js'
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
/** Which visualisation fills the stage — the "additional views" over one shared signal. */
type DisplayMode = 'panorama' | 'constellation' | 'scope'

const DISPLAY_TABS: { mode: DisplayMode; label: string }[] = [
  { mode: 'panorama', label: 'Panorama' },
  { mode: 'constellation', label: 'Constellation' },
  { mode: 'scope', label: 'Scope' },
]

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
  private displayMode: DisplayMode = 'panorama'
  private running = false
  private lastHealthLog = 0
  private stations: FoundStation[] = []
  private presets: (PresetSlot | null)[] = new Array(PRESET_SLOTS).fill(null)

  private readonly canvases: {
    spectrum: HTMLCanvasElement
    waterfall: HTMLCanvasElement
    constellation: HTMLCanvasElement
  }
  private readonly demoCanvas = el('canvas', 'demo-canvas')
  private readonly scopeCanvas = el('canvas', 'scope-canvas')
  private scope: ScopeDisplay | undefined

  // Console shell: a top bar, then a workspace split into a display stage and a control rail.
  private readonly stage = el('div', 'stage')
  private readonly rail = el('div', 'rail')
  private readonly displayArea = el('div', 'display-area')
  private readonly displayTabs = el('div', 'display-tabs')
  private readonly radioRail = el('div', 'rail-view radio-rail')
  private readonly spectrumRail = el('div', 'rail-view spectrum-rail')
  private readonly signalDb = el('span', 'signal-db', '—')

  // Shared DOM references.
  private readonly diagnostics = el('div', 'diagnostics')
  private readonly connectButton = el('button', 'primary connect-btn', 'Connect receiver')
  private readonly statusDot = el('span', 'status-dot')
  private readonly statusText = el('span', 'status-text', 'Offline')
  private readonly noticeArea = el('div')
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

    // Top bar: a compact app mark, the Radio/Spectrum switch, and the primary Connect action
    // with a live-status pill. The host page already headlines "sdr-bullet", so this bar
    // deliberately does *not* repeat that title — it is chrome, not a second masthead.
    const topbar = el('div', 'topbar')
    const mark = el('div', 'brand-mark')
    mark.append(icon('radar'), el('span', 'mark-text', 'Receiver'))

    const actions = el('div', 'topbar-actions')
    const statusPill = el('div', 'status-pill')
    statusPill.append(this.statusDot, this.statusText)
    actions.append(statusPill, this.connectButton)

    topbar.append(mark, this.buildViewToggle(), actions)
    this.root.append(topbar)
    this.root.classList.add('offline')

    const capability = Pipeline.capabilities()
    if (!capability.ok) {
      const notice = el('div', 'notice error')
      notice.append(el('strong', undefined, 'This browser cannot run the receiver.'))
      notice.append(el('pre', undefined, capability.reason ?? ''))
      this.root.append(notice)
      return
    }

    this.root.append(this.noticeArea)

    // Workspace: a display stage and a control rail, both bounded by the frame. The rail
    // scrolls internally if its panels overflow, so the app itself never scrolls — it is
    // sized to fit the embed exactly rather than running off the bottom of it.
    const workspace = el('div', 'workspace')
    this.buildStage()
    this.buildRadioRail()
    this.buildSpectrumRail()
    this.rail.append(this.radioRail, this.spectrumRail)
    workspace.append(this.stage, this.rail)
    this.root.append(workspace)

    const footer = el('div', 'app-footer')
    footer.append(this.diagnostics)
    this.root.append(footer)

    this.setView('radio')
    this.setDisplayMode('panorama')
    this.updateRadioDisplay()
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

  /**
   * The display stage: a selector for which visualisation fills it, the display area itself,
   * and the tuning transport pinned under it. The three displays are stacked in the same box
   * and cross-faded by the selector rather than torn down, so the GPU render targets keep
   * their size and switching views is instant.
   */
  private buildStage() {
    for (const { mode, label } of DISPLAY_TABS) {
      const button = el('button', undefined, label)
      button.dataset['disp'] = mode
      button.addEventListener('click', () => this.setDisplayMode(mode))
      this.displayTabs.append(button)
    }

    // Panorama — the GPU spectrum trace over the scrolling waterfall, with the demo / 2D
    // fallback canvas overlaid on top of it.
    const panorama = el('div', 'disp disp-panorama active')
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
    const demoWrap = el('div', 'canvas-wrap demo-wrap')
    demoWrap.append(this.demoCanvas)
    demoWrap.append(el('div', 'canvas-label', 'Demo signal · Connect for live reception'))
    panorama.append(this.gpuCanvasWrap, demoWrap)

    // Constellation — the GPU I/Q density plot.
    const constellation = el('div', 'disp disp-constellation')
    const constellationWrap = el('div', 'canvas-wrap constellation-wrap')
    constellationWrap.append(this.canvases.constellation)
    constellationWrap.append(el('div', 'canvas-label', 'Constellation · I/Q density'))
    constellation.append(
      constellationWrap,
      el('div', 'disp-hint', 'Connect the receiver to plot the constellation'),
    )

    // Scope — the Canvas2D baseband oscilloscope.
    const scope = el('div', 'disp disp-scope')
    const scopeWrap = el('div', 'canvas-wrap scope-wrap')
    scopeWrap.append(this.scopeCanvas)
    scopeWrap.append(el('div', 'canvas-label', 'Scope · baseband I/Q vs time'))
    scope.append(scopeWrap)

    this.displayArea.append(panorama, constellation, scope)
    this.stage.append(this.displayTabs, this.displayArea, this.buildTransport())
  }

  /**
   * The tuning transport, pinned under the display in both views: band select, a large
   * frequency readout flanked by seek/tune (the readout itself tunes on scroll or arrow
   * keys), and a compact "what am I hearing" strip — signal meter, stereo, station name.
   */
  private buildTransport(): HTMLElement {
    const bar = el('div', 'transport')

    const bands = el('div', 'transport-bands')
    for (const name of ['fm', 'am'] as const) {
      const button = el('button', undefined, SCAN_BANDS[name].label)
      button.addEventListener('click', () => void this.setBand(name))
      this.bandTabs.set(name, button)
      bands.append(button)
    }

    const mk = (title: string, glyph: string, onClick: () => void) => {
      const b = el('button', 'radio-btn')
      b.title = title
      b.setAttribute('aria-label', title)
      b.append(icon(glyph))
      b.addEventListener('click', onClick)
      return b
    }
    this.radioFreq.tabIndex = 0
    this.radioFreq.addEventListener('wheel', (event) => {
      event.preventDefault()
      this.tuneStep(event.deltaY < 0 ? 1 : -1)
    }, { passive: false })
    this.radioFreq.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowUp' || event.key === 'ArrowRight') this.tuneStep(1)
      if (event.key === 'ArrowDown' || event.key === 'ArrowLeft') this.tuneStep(-1)
    })
    const freqWrap = el('div', 'tuner-freq')
    freqWrap.append(this.radioFreq, this.radioBandLabel)
    const tuner = el('div', 'tuner')
    tuner.append(
      mk('Seek down', 'keyboard_double_arrow_left', () => void this.seek(-1)),
      mk('Tune down', 'chevron_left', () => this.tuneStep(-1)),
      freqWrap,
      mk('Tune up', 'chevron_right', () => this.tuneStep(1)),
      mk('Seek up', 'keyboard_double_arrow_right', () => void this.seek(1)),
    )

    const meter = el('div', 'meter')
    meter.append(this.signalBar)
    const meterWrap = el('div', 'meter-wrap')
    meterWrap.append(el('span', 'meter-cap', 'Signal'), meter, this.signalDb)
    const rds = el('div', 'transport-rds')
    this.radioName.textContent = '—'
    rds.append(this.radioName, this.radioText)
    const readout = el('div', 'transport-readout')
    readout.append(this.radioStereo, meterWrap, rds)

    bar.append(bands, tuner, readout)
    return bar
  }

  // -- Radio rail (consumer controls) -------------------------------------

  private buildRadioRail() {
    const audio = el('div', 'panel')
    audio.append(el('h2', undefined, 'Audio'))
    const volumeRow = el('div', 'control-row')
    volumeRow.append(el('label', undefined, 'Volume'))
    const volume = el('input')
    volume.type = 'range'
    volume.min = '0'
    volume.max = '100'
    volume.value = '70'
    volume.addEventListener('input', () => this.pipeline.setVolume(Number(volume.value) / 100))
    volumeRow.append(volume)
    audio.append(volumeRow)
    this.radioRail.append(audio)

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
    presetPanel.append(el('p', 'hint', 'Click to recall · right-click to save'))
    this.radioRail.append(presetPanel)

    // Station catalogue + scan.
    const scanPanel = el('div', 'panel scan-panel')
    const scanHeader = el('div', 'panel-header')
    scanHeader.append(el('h2', undefined, 'Stations'))
    this.scanButton.append(icon('radar'), document.createTextNode('Scan'))
    this.scanButton.addEventListener('click', () => void this.scan())
    scanHeader.append(this.scanButton)
    scanPanel.append(scanHeader)
    scanPanel.append(this.stationList)
    this.radioRail.append(scanPanel)

    this.renderStations()
    this.syncBandTabs()
    this.renderPresets()
  }

  // -- Spectrum rail (advanced controls) ---------------------------------

  private buildSpectrumRail() {
    const tuning = el('div', 'panel')
    tuning.append(el('h2', undefined, 'Tuning'))
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
    this.spectrumRail.append(tuning)

    const modePanel = el('div', 'panel')
    modePanel.append(el('h2', undefined, 'Demodulator'))
    const modeGroup = el('div', 'mode-group')
    for (const { name, label } of MODES) {
      const button = el('button', undefined, label)
      button.addEventListener('click', () => void this.setMode(name))
      this.modeButtons.set(name, button)
      modeGroup.append(button)
    }
    modePanel.append(modeGroup)
    this.syncModeButtons()
    this.spectrumRail.append(modePanel)

    this.spectrumRail.append(this.buildAudioPanel())

    const rdsPanel = el('div', 'panel rds-panel')
    rdsPanel.append(el('h2', undefined, 'Radio data'))
    rdsPanel.append(this.rdsName, this.rdsText)
    this.spectrumRail.append(rdsPanel)

    const statusPanel = el('div', 'panel')
    statusPanel.append(el('h2', undefined, 'Signal'))
    statusPanel.append(this.stats)
    this.spectrumRail.append(statusPanel)
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
    this.radioRail.classList.toggle('hidden', view !== 'radio')
    this.spectrumRail.classList.toggle('hidden', view !== 'spectrum')
    for (const button of this.viewToggle.querySelectorAll('button')) {
      button.classList.toggle('active', button.dataset['view'] === view)
    }
  }

  /** Cross-fades the stage to the chosen visualisation without disturbing the render loop. */
  private setDisplayMode(mode: DisplayMode) {
    this.displayMode = mode
    for (const disp of this.displayArea.querySelectorAll<HTMLElement>('.disp')) {
      disp.classList.toggle('active', disp.classList.contains(`disp-${mode}`))
    }
    for (const button of this.displayTabs.querySelectorAll('button')) {
      button.classList.toggle('active', button.dataset['disp'] === mode)
    }
  }

  // -- Demo ---------------------------------------------------------------

  /** Creates and starts the 2D canvas display if it does not exist yet. */
  private ensureDemo() {
    if (this.demo) return
    try {
      this.demo = new DemoDisplay(this.demoCanvas)
      this.demo.start()
      this.demoResizeObserver = new ResizeObserver(() => this.demo?.resize())
      this.demoResizeObserver.observe(this.demoCanvas)
    } catch {
      // A missing 2D context is not worth failing the whole app over.
    }
  }

  /** The scope runs for the life of the app — an idle sweep offline, live baseband when
   * connected — so switching to the Scope tab is always instant and never blank. */
  private ensureScope() {
    if (this.scope) return
    try {
      this.scope = new ScopeDisplay(this.scopeCanvas)
      this.scope.start()
    } catch {
      // No 2D context: the scope tab simply stays empty.
    }
  }

  /** The synthetic pre-connect demo. */
  private startDemo() {
    if (this.running) return
    this.ensureDemo()
    this.ensureScope()
    this.demo?.setSource(null)
    this.scope?.setSource(null)
    document.querySelector('.demo-wrap')?.classList.remove('hidden')
    this.gpuCanvasWrap.classList.add('hidden')
  }

  /**
   * The live 2D spectrum shown when WebGPU is unavailable or drops out — the same canvas as
   * the demo, but fed the real spectrum so the receiver still has a working display.
   */
  private startLiveFallback() {
    this.ensureDemo()
    this.demo?.setSource(() => this.pipeline.latestSpectrum())
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

  /** WebGPU dropped out after start-up; switch the displays to the live 2D fallback. */
  private onGpuLost() {
    if (!this.running) return
    this.render?.stop()
    this.render = undefined
    this.startLiveFallback()
    this.showNotice('The GPU dropped out — switched to a 2D spectrum. Audio is unaffected.', false)
  }

  // -- Tuning -------------------------------------------------------------

  private setFrequency(hz: number) {
    this.frequencyHz = Math.max(MIN_FREQUENCY_HZ, Math.min(MAX_FREQUENCY_HZ, hz))
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
    for (const button of this.stage.querySelectorAll<HTMLButtonElement>('.radio-btn')) {
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

  /** Reflects the connection state across the header pill, the button, and the root class. */
  private setConnectionUi(state: 'offline' | 'connecting' | 'live') {
    this.root.classList.toggle('offline', state === 'offline')
    this.root.classList.toggle('connecting', state === 'connecting')
    this.root.classList.toggle('live', state === 'live')
    this.statusText.textContent =
      state === 'live' ? 'Live' : state === 'connecting' ? 'Connecting' : 'Offline'
    this.connectButton.textContent =
      state === 'live' ? 'Disconnect' : state === 'connecting' ? 'Connecting…' : 'Connect receiver'
    this.connectButton.disabled = state === 'connecting'
  }

  private async startPipeline() {
    this.clearNotice()
    this.setConnectionUi('connecting')
    // Unlock audio now, while the click's user activation is still live — before
    // requestDevice() opens the USB chooser and spends it. A context created after the
    // chooser is born suspended and stays silent, which is the "connected but no audio" bug.
    void this.pipeline.unlockAudio()

    try {
      await this.pipeline.requestDevice()
      await this.pipeline.start(this.frequencyHz, this.mode)

      const defaults = defaultsFor(this.mode)
      this.pipeline.setSquelch(defaults.squelch, defaults.squelchThreshold)
      this.pipeline.setAgc(defaults.agc)
      if (this.mode === 'wfm') this.pipeline.setDeemphasis(DEFAULT_DEEMPHASIS_US)

      // Audio and capture are live now. The displays are optional and must never take the
      // receiver down: mark it live before touching the GPU so a display failure is isolated.
      this.running = true
      this.setConnectionUi('live')
      // Feed the scope the live baseband; it peeks the same I/Q slot the constellation drains.
      this.scope?.setSource(() => this.pipeline.peekConstellation())

      // Try WebGPU; if it is unavailable or throws — common for WebGPU on Linux inside this
      // cross-origin-isolated frame, where the Vulkan external-semaphore share fails — fall
      // back to a live 2D spectrum on the demo canvas so the receiver still has a display.
      let gpuOk = false
      try {
        this.render = await startRendering(this.pipeline, this.canvases, () => this.onGpuLost())
        gpuOk = this.render.usingGpu
      } catch (error) {
        this.render = undefined
        console.warn('sdr-bullet: WebGPU displays unavailable', error)
      }

      if (gpuOk) {
        this.stopDemo()
      } else {
        this.render?.stop()
        this.render = undefined
        this.startLiveFallback()
        this.showNotice(
          'WebGPU is unavailable here — showing a 2D spectrum. Audio is full quality.',
          false,
        )
      }
    } catch (error) {
      this.onError(error instanceof Error ? error.message : String(error), true)
      await this.stop()
    } finally {
      if (!this.running) this.setConnectionUi('offline')
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
    this.scope?.setSource(null)
    this.setConnectionUi('offline')
    if (!options.keepDemoOff) this.startDemo()
  }

  // -- Status -------------------------------------------------------------

  private onStatus(status: PipelineStatus) {
    // A throttled health line, so a silent pipeline can be diagnosed from the console:
    // throughput 0 → the device is not streaming; framesDelivered stuck at 0 → the audio
    // context never started (worklet not running); underruns climbing while framesDelivered
    // grows → the ring is starving. All three read very differently here.
    const now = performance.now()
    if (now - this.lastHealthLog > 3000) {
      this.lastHealthLog = now
      const cap = status.capture
      console.info(
        `[sdr] throughput ${cap ? (cap.bytesPerSecond / 1e6).toFixed(2) : '—'} MB/s · ` +
          `tuned ${cap ? (cap.tunedHz / 1e6).toFixed(3) : '—'} MHz · ` +
          `PLL ${cap ? (cap.locked ? 'lock' : 'unlock') : '—'} · ` +
          `signal ${cap ? cap.powerDbfs.toFixed(1) : '—'} dBFS · ` +
          `pilot ${status.dsp ? status.dsp.pilotLevel.toFixed(4) : '—'} ${status.dsp?.stereo ? 'STEREO' : 'mono'} · ` +
          `audio-real ${status.dsp ? status.dsp.audioAutocorr.toFixed(2) : '—'} · ` +
          `dropped ${cap ? cap.dropped : '—'} · ` +
          `IQ ring ${status.dsp ? Math.round(status.dsp.ringFill * 100) : 0}% · ` +
          `audio frames ${status.audio.framesDelivered} · underruns ${status.audio.underruns}`,
      )
    }

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
      rows.push(['Signal', `${status.capture.powerDbfs.toFixed(1)} dBFS`])
      rows.push(['Throughput', `${(status.capture.bytesPerSecond / 1e6).toFixed(1)} MB/s`])
      if (status.capture.dropped > 0) rows.push(['Dropped', String(status.capture.dropped)])
      // Radio-view strength meter from the real capture power: a noise floor sits near
      // -55 dBFS, a strong local station near -10, so map that span onto the bar.
      const strength = Math.max(0, Math.min(1, (status.capture.powerDbfs + 55) / 50))
      this.signalBar.style.width = `${Math.round(strength * 100)}%`
      this.signalDb.textContent = `${status.capture.powerDbfs.toFixed(0)} dBFS`
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
    this.scope?.stop()
    this.scope = undefined
    await this.stop({ keepDemoOff: true })
    this.root.replaceChildren()
  }
}
