import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Tensor,
  getGlobalLiteRtPromise,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
  supportsFeature,
  type Accelerator,
  type CompiledModel,
} from '@litertjs/core'
import './App.css'

type Normalization = {
  mode: 'mobilenet_v2'
  scale: number
  offset: number
}

type ModelMetadata = {
  model: string
  inputShape: [number, number, number, number]
  inputDtype: 'float32'
  normalization: Normalization
  classOrder: ['not_hotdog', 'hotdog'] | ['hotdog', 'not_hotdog']
  threshold: number
  gitSha?: string
  metrics?: {
    accuracy: number
    hotdogPrecision: number
    hotdogRecall: number
  }
}

type ClassifierState =
  | { kind: 'loading'; message: string; wasmReady: boolean }
  | { kind: 'missing'; message: string; wasmReady: boolean }
  | { kind: 'error'; message: string; wasmReady: boolean }
  | {
      kind: 'ready'
      accelerator: Accelerator
      fallbackMessage?: string
      metadata: ModelMetadata
      model: CompiledModel
      wasmReady: true
    }

type Verdict = {
  rawProbability: number
  smoothProbability: number
  latencyMs: number
  source: 'camera' | 'file'
  session: number
}

const appBase = import.meta.env.BASE_URL
const targetIntervalMs = 100
const smoothing = 0.72

const assetPath = (path: string) => `${appBase}${path.replace(/^\//, '')}`

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const parseNumber = (value: unknown, label: string) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new Error(`metadata.${label} must be a number`)
  }

  return value
}

const parseMetadata = (value: unknown): ModelMetadata => {
  if (!isRecord(value)) {
    throw new Error('metadata.json must contain an object')
  }

  const normalization = value.normalization
  const metrics = value.metrics

  if (typeof value.model !== 'string') {
    throw new Error('metadata.model must be a string')
  }

  if (
    !Array.isArray(value.inputShape) ||
    value.inputShape.length !== 4 ||
    !value.inputShape.every((item) => typeof item === 'number')
  ) {
    throw new Error('metadata.inputShape must be [1, height, width, channels]')
  }

  if (value.inputDtype !== 'float32') {
    throw new Error('Only float32 model input is supported')
  }

  if (!isRecord(normalization) || normalization.mode !== 'mobilenet_v2') {
    throw new Error('Only mobilenet_v2 normalization is supported')
  }

  if (
    !Array.isArray(value.classOrder) ||
    value.classOrder.length !== 2 ||
    !value.classOrder.includes('hotdog') ||
    !value.classOrder.includes('not_hotdog')
  ) {
    throw new Error('metadata.classOrder must include hotdog and not_hotdog')
  }

  const parsedMetrics = isRecord(metrics)
    ? {
        accuracy: parseNumber(metrics.accuracy, 'metrics.accuracy'),
        hotdogPrecision: parseNumber(metrics.hotdogPrecision, 'metrics.hotdogPrecision'),
        hotdogRecall: parseNumber(metrics.hotdogRecall, 'metrics.hotdogRecall'),
      }
    : undefined

  return {
    model: value.model,
    inputShape: value.inputShape as ModelMetadata['inputShape'],
    inputDtype: 'float32',
    normalization: {
      mode: 'mobilenet_v2',
      scale: parseNumber(normalization.scale, 'normalization.scale'),
      offset: parseNumber(normalization.offset, 'normalization.offset'),
    },
    classOrder: value.classOrder as ModelMetadata['classOrder'],
    threshold: parseNumber(value.threshold, 'threshold'),
    gitSha: typeof value.gitSha === 'string' ? value.gitSha : undefined,
    metrics: parsedMetrics,
  }
}

const fetchJson = async (url: string) => {
  const response = await fetch(url, { cache: 'no-store' })

  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`)
  }

  const contentType = response.headers.get('content-type') ?? ''

  if (!contentType.includes('application/json')) {
    throw new Error(`${url} was not JSON`)
  }

  try {
    return (await response.json()) as unknown
  } catch {
    throw new Error(`${url} is not valid JSON`)
  }
}

const formatPercent = (value: number) => `${Math.round(value * 100)}%`

const clampProbability = (value: number) => Math.min(1, Math.max(0, value))

const isLiteRtLog = (args: Parameters<typeof console.error>) =>
  typeof args[0] === 'string' && /^(INFO|WARNING): \[/.test(args[0])

const withLiteRtConsoleFilter = async <T,>(work: () => Promise<T>) => {
  const originalError = console.error

  console.error = (...args: Parameters<typeof console.error>) => {
    if (isLiteRtLog(args)) {
      console.info(...args)
      return
    }

    originalError(...args)
  }

  try {
    return await work()
  } finally {
    console.error = originalError
  }
}

const ensureLiteRtLoaded = async () => {
  const existingLoad = getGlobalLiteRtPromise()

  if (existingLoad) {
    await existingLoad
    return
  }

  const jspi = await supportsFeature('jspi')

  try {
    await withLiteRtConsoleFilter(() => loadLiteRt(assetPath('wasm/'), jspi ? { jspi: true } : undefined))
  } catch (error) {
    const retryLoad = getGlobalLiteRtPromise()

    if (error instanceof Error && error.message.includes('already loading / loaded') && retryLoad) {
      await retryLoad
      return
    }

    throw error
  }
}

const App = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const frameRef = useRef<number | null>(null)
  const inFlightRef = useRef(false)
  const lastInferenceRef = useRef(0)
  // Bumped whenever the input changes: camera start, camera stop, or a new file. Inference is
  // async, so a frame captured before the change can still resolve after it. The token lets a
  // late result recognise that it belongs to an input nobody is looking at any more.
  const sessionRef = useRef(0)
  const classifierRef = useRef<ClassifierState>({ kind: 'loading', message: 'Loading LiteRT.js…', wasmReady: false })

  const [classifier, setClassifierState] = useState<ClassifierState>(classifierRef.current)
  const [verdict, setVerdict] = useState<Verdict | null>(null)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)

  const setClassifier = useCallback((next: ClassifierState) => {
    classifierRef.current = next
    setClassifierState(next)
  }, [])

  useEffect(() => {
    let cancelled = false
    let loadedModel: CompiledModel | null = null

    const loadClassifier = async () => {
      let wasmReady = false

      try {
        await ensureLiteRtLoaded()
        wasmReady = true

        if (cancelled) {
          return
        }

        setClassifier({ kind: 'loading', message: 'Reading model metadata…', wasmReady: true })
        const metadata = parseMetadata(await fetchJson(assetPath('models/metadata.json')))
        const accelerator: Accelerator = 'wasm'
        const fallbackMessage = isWebGPUSupported()
          ? 'Using WASM to match TensorFlow Lite scores.'
          : 'WebGPU unavailable, using WASM.'

        loadedModel = await loadAndCompile(assetPath(`models/${metadata.model}`), { accelerator })

        if (cancelled) {
          loadedModel.delete()
          return
        }

        setClassifier({
          kind: 'ready',
          accelerator,
          fallbackMessage,
          metadata,
          model: loadedModel,
          wasmReady: true,
        })
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown LiteRT error'
        const isMissing = message.includes('metadata.json') || message.includes('.tflite') || message.includes('404')

        if (!cancelled) {
          setClassifier({
            kind: isMissing ? 'missing' : 'error',
            wasmReady,
            message: isMissing
              ? 'Model files are not here yet. Drop models/metadata.json and models/not-hotdog.tflite, then rebuild.'
              : message,
          })
        }
      }
    }

    void loadClassifier()

    return () => {
      cancelled = true
      loadedModel?.delete()
    }
  }, [setClassifier])

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    sessionRef.current += 1
    setIsCameraActive(false)
  }, [])

  useEffect(() => stopCamera, [stopCamera])

  const getCanvasContext = useCallback((width: number, height: number) => {
    const canvas = canvasRef.current ?? document.createElement('canvas')
    canvasRef.current = canvas
    canvas.width = width
    canvas.height = height

    const context = canvas.getContext('2d', { willReadFrequently: true })

    if (!context) {
      throw new Error('Could not create image processing canvas')
    }

    return context
  }, [])

  const classifySource = useCallback(
    async (source: CanvasImageSource, sourceWidth: number, sourceHeight: number, inputSource: Verdict['source']) => {
      const current = classifierRef.current
      const session = sessionRef.current

      if (current.kind !== 'ready') {
        return
      }

      const [, inputHeight, inputWidth, channels] = current.metadata.inputShape

      if (channels !== 3) {
        throw new Error(`Expected 3 input channels, got ${channels}`)
      }

      const cropSize = Math.min(sourceWidth, sourceHeight)
      const cropX = (sourceWidth - cropSize) / 2
      const cropY = (sourceHeight - cropSize) / 2
      const context = getCanvasContext(inputWidth, inputHeight)

      context.drawImage(source, cropX, cropY, cropSize, cropSize, 0, 0, inputWidth, inputHeight)

      const pixels = context.getImageData(0, 0, inputWidth, inputHeight).data
      const values = new Float32Array(inputWidth * inputHeight * channels)
      const { scale, offset } = current.metadata.normalization

      for (let pixelIndex = 0, valueIndex = 0; pixelIndex < pixels.length; pixelIndex += 4) {
        values[valueIndex] = pixels[pixelIndex] * scale + offset
        values[valueIndex + 1] = pixels[pixelIndex + 1] * scale + offset
        values[valueIndex + 2] = pixels[pixelIndex + 2] * scale + offset
        valueIndex += 3
      }

      const inputTensor = new Tensor(values, current.metadata.inputShape)
      let outputTensors: Tensor[] = []
      let cpuOutput: Tensor | null = null
      const startedAt = performance.now()

      try {
        outputTensors = await current.model.run(inputTensor)
        const output = outputTensors[0]
        cpuOutput = output.accelerator === 'wasm' ? output : await output.copyTo('wasm')
        const [probability = 0] = cpuOutput.toTypedArray()
        const rawProbability = clampProbability(Number(probability))
        const latencyMs = performance.now() - startedAt

        // The input changed while this ran. Publishing now would let a stopped camera overwrite
        // the file the user just dropped.
        if (sessionRef.current !== session) {
          return
        }

        setVerdict((previous) => {
          // Smoothing exists to stop a live camera flickering between verdicts on noisy frames.
          // A dropped file is a single decision, so it should read its own score rather than
          // inheriting momentum from whatever was on screen before it. Comparing the session too
          // means a restarted camera starts fresh instead of resuming the previous stream's EMA.
          const isContinuous =
            inputSource === 'camera' &&
            previous?.source === 'camera' &&
            previous.session === session

          return {
            rawProbability,
            smoothProbability: isContinuous
              ? previous.smoothProbability * smoothing + rawProbability * (1 - smoothing)
              : rawProbability,
            latencyMs,
            source: inputSource,
            session,
          }
        })
      } finally {
        inputTensor.delete()
        if (cpuOutput && !outputTensors.includes(cpuOutput)) {
          cpuOutput.delete()
        }
        outputTensors.forEach((tensor) => tensor.delete())
      }
    },
    [getCanvasContext],
  )

  const runCameraLoop = useCallback(
    (timestamp: number) => {
      frameRef.current = requestAnimationFrame(runCameraLoop)

      if (timestamp - lastInferenceRef.current < targetIntervalMs || inFlightRef.current) {
        return
      }

      const video = videoRef.current

      if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth === 0) {
        return
      }

      lastInferenceRef.current = timestamp
      inFlightRef.current = true

      classifySource(video, video.videoWidth, video.videoHeight, 'camera')
        .catch((error: unknown) => {
          const message = error instanceof Error ? error.message : 'Camera inference failed'
          setCameraError(message)
        })
        .finally(() => {
          inFlightRef.current = false
        })
    },
    [classifySource],
  )

  useEffect(() => {
    if (!isCameraActive || classifier.kind !== 'ready') {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current)
        frameRef.current = null
      }
      return
    }

    frameRef.current = requestAnimationFrame(runCameraLoop)

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current)
        frameRef.current = null
      }
    }
  }, [classifier.kind, isCameraActive, runCameraLoop])

  const startCamera = useCallback(async () => {
    setCameraError(null)
    setFileName(null)

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('This browser does not expose camera access.')
      }

      stopCamera()
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: 'environment' } },
      })
      const video = videoRef.current

      if (!video) {
        throw new Error('Video element is not ready.')
      }

      streamRef.current = stream
      video.srcObject = stream
      await video.play()
      setIsCameraActive(true)
    } catch (error) {
      stopCamera()
      const message = error instanceof Error ? error.message : 'Camera permission was denied.'
      setCameraError(message)
    }
  }, [stopCamera])

  const classifyFile = useCallback(
    async (file: File) => {
      setCameraError(null)
      setFileName(file.name)
      stopCamera()

      if (!file.type.startsWith('image/')) {
        setCameraError('Drop an image file, preferably with suspicious meat content.')
        return
      }

      try {
        const bitmap = await createImageBitmap(file)
        await classifySource(bitmap, bitmap.width, bitmap.height, 'file')
        bitmap.close()
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Could not classify that image.'
        setCameraError(message)
      }
    },
    [classifySource, stopCamera],
  )

  const onFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const [file] = Array.from(event.currentTarget.files ?? [])

      if (file) {
        void classifyFile(file)
      }

      event.currentTarget.value = ''
    },
    [classifyFile],
  )

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLLabelElement>) => {
      event.preventDefault()
      setIsDragging(false)
      const [file] = Array.from(event.dataTransfer.files)

      if (file) {
        void classifyFile(file)
      }
    },
    [classifyFile],
  )

  const isReady = classifier.kind === 'ready'
  const threshold = isReady ? classifier.metadata.threshold : 1
  const isHotdog = verdict ? verdict.smoothProbability >= threshold : false
  const heroText = useMemo(() => {
    if (!isReady) {
      return classifier.kind === 'loading' ? 'LOADING' : 'MODEL MISSING'
    }

    if (!verdict) {
      return 'HOTDOG?'
    }

    return isHotdog ? 'HOTDOG' : 'NOT HOTDOG'
  }, [classifier.kind, isHotdog, isReady, verdict])

  const heroClass = verdict ? (isHotdog ? 'hotdog' : 'notHotdog') : 'idle'

  return (
    <main className="appShell">
      <section className={`verdictPanel ${heroClass}`} aria-live="polite">
        <div className="noise" />
        <video ref={videoRef} className="cameraPreview" muted playsInline autoPlay />
        <div className="verdictContent">
          <p className="eyebrow">Not Hotdog</p>
          <h1>{heroText}</h1>
          <p className="confidence">
            {verdict
              ? verdict.smoothProbability === verdict.rawProbability
                ? `${formatPercent(verdict.rawProbability)} confidence`
                : `${formatPercent(verdict.smoothProbability)} confidence · raw ${formatPercent(verdict.rawProbability)}`
              : isReady
                ? `Threshold ${formatPercent(threshold)}. Bring on the tube steak.`
                : classifier.message}
          </p>
        </div>
      </section>

      <section className="controlDeck">
        <div className="statusGrid">
          <div>
            <span>Runtime</span>
            <strong>{classifier.wasmReady ? 'LiteRT.js WASM ready' : 'Loading LiteRT.js'}</strong>
          </div>
          <div>
            <span>Accelerator</span>
            <strong>{isReady ? classifier.accelerator.toUpperCase() : 'Pending'}</strong>
          </div>
          <div>
            <span>Latency</span>
            <strong>{verdict ? `${Math.round(verdict.latencyMs)} ms` : 'Waiting'}</strong>
          </div>
          <div>
            <span>Local only</span>
            <strong>100% browser inference</strong>
          </div>
        </div>

        {isReady && classifier.fallbackMessage ? <p className="notice">{classifier.fallbackMessage}</p> : null}
        {classifier.kind === 'missing' ? <p className="notice warning">{classifier.message}</p> : null}
        {classifier.kind === 'error' ? <p className="notice warning">LiteRT could not start: {classifier.message}</p> : null}
        {cameraError ? <p className="notice warning">{cameraError}</p> : null}

        <div className="actions">
          <button type="button" onClick={() => void startCamera()} disabled={!isReady}>
            {isCameraActive ? 'Restart camera' : 'Start camera'}
          </button>
          {isCameraActive ? (
            <button type="button" className="secondary" onClick={stopCamera}>
              Stop camera
            </button>
          ) : null}
        </div>

        <label
          className={`dropZone ${isDragging ? 'dragging' : ''}`}
          onDragEnter={(event) => {
            event.preventDefault()
            setIsDragging(true)
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
        >
          <input type="file" accept="image/*" onChange={onFileChange} disabled={!isReady} />
          <span>Drop an image or tap to upload</span>
          <strong>{fileName ?? 'File mode works without camera permission'}</strong>
        </label>
      </section>
    </main>
  )
}

export default App
