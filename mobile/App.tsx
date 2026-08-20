import { useCallback, useMemo, useState } from 'react';
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  Camera,
  runAtTargetFps,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { NitroModules } from 'react-native-nitro-modules';
import { useRunOnJS } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import {
  bundledMetadata,
  bundledModelAsset,
} from './src/generated/modelAssets';

type Normalization = {
  mode: 'mobilenet_v2';
  scale: number;
  offset: number;
};

type ModelMetadata = {
  model: string;
  inputShape: [number, number, number, number];
  inputDtype: 'float32';
  normalization: Normalization;
  classOrder: ['not_hotdog', 'hotdog'] | ['hotdog', 'not_hotdog'];
  threshold: number;
  gitSha?: string;
  metrics?: {
    accuracy: number;
    hotdogPrecision: number;
    hotdogRecall: number;
  };
};

type ModelStatus =
  | { kind: 'missing'; message: string }
  | { kind: 'error'; message: string }
  | { kind: 'ready'; metadata: ModelMetadata; modelAsset: number };

type Verdict = {
  rawProbability: number;
  smoothProbability: number;
  latencyMs: number;
};

const smoothing = 0.72;
const targetFps = 10;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const parseNumber = (value: unknown, label: string) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new Error(`metadata.${label} must be a number`);
  }

  return value;
};

const parseMetadata = (value: unknown): ModelMetadata => {
  if (!isRecord(value)) {
    throw new Error('metadata.json must contain an object');
  }

  const normalization = value.normalization;
  const metrics = value.metrics;

  if (typeof value.model !== 'string') {
    throw new Error('metadata.model must be a string');
  }

  if (
    !Array.isArray(value.inputShape) ||
    value.inputShape.length !== 4 ||
    !value.inputShape.every(item => typeof item === 'number')
  ) {
    throw new Error('metadata.inputShape must be [1, height, width, channels]');
  }

  if (value.inputShape[3] !== 3) {
    throw new Error(`Expected 3 input channels, got ${value.inputShape[3]}`);
  }

  if (value.inputDtype !== 'float32') {
    throw new Error('Only float32 model input is supported');
  }

  if (!isRecord(normalization) || normalization.mode !== 'mobilenet_v2') {
    throw new Error('Only mobilenet_v2 normalization is supported');
  }

  if (
    !Array.isArray(value.classOrder) ||
    value.classOrder.length !== 2 ||
    !value.classOrder.includes('hotdog') ||
    !value.classOrder.includes('not_hotdog')
  ) {
    throw new Error('metadata.classOrder must include hotdog and not_hotdog');
  }

  const parsedMetrics = isRecord(metrics)
    ? {
        accuracy: parseNumber(metrics.accuracy, 'metrics.accuracy'),
        hotdogPrecision: parseNumber(
          metrics.hotdogPrecision,
          'metrics.hotdogPrecision',
        ),
        hotdogRecall: parseNumber(metrics.hotdogRecall, 'metrics.hotdogRecall'),
      }
    : undefined;

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
  };
};

const resolveModelStatus = (): ModelStatus => {
  if (bundledMetadata === null || bundledModelAsset === null) {
    return {
      kind: 'missing',
      message:
        'Model files are not here yet. Drop models/metadata.json and models/not-hotdog.tflite, then rebuild.',
    };
  }

  try {
    return {
      kind: 'ready',
      metadata: parseMetadata(bundledMetadata),
      modelAsset: bundledModelAsset,
    };
  } catch (error) {
    return {
      kind: 'error',
      message: error instanceof Error ? error.message : 'Invalid metadata.json',
    };
  }
};

const formatPercent = (value: number) => `${Math.round(value * 100)}%`;

const clampProbability = (value: number) => Math.min(1, Math.max(0, value));

const App = () => {
  const modelStatus = useMemo(resolveModelStatus, []);
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);

  const onResult = useCallback((rawProbability: number, latencyMs: number) => {
    setVerdict(previous => ({
      rawProbability,
      smoothProbability: previous
        ? previous.smoothProbability * smoothing + rawProbability * (1 - smoothing)
        : rawProbability,
      latencyMs,
    }));
  }, []);

  const onFrameError = useCallback((message: string) => {
    setCameraError(message);
  }, []);

  const threshold =
    modelStatus.kind === 'ready' ? modelStatus.metadata.threshold : 1;
  const isReady = modelStatus.kind === 'ready';
  const isHotdog = verdict ? verdict.smoothProbability >= threshold : false;
  const heroText = !isReady
    ? modelStatus.kind === 'missing'
      ? 'MODEL MISSING'
      : 'MODEL ERROR'
    : verdict
      ? isHotdog
        ? 'HOTDOG'
        : 'NOT HOTDOG'
      : 'HOTDOG?';
  const heroStyle = verdict
    ? isHotdog
      ? styles.hotdogPanel
      : styles.notHotdogPanel
    : styles.idlePanel;

  return (
    <View style={styles.appShell}>
      <StatusBar barStyle="light-content" />
      <View style={[styles.verdictPanel, heroStyle]}>
        {isReady ? (
          <LiveClassifier
            isActive={isCameraActive}
            metadata={modelStatus.metadata}
            modelAsset={modelStatus.modelAsset}
            onFrameError={onFrameError}
            onResult={onResult}
          />
        ) : null}
        <View style={styles.noise} />
        <View style={styles.verdictContent}>
          <Text style={styles.eyebrow}>Not Hotdog</Text>
          <Text style={styles.hero}>{heroText}</Text>
          <Text style={styles.confidence}>
            {verdict
              ? `${formatPercent(
                  verdict.smoothProbability,
                )} confidence · raw ${formatPercent(verdict.rawProbability)}`
              : isReady
                ? `Threshold ${formatPercent(
                    threshold,
                  )}. Bring on the tube steak.`
                : modelStatus.message}
          </Text>
        </View>
      </View>

      <SafeAreaView style={styles.controlDeck}>
        <View style={styles.statusGrid}>
          <StatusCard label="Runtime" value="LiteRT on-device" />
          <StatusCard
            label="Model"
            value={isReady ? modelStatus.metadata.model : 'Pending'}
          />
          <StatusCard
            label="Latency"
            value={verdict ? `${Math.round(verdict.latencyMs)} ms` : 'Waiting'}
          />
          <StatusCard label="Local only" value="No uploads. Very smug." />
        </View>

        {modelStatus.kind === 'missing' ? (
          <Notice warning message={modelStatus.message} />
        ) : null}
        {modelStatus.kind === 'error' ? (
          <Notice warning message={`LiteRT could not start: ${modelStatus.message}`} />
        ) : null}
        {cameraError ? <Notice warning message={cameraError} /> : null}

        <View style={styles.actions}>
          <TouchableOpacity
            activeOpacity={0.85}
            disabled={!isReady}
            onPress={() => {
              setCameraError(null);
              setIsCameraActive(true);
            }}
            style={[styles.button, !isReady && styles.buttonDisabled]}>
            <Text style={styles.buttonText}>
              {isCameraActive ? 'Restart camera' : 'Start camera'}
            </Text>
          </TouchableOpacity>
          {isCameraActive ? (
            <TouchableOpacity
              activeOpacity={0.85}
              onPress={() => setIsCameraActive(false)}
              style={[styles.button, styles.secondaryButton]}>
              <Text style={[styles.buttonText, styles.secondaryButtonText]}>
                Stop camera
              </Text>
            </TouchableOpacity>
          ) : null}
        </View>
      </SafeAreaView>
    </View>
  );
};

const LiveClassifier = ({
  isActive,
  metadata,
  modelAsset,
  onFrameError,
  onResult,
}: {
  isActive: boolean;
  metadata: ModelMetadata;
  modelAsset: number;
  onFrameError: (message: string) => void;
  onResult: (rawProbability: number, latencyMs: number) => void;
}) => {
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const tflite = useTensorflowModel(modelAsset, []);
  const model = tflite.state === 'loaded' ? tflite.model : undefined;
  const boxedModel = useMemo(
    () => (model ? NitroModules.box(model) : undefined),
    [model],
  );
  const { resize } = useResizePlugin();
  const runOnResult = useRunOnJS(onResult, [onResult]);
  const runOnFrameError = useRunOnJS(onFrameError, [onFrameError]);
  const [, inputHeight, inputWidth, channels] = metadata.inputShape;
  const hotdogIndex = metadata.classOrder.indexOf('hotdog');
  const normalizationOffset = metadata.normalization.offset;
  const normalizationScale = metadata.normalization.scale;

  const frameProcessor = useFrameProcessor(
    frame => {
      'worklet';

      if (!boxedModel || channels !== 3) {
        return;
      }

      runAtTargetFps(targetFps, () => {
        'worklet';

        const startedAt = Date.now();

        try {
          const pixels = resize(frame, {
            scale: {
              width: inputWidth,
              height: inputHeight,
            },
            pixelFormat: 'rgb',
            dataType: 'float32',
          });

          for (let index = 0; index < pixels.length; index += 1) {
            pixels[index] = pixels[index] * normalizationScale + normalizationOffset;
          }

          const inputBuffer = pixels.buffer.slice(
            pixels.byteOffset,
            pixels.byteOffset + pixels.byteLength,
          ) as ArrayBuffer;
          const outputBuffers = boxedModel.unbox().runSync([inputBuffer]);
          const output = new Float32Array(outputBuffers[0]);
          const probability =
            output.length > 1 ? output[hotdogIndex] : output[0] ?? 0;
          const latencyMs = Date.now() - startedAt;

          runOnResult(clampProbability(Number(probability)), latencyMs);
        } catch (error) {
          runOnFrameError(
            error instanceof Error ? error.message : 'Camera inference failed',
          );
        }
      });
    },
    [
      boxedModel,
      channels,
      hotdogIndex,
      inputHeight,
      inputWidth,
      normalizationOffset,
      normalizationScale,
      resize,
      runOnFrameError,
      runOnResult,
    ],
  );

  const requestCamera = useCallback(() => {
    requestPermission()
      .then(granted => {
        if (!granted) {
          onFrameError('Camera permission was denied.');
        }
      })
      .catch(() => onFrameError('Camera permission request failed.'));
  }, [onFrameError, requestPermission]);

  if (!hasPermission) {
    return (
      <View style={styles.cameraFallback}>
        <Text style={styles.noticeTitle}>Camera permission needed</Text>
        <Text style={styles.noticeText}>
          The app classifies frames locally, but it still needs to see the meat.
        </Text>
        <TouchableOpacity
          activeOpacity={0.85}
          onPress={requestCamera}
          style={styles.button}>
          <Text style={styles.buttonText}>Allow camera</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.cameraFallback}>
        <Text style={styles.noticeTitle}>No back camera found</Text>
        <Text style={styles.noticeText}>
          Very bold for a camera app, but here we are.
        </Text>
      </View>
    );
  }

  if (tflite.state === 'error') {
    return (
      <View style={styles.cameraFallback}>
        <Text style={styles.noticeTitle}>LiteRT could not load</Text>
        <Text style={styles.noticeText}>{tflite.error.message}</Text>
      </View>
    );
  }

  return (
    <Camera
      device={device}
      frameProcessor={frameProcessor}
      isActive={isActive && tflite.state === 'loaded'}
      pixelFormat="rgb"
      preview
      resizeMode="cover"
      style={styles.cameraPreview}
    />
  );
};

const StatusCard = ({ label, value }: { label: string; value: string }) => (
  <View style={styles.statusCard}>
    <Text style={styles.statusLabel}>{label}</Text>
    <Text style={styles.statusValue}>{value}</Text>
  </View>
);

const Notice = ({ message, warning }: { message: string; warning?: boolean }) => (
  <View style={[styles.notice, warning && styles.warningNotice]}>
    <Text style={styles.noticeText}>{message}</Text>
  </View>
);

const styles = StyleSheet.create({
  actions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginTop: 16,
  },
  appShell: {
    backgroundColor: '#07080d',
    flex: 1,
  },
  button: {
    backgroundColor: '#22c55e',
    borderRadius: 999,
    paddingHorizontal: 20,
    paddingVertical: 14,
  },
  buttonDisabled: {
    backgroundColor: '#1e293b',
  },
  buttonText: {
    color: '#07130a',
    fontSize: 16,
    fontWeight: '900',
  },
  cameraFallback: {
    alignItems: 'center',
    backgroundColor: 'rgba(2, 6, 23, 0.5)',
    bottom: 0,
    gap: 12,
    justifyContent: 'center',
    left: 0,
    padding: 24,
    position: 'absolute',
    right: 0,
    top: 0,
  },
  cameraPreview: {
    bottom: 0,
    left: 0,
    opacity: 0.32,
    position: 'absolute',
    right: 0,
    top: 0,
  },
  confidence: {
    alignSelf: 'center',
    backgroundColor: 'rgba(2, 6, 23, 0.35)',
    borderColor: 'rgba(255, 255, 255, 0.24)',
    borderRadius: 999,
    borderWidth: 1,
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 16,
    fontWeight: '800',
    marginTop: 18,
    overflow: 'hidden',
    paddingHorizontal: 16,
    paddingVertical: 12,
    textAlign: 'center',
  },
  controlDeck: {
    backgroundColor: '#09090b',
    padding: 14,
  },
  eyebrow: {
    color: 'rgba(255, 255, 255, 0.74)',
    fontSize: 14,
    fontWeight: '800',
    letterSpacing: 6,
    marginBottom: 12,
    textTransform: 'uppercase',
  },
  hero: {
    color: '#fff',
    fontSize: 76,
    fontWeight: '900',
    letterSpacing: -6,
    lineHeight: 72,
    textAlign: 'center',
    textTransform: 'uppercase',
  },
  hotdogPanel: {
    backgroundColor: '#02853d',
  },
  idlePanel: {
    backgroundColor: '#111827',
  },
  noise: {
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderWidth: 1,
    bottom: 0,
    left: 0,
    opacity: 0.18,
    position: 'absolute',
    right: 0,
    top: 0,
  },
  notHotdogPanel: {
    backgroundColor: '#8f0817',
  },
  notice: {
    backgroundColor: 'rgba(15, 23, 42, 0.74)',
    borderColor: 'rgba(148, 163, 184, 0.18)',
    borderRadius: 20,
    borderWidth: 1,
    marginTop: 14,
    padding: 14,
  },
  noticeText: {
    color: '#dbeafe',
    fontSize: 15,
    fontWeight: '700',
    textAlign: 'center',
  },
  noticeTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '900',
    textAlign: 'center',
  },
  secondaryButton: {
    backgroundColor: 'rgba(51, 65, 85, 0.92)',
  },
  secondaryButtonText: {
    color: '#e2e8f0',
  },
  statusCard: {
    backgroundColor: 'rgba(15, 23, 42, 0.74)',
    borderColor: 'rgba(148, 163, 184, 0.18)',
    borderRadius: 20,
    borderWidth: 1,
    flex: 1,
    minWidth: '47%',
    padding: 14,
  },
  statusGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  statusLabel: {
    color: '#94a3b8',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 1.8,
    textTransform: 'uppercase',
  },
  statusValue: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '800',
    marginTop: 6,
  },
  verdictContent: {
    alignItems: 'center',
    padding: 20,
  },
  verdictPanel: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
    overflow: 'hidden',
  },
  warningNotice: {
    backgroundColor: 'rgba(127, 29, 29, 0.54)',
    borderColor: 'rgba(248, 113, 113, 0.38)',
  },
});

export default App;
