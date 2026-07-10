import type { Channel, DashboardRuntime, Tone } from '../../types';
import type { AnalysisCardMock, ChannelDashboardTelemetry } from '../../mock/dashboardMock';

function finiteNumber(value: unknown): value is number {
  return Number.isFinite(value);
}

function clippedLevelTone(level: number, fallback: Tone): Tone {
  if (level >= -3) return 'red';
  if (level >= -10) return 'yellow';
  if (level <= -35) return 'blue';
  return fallback;
}

function safeStateForLevel(level: number): string {
  if (level >= -3) return 'CLIP';
  if (level >= -10) return 'HOT';
  if (level <= -35) return 'LOW';
  return 'OK';
}

function tracePoint(value: number): number {
  return Math.max(8, Math.min(88, value));
}

function fallbackSpectrum(values: number[]): number[] {
  return values.map((value, index) => tracePoint(value + (index % 3) * 4));
}

function backendTone(label: string): Tone {
  return label.toLowerCase().includes('backend') ? 'green' : 'blue';
}

function backendLabel(label: string): string {
  return label.toLowerCase().includes('backend') ? 'Backend DTO' : label;
}

export function dashboardSpectrumTrace(
  runtime: Pick<DashboardRuntime, 'sourceLabel' | 'spectrumBins' | 'spectrumAvailable' | 'audioCaptureRunning'>,
  fallback: number[]
): { values: number[]; sourceLabel: string; sourceTone: Tone; caption: string } {
  const runtimeBins = Array.isArray(runtime.spectrumBins)
    ? runtime.spectrumBins.filter(value => Number.isFinite(value))
    : [];

  if (runtime.spectrumAvailable && runtimeBins.length >= 4) {
    const backendSource = runtime.sourceLabel.toLowerCase().includes('backend');
    const hasSignal = runtimeBins.some(value => value > 1);
    return {
      values: runtimeBins.map(tracePoint),
      sourceLabel: backendSource ? (hasSignal ? 'Real audio' : 'No signal') : runtime.sourceLabel,
      sourceTone: backendSource ? (hasSignal ? 'green' : 'muted') : 'blue',
      caption: hasSignal ? 'Read-only analyzer trace from dashboard DTO' : 'Analyzer connected; no measurable signal'
    };
  }

  if (runtime.sourceLabel.toLowerCase().includes('backend')) {
    return {
      values: [],
      sourceLabel: runtime.audioCaptureRunning ? 'No signal' : 'Audio offline',
      sourceTone: 'muted',
      caption: runtime.audioCaptureRunning
        ? 'Analyzer connected; no measurable signal'
        : 'Start read-only audio meters to see spectrum'
    };
  }

  return {
    values: fallbackSpectrum(fallback),
    sourceLabel: 'Display-only',
    sourceTone: 'yellow',
    caption: 'Display-only analyzer trace'
  };
}

export function dashboardAnalysisCards(
  runtime: Pick<DashboardRuntime, 'sourceLabel' | 'analysisTraces'>,
  fallbackCards: AnalysisCardMock[]
): AnalysisCardMock[] {
  return fallbackCards.map(card => {
    const runtimeTrace = runtime.analysisTraces?.[card.id as keyof NonNullable<DashboardRuntime['analysisTraces']>];
    const runtimeValues = Array.isArray(runtimeTrace?.values)
      ? runtimeTrace.values.filter(value => Number.isFinite(value)).map(tracePoint)
      : [];

    if (!runtimeValues.length && runtime.sourceLabel.toLowerCase().includes('backend')) {
      return {
        ...card,
        values: [],
        primary: 'Not measured',
        secondary: 'Waiting for analyzer data',
        sourceLabel: 'Unavailable',
        sourceTone: 'muted'
      };
    }

    if (!runtimeValues.length) {
      return card;
    }

    const sourceLabel = backendLabel(runtimeTrace?.sourceLabel ?? runtime.sourceLabel);
    return {
      ...card,
      values: runtimeValues,
      primary: runtimeTrace?.primary || card.primary,
      secondary: runtimeTrace?.secondary || card.secondary,
      sourceLabel,
      sourceTone: runtimeTrace?.sourceTone ?? backendTone(sourceLabel)
    };
  });
}

export function dashboardTelemetryForChannel(
  channel: Channel,
  fallback: ChannelDashboardTelemetry,
  options: { suppressMockSolo?: boolean } = {}
): ChannelDashboardTelemetry {
  const hasRuntimeLevel = finiteNumber(channel.level);
  const backendChannel = channel.dataSource === 'backend';
  const meterAvailable = channel.meterAvailable !== false && hasRuntimeLevel;
  const level = meterAvailable ? channel.level : backendChannel ? -120 : fallback.meterL;
  const muted = channel.status === 'MUTED';
  const inactive = channel.status === 'INACTIVE';

  return {
    ...fallback,
    waveform: backendChannel && !meterAvailable ? [] : fallback.waveform,
    meterL: level,
    meterR: level,
    gateLabel: channel.gateAvailable === false ? 'G —' : channel.gateLabel || (backendChannel ? 'G —' : fallback.gateLabel),
    compLabel: channel.compAvailable === false ? 'C —' : channel.compLabel || (backendChannel ? 'C —' : fallback.compLabel),
    pan: channel.panAvailable === false ? 0 : finiteNumber(channel.pan) ? channel.pan : backendChannel ? 0 : fallback.pan,
    muted,
    solo: options.suppressMockSolo ? false : fallback.solo,
    safeState: muted ? 'MUTE' : !meterAvailable ? 'NO METER' : inactive ? 'NO SIGNAL' : safeStateForLevel(level),
    accent: muted
      ? 'red'
      : inactive
        ? 'muted'
        : clippedLevelTone(level, fallback.accent)
  };
}
