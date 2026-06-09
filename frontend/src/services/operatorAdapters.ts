import type {
  Agent,
  AutomixerMode,
  BackendUiState,
  BusRecord,
  Channel,
  Communication,
  DashboardRuntime,
  DashboardRuntimeTrace,
  DecisionTraceRecord,
  DeviceRecord,
  DirectorRuntime,
  ChannelIntelligenceRecord,
  LogRecord,
  MemoryRecord,
  Metric,
  ModuleStatusRecord,
  PerformanceRecord,
  ReplayRunRecord,
  RoutingSegment,
  RuleLibrarySnapshotRow,
  SafetyEvidenceItem,
  SafetyMonitorState,
  Task,
  TelemetryAlertsSnapshot,
  Tone,
  TrainingSessionSnapshotRow,
  TopologyLink,
  TopologyNode
} from '../types';

export type BackendOperatorMode = 'analyze' | 'assist' | 'supervised' | 'auto';

type UnknownRecord = Record<string, unknown>;

const modeToUi: Record<BackendOperatorMode, AutomixerMode> = {
  analyze: 'Analyze',
  assist: 'Assist',
  supervised: 'Supervised',
  auto: 'Auto'
};

const MODE_ALIASES_TO_CANONICAL_BACKEND: Record<string, BackendOperatorMode> = {
  analyze: 'analyze',
  'analysis': 'analyze',
  'analysis mode': 'analyze',
  'analysis_mode': 'analyze',
  'analysis_mode_mode': 'analyze',
  'analysismode': 'analyze',
  assist: 'assist',
  'assisted': 'assist',
  'assisted mixing': 'assist',
  'assistedmixing': 'assist',
  'assist mode': 'assist',
  'assisted mode': 'assist',
  'assisted_mixing': 'assist',
  'assisted_mixing_mode': 'assist',
  supervised: 'supervised',
  'supervised automix': 'supervised',
  'supervised automix mode': 'supervised',
  supervised_auto_mix: 'supervised',
  'supervised auto mix': 'supervised',
  'supervised_automix': 'supervised',
  'supervised_automix_mode': 'supervised',
  'supervised_auto_mode': 'supervised',
  'supervised mode': 'supervised',
  'supervised_mode': 'supervised',
  auto: 'auto',
  autonomous: 'auto',
  'full autonomous': 'auto',
  'full autonomous mix': 'auto',
  'full autonomous mix mode': 'auto',
  'full autonomous mixing': 'auto',
  'full autonomous mode': 'auto',
  'full autonomous mode mode': 'auto',
  'full_autonomous': 'auto',
  'full_autonomous_mix': 'auto',
  'full_autonomous_mix_mode': 'auto',
  'full_autonomous_mixing': 'auto',
  'full autonomous mixing mode': 'auto',
  'full_autonomous_mixing_mode': 'auto',
  full_autonomous_mode: 'auto',
  'auto mode': 'auto',
  'auto_mode': 'auto'
};

const modeToBackend: Record<AutomixerMode, BackendOperatorMode> = {
  Analyze: 'analyze',
  Assist: 'assist',
  Supervised: 'supervised',
  Auto: 'auto'
};

function normalizeModeInput(value: unknown): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[-\s]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');
}

export const mockBackendState: BackendUiState = {
  connectionState: 'mock',
  dataState: 'mock_fallback',
  dataStateLabel: 'Mock fallback',
  dataStateTone: 'blue',
  hasRuntimeData: false,
  isPartialData: false,
  staleSnapshot: false,
  sourceLabel: 'Mock',
  connectionLabel: 'Demo data',
  statusLabel: 'Offline',
  statusTone: 'blue',
  canCreateProposals: false,
  canApplyToConsole: false,
  requiresApproval: true,
  liveWritePolicy: 'mock_frontend_only',
  applyPath: 'none',
  writeScope: [],
  lastUpdated: 'mock',
  message: 'Backend is not connected. UI is using local prototype data.',
  proposalApplyResult: null
};

export function toBackendMode(mode: AutomixerMode): BackendOperatorMode {
  return modeToBackend[mode];
}

export function toUiMode(value: unknown): AutomixerMode {
  const normalized = normalizeModeInput(value);
  const canonicalMode = MODE_ALIASES_TO_CANONICAL_BACKEND[normalized] as BackendOperatorMode | undefined;
  return modeToUi[canonicalMode ?? 'assist'];
}

function asRecord(value: unknown): UnknownRecord {
  return value && typeof value === 'object' ? value as UnknownRecord : {};
}

function asArray(value: unknown): UnknownRecord[] {
  return Array.isArray(value) ? value.filter(item => item && typeof item === 'object') as UnknownRecord[] : [];
}

function num(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function boolish(value: unknown): boolean | undefined {
  if (value === null || value === undefined || value === '') return undefined;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;

  const normalized = String(value).trim().toLowerCase();
  if (['1', 'true', 'yes', 'on', 'muted'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off', 'unmuted'].includes(normalized)) return false;
  return undefined;
}

function metricNum(value: unknown, fallback: number): number {
  if (value === null || value === undefined || value === '') return fallback;
  return num(value, fallback);
}

function str(value: unknown, fallback = ''): string {
  if (value === null || value === undefined) return fallback;
  const next = String(value).trim();
  return next || fallback;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map(item => String(item)).filter(Boolean)
    : [];
}

function timeLabel(value: unknown, fallback = 'now'): string {
  const raw = str(value);
  if (!raw) return fallback;
  const match = raw.match(/T(\d{2}:\d{2})/);
  return match?.[1] ?? raw.slice(0, 5);
}

function toneForBackendStatus(status: string): Tone {
  const normalized = status.toLowerCase();
  if (normalized.includes('blocked') || normalized.includes('error') || normalized.includes('danger')) return 'red';
  if (normalized.includes('warn') || normalized.includes('standby') || normalized.includes('idle')) return 'yellow';
  if (normalized.includes('audio') || normalized.includes('osc') || normalized.includes('connected')) return 'blue';
  return 'green';
}

function priorityFromSeverity(severity: unknown): Task['priority'] {
  const normalized = str(severity, 'medium').toLowerCase();
  if (normalized === 'high' || normalized === 'critical') return 'High';
  if (normalized === 'low') return 'Low';
  return 'Medium';
}

function taskStatusFromBackend(status: unknown): Task['status'] {
  const normalized = str(status, 'pending').toLowerCase();
  if (['applied', 'accepted', 'done', 'approved'].includes(normalized)) return 'Applied';
  if (['dismissed', 'rejected'].includes(normalized)) return 'Rejected';
  if (normalized.includes('blocked') || normalized.includes('review')) return 'Review';
  return 'Pending';
}

function toneForSafetyEvidence(label: string, value: string): Tone {
  const normalized = `${label} ${value}`.toLowerCase();
  if (normalized.includes('blocked') || normalized.includes('danger') || normalized.includes('fail')) return 'red';
  if (
    normalized.includes('missing')
    || normalized.includes('stale')
    || normalized.includes('review')
    || normalized.includes('required')
    || normalized.includes('operator_review_only')
  ) {
    return 'yellow';
  }
  if (
    normalized.includes('clear')
    || normalized.includes('ready')
    || normalized.includes('active')
    || normalized.includes('coherent')
    || normalized.includes('low')
    || normalized.includes('not_restricted')
  ) {
    return 'green';
  }
  return 'blue';
}

function adaptSafetyEvidence(value: unknown): SafetyEvidenceItem[] | undefined {
  const rows = asArray(value);
  if (!rows.length) return undefined;

  return rows.map((row, index) => {
    const label = str(row.label, `Evidence ${index + 1}`);
    const rawValue = str(row.value, str(row.reason, 'review'));
    const target = str(row.target);
    const displayValue = target && rawValue ? `${rawValue}/${target}` : rawValue || 'review';
    return {
      label,
      value: displayValue.replace(/_/g, ' '),
      tone: toneForSafetyEvidence(label, displayValue)
    };
  });
}

function cleanToken(value: unknown, fallback = ''): string {
  return str(value, fallback).replace(/_/g, ' ');
}

function agentStatusFromBackend(status: unknown): Agent['status'] {
  const normalized = str(status, 'STANDBY').toUpperCase();
  if (normalized === 'ACTIVE') return 'ACTIVE';
  if (normalized === 'BLOCKED') return 'BLOCKED';
  if (normalized === 'UNAVAILABLE') return 'UNAVAILABLE';
  if (normalized === 'WARNING') return 'WARNING';
  return 'STANDBY';
}

function toneFromLogLevel(level: LogRecord['level']): Tone {
  if (level === 'ERROR') return 'red';
  if (level === 'WARN') return 'yellow';
  if (level === 'SUCCESS') return 'green';
  return 'blue';
}

function communicationStatusFromLogLevel(level: LogRecord['level']): string {
  if (level === 'ERROR') return 'Blocked';
  if (level === 'WARN') return 'Review';
  if (level === 'SUCCESS') return 'Approved';
  return 'Info';
}

function avg(values: number[], fallback: number): number {
  const usable = values.filter(value => Number.isFinite(value));
  if (!usable.length) return fallback;
  return usable.reduce((sum, value) => sum + value, 0) / usable.length;
}

function numericList(value: unknown, limit = 64): number[] {
  if (!Array.isArray(value)) return [];
  return value
    .map(item => Number(item))
    .filter(item => Number.isFinite(item))
    .slice(0, limit);
}

function firstNumericList(...values: unknown[]): number[] | undefined {
  for (const value of values) {
    const rows = numericList(value);
    if (rows.length) return rows;
  }
  return undefined;
}

type DashboardTraceKind = 'phase' | 'dynamics' | 'loudness' | 'generic';

function traceDisplayValue(value: number, kind: DashboardTraceKind): number {
  if (kind === 'phase' && value >= -1 && value <= 1) {
    return 50 + value * 38;
  }
  if (kind === 'loudness' && value <= 0) {
    return ((value + 60) / 60) * 100;
  }
  return value;
}

function boundedTrace(values: number[], kind: DashboardTraceKind): number[] {
  return values.map(value => {
    const displayValue = traceDisplayValue(value, kind);
    return Number(Math.max(8, Math.min(88, displayValue)).toFixed(1));
  });
}

function optionalText(...values: unknown[]): string | undefined {
  for (const value of values) {
    const next = str(value);
    if (next) return next;
  }
  return undefined;
}

function dashboardRuntimeTrace(
  values: number[] | undefined,
  kind: DashboardTraceKind,
  primary?: string,
  secondary?: string
): DashboardRuntimeTrace | undefined {
  if (!values?.length) return undefined;

  return {
    values: boundedTrace(values, kind),
    primary,
    secondary,
    sourceLabel: 'Backend analyzer DTO',
    sourceTone: 'green'
  };
}

function adaptDashboardAnalysisTraces(
  analyzer: UnknownRecord,
  master: UnknownRecord
): NonNullable<DashboardRuntime['analysisTraces']> {
  const traces: NonNullable<DashboardRuntime['analysisTraces']> = {};

  const phase = dashboardRuntimeTrace(
    firstNumericList(analyzer.phase_history, analyzer.phaseHistory, analyzer.phase_scope, analyzer.phaseScope),
    'phase',
    optionalText(analyzer.phase_label, analyzer.phase_status),
    optionalText(analyzer.phase_summary, analyzer.phase_source)
  );
  if (phase) traces.phase = phase;

  const correlation = dashboardRuntimeTrace(
    firstNumericList(analyzer.correlation_history, analyzer.correlationHistory, analyzer.correlation_trace),
    'phase',
    optionalText(analyzer.correlation_label, analyzer.correlation_status),
    optionalText(analyzer.correlation_summary, analyzer.mono_status)
  );
  if (correlation) traces.correlation = correlation;

  const dynamics = dashboardRuntimeTrace(
    firstNumericList(analyzer.dynamics_history, analyzer.dynamicsHistory, analyzer.gain_reduction_history, analyzer.gr_history),
    'dynamics',
    optionalText(analyzer.dynamics_label, analyzer.gain_reduction_label),
    optionalText(analyzer.dynamics_summary, analyzer.transient_status)
  );
  if (dynamics) traces.dynamics = dynamics;

  const loudness = dashboardRuntimeTrace(
    firstNumericList(
      analyzer.lufs_history,
      analyzer.loudness_history,
      analyzer.integrated_lufs_history,
      analyzer.short_term_lufs_history,
      master.lufs_history,
      master.short_term_lufs_history
    ),
    'loudness',
    optionalText(analyzer.lufs_label, analyzer.loudness_label),
    optionalText(analyzer.lufs_summary, analyzer.loudness_summary)
  );
  if (loudness) traces.loudness = loudness;

  return traces;
}

function busTone(group: string): Tone {
  const normalized = group.toUpperCase();
  if (normalized.includes('VOCAL')) return 'purple';
  if (normalized.includes('GUITAR')) return 'blue';
  if (normalized.includes('BASS')) return 'yellow';
  if (normalized.includes('FX') || normalized.includes('RETURN')) return 'blue';
  if (normalized.includes('UNKNOWN') || normalized.includes('UNASSIGNED')) return 'muted';
  return 'green';
}

function toneValue(value: unknown, fallback: Tone = 'blue'): Tone {
  const normalized = str(value, fallback).toLowerCase();
  if (['green', 'blue', 'purple', 'yellow', 'red', 'muted'].includes(normalized)) {
    return normalized as Tone;
  }
  return fallback;
}

function toneForModuleStatus(status: string, canWriteLive: boolean): Tone {
  const normalized = status.toLowerCase();
  if (normalized === 'canonical_supervised') return canWriteLive ? 'yellow' : 'red';
  if (normalized === 'hardware_proven_narrow') return canWriteLive ? 'yellow' : 'blue';
  if (normalized === 'dry_run_ready') return 'blue';
  if (normalized === 'canonical' && canWriteLive) return 'green';
  if (normalized === 'canonical') return 'blue';
  if (normalized === 'proposal_only' || normalized === 'shadow_only' || normalized === 'offline_only') return 'yellow';
  if (normalized === 'lab' || normalized === 'lab_only') return 'purple';
  if (normalized.includes('blocked') || normalized.includes('quarantined')) return 'red';
  if (normalized === 'placeholder') return 'red';
  if (normalized === 'deprecated') return 'muted';
  return canWriteLive ? 'yellow' : 'blue';
}

function toneForModeGateStatus(status: string): Tone {
  const normalized = status.toLowerCase();
  if (normalized === 'available') return 'green';
  if (normalized === 'limited_available') return 'yellow';
  if (normalized.includes('blocked')) return 'red';
  return 'blue';
}

function toneForAutonomyEvidenceOverlayStatus(status: string): Tone {
  const normalized = status.toLowerCase();
  if (normalized === 'manual_safety_review_candidate') return 'yellow';
  if (normalized === 'remediation_required') return 'red';
  if (normalized === 'dry_run_only') return 'blue';
  if (normalized === 'no_current_evidence') return 'yellow';
  return toneForModuleStatus(status, false);
}

function titleFromToken(value: string): string {
  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function toneForTelemetrySeverity(severity: string): Tone {
  const normalized = severity.toLowerCase();
  if (normalized === 'critical') return 'red';
  if (normalized === 'warning') return 'yellow';
  if (normalized === 'info') return 'blue';
  return 'muted';
}

function formatTelemetryNumber(value: unknown): string {
  if (value === null || value === undefined || value === '') return '';
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return str(value);
  return Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(2);
}

function channelPanPercent(value: unknown, fallback?: number): number | undefined {
  if (value === null || value === undefined || value === '') return fallback;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  if (numeric >= 0 && numeric <= 1) return Math.round((numeric - 0.5) * 100);
  if (numeric >= -100 && numeric <= 100) return Math.round(numeric);
  return fallback;
}

function channelControlLabel(prefix: string, value: unknown, fallback?: string): string | undefined {
  if (value === null || value === undefined || value === '') return fallback;
  const raw = str(value);
  if (!raw) return fallback;
  if (/^[A-Z]\s/i.test(raw)) return raw;

  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    return `${prefix} ${numeric.toFixed(1)}`;
  }

  return `${prefix} ${raw}`;
}

function telemetryTimeLabel(timestampMs: unknown, fallback: unknown): string {
  const numeric = Number(timestampMs);
  if (Number.isFinite(numeric)) return `${Math.trunc(numeric)} ms`;
  return timeLabel(fallback, 'n/a');
}

function busName(group: string): string {
  const normalized = group.trim();
  if (!normalized) return 'Unassigned Bus';
  if (normalized.toUpperCase().includes('BUS')) return normalized;
  return `${normalized
    .toLowerCase()
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')} Bus`;
}

function busRole(group: string): string {
  const normalized = group.toUpperCase();
  if (normalized.includes('FX') || normalized.includes('RETURN')) return 'Return';
  if (normalized.includes('MAIN') || normalized.includes('MASTER')) return 'Master';
  return 'Group';
}

function busStatus(channels: Channel[]): string {
  if (channels.some(channel => channel.status === 'ACTIVE')) return 'Active';
  if (channels.some(channel => channel.status === 'MUTED')) return 'Muted';
  return 'Inactive';
}

export function adaptBackendState(
  payload: unknown,
  connectionState: BackendUiState['connectionState'],
  message = ''
): BackendUiState {
  const data = asRecord(payload);
  const mode = asRecord(data.operator_mode ?? data);
  const capabilities = asRecord(mode.capabilities);
  const connection = asRecord(data.connection);
  const safety = asRecord(data.safety);
  const manual = asRecord(safety.manual_write_supervision);
  const connected = Boolean(connection.connected);
  const emergencyStopped = Boolean(manual.emergency_stopped ?? manual.reset_required);
  const liveWritePolicy = str(mode.live_write_policy, connected ? 'connected_read_only' : 'backend_read_only');
  const canApplyToConsole = Boolean(capabilities.can_apply_to_console) && !emergencyStopped;
  const canCreateProposals = Boolean(capabilities.can_create_proposals);
  const requiresApproval = capabilities.requires_approval !== false;
  const writeScope = Array.isArray(capabilities.allowed_live_write_kinds)
    ? capabilities.allowed_live_write_kinds.map(item => String(item))
    : [];

  return {
    connectionState,
    dataState: connectionState === 'connected' ? 'connected' : connectionState === 'connecting' ? 'connecting' : connectionState === 'error' ? 'error' : connectionState === 'offline' ? 'disconnected' : 'mock_fallback',
    dataStateLabel: connectionState === 'connected' ? 'Product Layer live' : connectionState === 'connecting' ? 'Connecting' : connectionState === 'error' ? 'Backend unavailable' : connectionState === 'offline' ? 'Disconnected' : 'Mock fallback',
    dataStateTone: connectionState === 'connected' ? 'green' : connectionState === 'connecting' ? 'yellow' : connectionState === 'error' ? 'red' : connectionState === 'offline' ? 'yellow' : 'blue',
    hasRuntimeData: connectionState === 'connected',
    isPartialData: false,
    staleSnapshot: false,
    sourceLabel: connectionState === 'connected' ? 'Backend' : mockBackendState.sourceLabel,
    connectionLabel: connected
      ? `${str(connection.connection_mode, 'mixer')} ${str(connection.ip, '')}`.trim()
      : connectionState === 'connected' ? 'Backend online' : mockBackendState.connectionLabel,
    statusLabel: emergencyStopped ? 'Emergency reset required' : connectionState === 'connected' ? 'Live data' : connectionState === 'connecting' ? 'Connecting' : 'Offline',
    statusTone: emergencyStopped ? 'red' : connectionState === 'connected' ? 'green' : connectionState === 'connecting' ? 'yellow' : 'blue',
    canCreateProposals,
    canApplyToConsole,
    requiresApproval,
    liveWritePolicy,
    applyPath: str(mode.apply_path, 'none'),
    writeScope,
    lastUpdated: str(data.generated_at, new Date().toISOString()),
    message: emergencyStopped
      ? 'Emergency stop is active. Supervised apply is blocked until explicit operator reset.'
      : message || str(data.reason, connectionState === 'connected' ? 'Product Layer connected.' : mockBackendState.message)
  };
}

export function adaptChannels(payload: unknown, fallback: Channel[]): Channel[] {
  const data = asRecord(payload);
  const rows = asArray(data.channels);
  if (!rows.length) return fallback;

  return rows.map((row, index) => {
    const base = fallback[index] ?? fallback[fallback.length - 1];
    const meter = asRecord(row.meter);
    const control = asRecord(row.control_state);
    const recognition = asRecord(row.ai_recognition);
    const id = Math.trunc(num(row.id ?? row.number, base?.id ?? index + 1));
    const status = str(row.status, 'unknown').toLowerCase();
    const controlMute = boolish(control.mute ?? control.muted ?? row.mute ?? row.muted);
    const confidence = num(recognition.confidence, base?.confidence ?? 72);
    const pan = channelPanPercent(control.pan ?? control.pan_percent ?? row.pan, base?.pan);
    const gateLabel = channelControlLabel(
      'G',
      control.gate_label ?? row.gate_label ?? control.gate_threshold_db ?? row.gate_threshold_db,
      base?.gateLabel
    );
    const compLabel = channelControlLabel(
      'C',
      control.comp_label
        ?? control.compressor_label
        ?? row.comp_label
        ?? row.compressor_label
        ?? control.compressor_gain_reduction_db
        ?? row.compressor_gain_reduction_db
        ?? control.compressor_threshold_db
        ?? row.compressor_threshold_db,
      base?.compLabel
    );
    const channelStatus: Channel['status'] = controlMute === true || status === 'muted'
      ? 'MUTED'
      : status === 'active'
        ? 'ACTIVE'
        : 'INACTIVE';
    return {
      id,
      name: str(row.name, base?.name ?? `Ch ${id}`),
      type: str(row.role, base?.type ?? 'Channel'),
      source: str(row.source, base?.source ?? `In ${String(id).padStart(2, '0')}`),
      level: num(row.level_db ?? meter.rms_db, base?.level ?? -60),
      lufs: num(row.lufs ?? meter.lufs, base?.lufs ?? -60),
      dynamic: num(row.dynamic_range_db ?? meter.dynamic_range_db, base?.dynamic ?? 0),
      status: channelStatus,
      recognition: `${str(recognition.label, str(row.role, base?.type ?? 'Channel'))}${confidence ? ` (${Math.round(confidence)}%)` : ''}`,
      group: str(row.group, base?.group ?? 'CHANNELS').toUpperCase(),
      fader: num(control.fader_db, base?.fader ?? -50),
      pan,
      gateLabel,
      compLabel,
      confidence
    };
  });
}

export function adaptModuleStatusRecords(payload: unknown): ModuleStatusRecord[] {
  const data = asRecord(payload);
  const rows = asArray(data.module_status_map);

  const moduleRows = rows.map(row => {
    const status = str(row.status, 'unknown');
    const canWriteLive = Boolean(row.can_write_live);
    return {
      moduleId: str(row.module_id, str(row.id, 'unknown_module')),
      label: str(row.label, str(row.module_id, 'Unknown module')),
      domain: str(row.domain, 'unknown'),
      status,
      entrypoint: str(row.entrypoint, str(row.module, 'none')),
      module: str(row.module, ''),
      uiSurface: str(row.ui_surface, 'Status-only surface'),
      canWriteLive,
      liveWriteKinds: Array.isArray(row.live_write_kinds)
        ? row.live_write_kinds.map(item => String(item)).filter(Boolean)
        : [],
      pipelineProfiles: Array.isArray(row.pipeline_profiles)
        ? row.pipeline_profiles.map(item => String(item)).filter(Boolean)
        : [],
      blockedReason: str(row.blocked_reason, ''),
      nextStep: str(row.next_step, 'No next step published.'),
      tone: toneForModuleStatus(status, canWriteLive)
    };
  });

  const inventory = asRecord(data.write_surface_inventory);
  const surfaces = asArray(inventory.surfaces);
  const surfaceRows = surfaces.map(row => {
    const surfaceId = str(row.id, 'unknown_surface');
    const status = str(row.status, 'unknown');
    const canWriteLive = Boolean(row.allowed_live);
    const paths = Array.isArray(row.paths)
      ? row.paths.map(item => String(item)).filter(Boolean)
      : [];
    const entrypoints = Array.isArray(row.entrypoints)
      ? row.entrypoints.map(item => String(item)).filter(Boolean)
      : [];
    return {
      moduleId: `write_surface:${surfaceId}`,
      label: str(row.label, surfaceId),
      domain: 'write_surface',
      status,
      entrypoint: entrypoints.length ? entrypoints.join(' / ') : 'none',
      module: paths.join(', '),
      uiSurface: str(row.gate, 'Write-surface inventory'),
      canWriteLive,
      liveWriteKinds: Array.isArray(row.write_kinds)
        ? row.write_kinds.map(item => String(item)).filter(Boolean)
        : [],
      pipelineProfiles: [status],
      blockedReason: canWriteLive ? '' : str(row.current_policy, 'No live authority.'),
      nextStep: str(row.next_action, 'Keep write-surface policy explicit.'),
      tone: toneForModuleStatus(status, canWriteLive)
    };
  });

  const coverage = asRecord(data.safety_coverage_matrix);
  const coverageRows = asArray(coverage.rows).map(row => {
    const rowId = str(row.id, 'unknown_coverage');
    const status = str(row.status, 'unknown');
    const canWriteLive = Boolean(row.live_allowed);
    return {
      moduleId: `safety_coverage:${rowId}`,
      label: str(row.label, rowId),
      domain: 'safety_coverage',
      status,
      entrypoint: str(row.canonical_gate, 'none'),
      module: str(row.action_type, 'unknown'),
      uiSurface: 'Safety proof matrix',
      canWriteLive,
      liveWriteKinds: canWriteLive ? [str(row.action_type, 'unknown')] : [],
      pipelineProfiles: [status],
      blockedReason: canWriteLive ? str(row.current_blocker, '') : str(row.current_blocker, 'No live proof.'),
      nextStep: str(row.next_required_evidence, 'Collect proof before promotion.'),
      tone: toneForModuleStatus(status, canWriteLive)
    };
  });

  const readiness = asRecord(data.autonomy_promotion_readiness);
  const readinessSummary = asRecord(readiness.summary);
  const gapSummary = asRecord(readinessSummary.promotion_gap_summary);
  const modeGate = asRecord(data.autonomy_mode_transition_gate);
  const autonomyPromotionFullAutonomousReady = Boolean(readinessSummary.autonomy_promotion_full_autonomous_ready);
  const autonomyModeTransitionFullAutonomousReady = Boolean(readinessSummary.autonomy_mode_transition_full_autonomous_ready);
  const autonomyPromotionSupervisedLiveAllowedTargetCount = Math.trunc(
    num(readinessSummary.autonomy_promotion_supervised_live_allowed_target_count, 0),
  );
  const autonomyPromotionBlockedTargetCount = Math.trunc(
    num(readinessSummary.autonomy_promotion_blocked_target_count, 0),
  );
  const autonomyModeTransitionGateModeCount = Math.trunc(num(readinessSummary.autonomy_mode_transition_gate_mode_count, 0));
  const autonomyModeTransitionGateBlockedModeCount = Math.trunc(
    num(readinessSummary.autonomy_mode_transition_gate_blocked_mode_count, 0),
  );
  const autonomyModeTransitionGateBlockedModes = stringList(readinessSummary.autonomy_mode_transition_gate_blocked_modes);
  const autonomyModeTransitionGateLiveApplyAllowedModes = stringList(readinessSummary.autonomy_mode_transition_gate_live_apply_allowed_modes);
  const autonomyModeTransitionGateSupervisedCurrentLiveWriteKinds = stringList(
    readinessSummary.autonomy_mode_transition_gate_supervised_current_live_write_kinds,
  );
  const autonomyModeTransitionGateSupervisedCurrentLiveWriteSurfaces = stringList(
    readinessSummary.autonomy_mode_transition_gate_supervised_current_live_write_surfaces,
  );
  const hasAutonomyReadinessSummarySignals = (
    autonomyPromotionFullAutonomousReady
    || autonomyModeTransitionFullAutonomousReady
    || autonomyPromotionSupervisedLiveAllowedTargetCount > 0
    || autonomyPromotionBlockedTargetCount > 0
    || autonomyModeTransitionGateModeCount > 0
    || autonomyModeTransitionGateBlockedModeCount > 0
    || autonomyModeTransitionGateBlockedModes.length > 0
    || autonomyModeTransitionGateLiveApplyAllowedModes.length > 0
    || autonomyModeTransitionGateSupervisedCurrentLiveWriteKinds.length > 0
    || autonomyModeTransitionGateSupervisedCurrentLiveWriteSurfaces.length > 0
  );
  const modeGateRows = asArray(modeGate.rows).map(row => {
    const mode = str(row.mode, 'unknown_mode');
    const status = str(row.status, 'unknown');
    const blockers = stringList(row.blockers);
    const nextActions = stringList(row.next_required_actions);
    const writeKinds = stringList(row.current_live_write_kinds);
    const writeSurfaces = stringList(row.current_live_write_surfaces);
    const writePolicyText = [
      writeKinds.length ? `policy:${writeKinds.join(',')}` : '',
      writeSurfaces.length ? `surfaces:${writeSurfaces.join(',')}` : '',
    ].filter(Boolean).join(' / ');
    const gapCounts = asRecord(row.readiness_gap_bucket_counts);
    const gapText = Object.entries(gapCounts)
      .map(([bucket, rawCount]) => {
        const count = Math.trunc(num(rawCount, 0));
        return count > 0 ? `${bucket}:${count}` : '';
      })
      .filter(Boolean)
      .join(', ');
    const nextStep = [
      nextActions.join(' / '),
      writeKinds.length ? `Current policy live-write kinds: ${writeKinds.join(', ')}` : '',
      !Boolean(row.live_apply_allowed) ? 'No live-write authority for this mode' : '',
      gapText ? `Full Auto gaps: ${gapText}` : '',
    ].filter(Boolean).join(' / ') || 'Mode gate is read-only.';
    return {
      moduleId: `mode_gate:${mode}`,
      label: str(row.project_mode_name, `Mode gate: ${titleFromToken(mode)}`),
      domain: 'autonomy_mode_gate',
      status,
      entrypoint: mode,
      module: writePolicyText || 'no_current_write_authority',
      uiSurface: 'Autonomy mode transition gate',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `runtime_entry:${Boolean(row.runtime_entry_allowed)}`,
        `proposal_creation:${Boolean(row.proposal_creation_allowed)}`,
        `live_apply:${Boolean(row.live_apply_allowed)}`,
        ...writeKinds.map(kind => `current_write:${kind}`),
        ...writeSurfaces.map(surface => `current_write_surface:${surface}`),
      ],
      blockedReason: blockers.length
        ? blockers.join(', ')
        : 'Mode gate is status-only and does not change runtime mode.',
      nextStep,
      tone: toneForModeGateStatus(status)
    };
  });
  const readinessRows = asArray(readiness.rows).map(row => {
    const targetId = str(row.id, 'unknown_promotion_target');
    const status = str(row.status, 'unknown');
    const blockers = stringList(row.blockers);
    const nextRequired = stringList(row.next_required_evidence);
    const evidenceRequirements = stringList(row.evidence_requirement_ids);
    const readSurfaceStatuses = stringList(row.read_surface_statuses);
    const writeSurfaceStatuses = stringList(row.write_surface_statuses);
    const profileTags = [
      str(row.target_mode),
      str(row.safety_status),
      ...readSurfaceStatuses,
      ...writeSurfaceStatuses
    ].filter(Boolean);
    const evidenceText = evidenceRequirements.length
      ? `Evidence: ${evidenceRequirements.join(', ')}`
      : '';
    const nextStep = [...nextRequired, evidenceText].filter(Boolean).join(' / ')
      || 'Collect proof before promotion.';
    return {
      moduleId: `promotion_readiness:${targetId}`,
      label: str(row.label, targetId),
      domain: 'promotion_readiness',
      status,
      entrypoint: str(row.action_family, 'unknown_action_family'),
      module: str(row.safety_row_id, 'no_safety_row'),
      uiSurface: 'Autonomy promotion readiness',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: profileTags,
      blockedReason: blockers.length
        ? blockers.join(', ')
        : 'Static readiness row; apply authority remains in Product Layer safety gates.',
      nextStep,
      tone: toneForModuleStatus(status, false)
    };
  });

  const readinessSummaryRows = hasAutonomyReadinessSummarySignals
    ? [{
      moduleId: 'autonomy_promotion_summary',
      label: 'Autonomy readiness summary',
      domain: 'autonomy_promotion_summary',
      status: autonomyPromotionFullAutonomousReady && autonomyModeTransitionFullAutonomousReady
        ? 'available'
        : 'blocked',
      entrypoint: 'main_pipeline_preview.summary',
      module: 'main_pipeline_preview.summary',
      uiSurface: 'Autonomy readiness summary',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `promotion_full_autonomous_ready:${autonomyPromotionFullAutonomousReady}`,
        `mode_transition_full_autonomous_ready:${autonomyModeTransitionFullAutonomousReady}`,
        `promotion_supervised_live_allowed_target_count:${autonomyPromotionSupervisedLiveAllowedTargetCount}`,
        `promotion_blocked_target_count:${autonomyPromotionBlockedTargetCount}`,
        `mode_transition_gate_mode_count:${autonomyModeTransitionGateModeCount}`,
        `mode_transition_gate_blocked_mode_count:${autonomyModeTransitionGateBlockedModeCount}`,
        ...autonomyModeTransitionGateBlockedModes.map(mode => `mode_transition_gate_blocked_mode:${mode}`),
        ...autonomyModeTransitionGateLiveApplyAllowedModes.map(mode => `mode_transition_gate_live_apply_mode:${mode}`),
        ...autonomyModeTransitionGateSupervisedCurrentLiveWriteKinds.map(kind => `supervised_current_live_write_kind:${kind}`),
        ...autonomyModeTransitionGateSupervisedCurrentLiveWriteSurfaces.map(
          surface => `supervised_current_live_write_surface:${surface}`,
        ),
      ],
      blockedReason: autonomyPromotionFullAutonomousReady && autonomyModeTransitionFullAutonomousReady
        ? 'No outstanding autonomy readiness blockers.'
        : `Mode transition full autonomous ready: ${autonomyModeTransitionFullAutonomousReady}; Promotion full autonomous ready: ${autonomyPromotionFullAutonomousReady}`,
      nextStep: [
        `supervised_live_allowed_target_count:${autonomyPromotionSupervisedLiveAllowedTargetCount}`,
        `blocked_target_count:${autonomyPromotionBlockedTargetCount}`,
        ...autonomyModeTransitionGateBlockedModes.map(mode => `blocked_transition_mode:${mode}`),
      ].join(' / '),
      tone: autonomyPromotionFullAutonomousReady && autonomyModeTransitionFullAutonomousReady ? 'green' : 'red'
    }]
    : [];

  const projectGoalCapabilityMatrix = asRecord(data.project_goal_capability_matrix);
  const projectGoalRows = asArray(projectGoalCapabilityMatrix.rows);
  const projectGoalCapabilitySummary = asRecord(projectGoalCapabilityMatrix.summary);
  const projectGoalCapabilityRows: ModuleStatusRecord[] = projectGoalRows.map(row => {
    const status = str(row.status, 'unknown');
    const canWriteLive = Boolean(row.currently_live_writable);
    const goalArea = str(row.goal_area, 'unscoped');
    const blocker = str(row.blocker, '').trim();
    const methods = stringList(row.method_ids);
    const supportPaths = stringList(row.support_paths);
    const liveKinds = stringList(row.current_live_write_kinds);
    const targetSurfaces = stringList(row.target_console_surfaces);

    return {
      moduleId: `project_goal_capability:${str(row.id, 'unknown_capability')}`,
      label: str(row.label, str(row.id, 'Unnamed capability')),
      domain: 'project_goal_capability',
      status,
      entrypoint: methods[0] || str(row.goal_area, 'project_goal_capability_matrix'),
      module: str(row.goal_area, 'project_goal_capability'),
      uiSurface: 'Project goal capability matrix',
      canWriteLive,
      liveWriteKinds: liveKinds,
      pipelineProfiles: [
        `area:${goalArea}`,
        `ceiling:${str(row.autonomy_mode_ceiling, 'unknown')}`,
        `blockers:${blocker ? 1 : 0}`,
        `target_surfaces:${targetSurfaces.length}`,
        `support_paths:${supportPaths.length}`,
        `methods:${methods.length}`,
        `current_live_writable:${canWriteLive}`,
        `goal_next_step:${str(row.next_step, '').slice(0, 64)}`,
        `status:${status}=1`,
      ].filter(Boolean),
      blockedReason: blocker
        ? blocker
        : `No explicit blocker currently recorded for ${str(row.label, 'this capability')}.`,
      nextStep: str(row.next_step, str(projectGoalCapabilitySummary.next_global_step, 'Advance roadmap proof and safety checks.')),
      tone: toneForModuleStatus(status, canWriteLive),
    };
  });
  const projectGoalCapabilitySummaryRows: ModuleStatusRecord[] = Object.keys(projectGoalCapabilitySummary).length
    ? [{
      moduleId: 'project_goal_capability_summary',
      label: 'Project goal capability matrix summary',
      domain: 'project_goal_capability',
      status: str(projectGoalCapabilityMatrix.status, 'available'),
      entrypoint: 'project_goal_capability_matrix.summary',
      module: `project_goal_capability_matrix.row_count:${Math.trunc(num(projectGoalCapabilitySummary.row_count, 0))}`,
      uiSurface: 'Project goal capability matrix summary',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `row_count:${Math.trunc(num(projectGoalCapabilitySummary.row_count, 0))}`,
        `missing_capability_count:${stringList(projectGoalCapabilitySummary.missing_capability_ids).length}`,
        `blocked_live_count:${stringList(projectGoalCapabilitySummary.blocked_live_capability_ids).length}`,
        `currently_live_writable_capability_count:${stringList(projectGoalCapabilitySummary.currently_live_writable_capability_ids).length}`,
        `next_global_step:${str(projectGoalCapabilitySummary.next_global_step, '').slice(0, 96)}`,
      ].filter(Boolean),
      blockedReason: stringList(projectGoalCapabilitySummary.blocked_live_capability_ids).length > 0
        ? `Blocked capabilities: ${stringList(projectGoalCapabilitySummary.blocked_live_capability_ids).slice(0, 8).join(', ')}`
        : 'No blocked-live capabilities currently listed by capability-matrix summary.',
      nextStep: str(projectGoalCapabilitySummary.next_global_step, 'Continue roadmap proof and safety work.'),
      tone: stringList(projectGoalCapabilitySummary.blocked_live_capability_ids).length
        ? 'red'
        : stringList(projectGoalCapabilitySummary.missing_capability_ids).length
          ? 'yellow'
          : 'blue',
    }]
    : [];

  const progressSnapshot = asRecord(data.project_goal_capability_progress_snapshot);
  const progressSummary = asRecord(progressSnapshot.summary);
  const progressRows = asArray(progressSnapshot.next_steps);
  const progressSummaryRows: ModuleStatusRecord[] = Object.keys(progressSummary).length
    ? [{
      moduleId: 'project_goal_capability_progress_summary',
      label: 'Project goal capability progress summary',
      domain: 'project_goal_capability',
      status: str(progressSnapshot.status, 'available'),
      entrypoint: 'project_goal_capability_progress_snapshot.summary',
      module: `project_goal_progress_snapshot.row_count:${Math.trunc(num(progressSummary.row_count, 0))}`,
      uiSurface: 'Project goal capability progress',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `row_count:${Math.trunc(num(progressSummary.row_count, 0))}`,
        `missing:${stringList(progressSummary.missing_capability_ids).length}`,
        `blocked_live:${stringList(progressSummary.blocked_live_capability_ids).length}`,
        `proposal_only:${stringList(progressSummary.proposal_only_capability_ids).length}`,
        `current_supervised_candidates:${stringList(progressSummary.supervised_narrow_live_candidates).length}`,
        `goal_areas_with_blockers:${stringList(progressSummary.goal_areas_with_blockers).join(',')}`,
        `next_global_step:${str(progressSummary.next_global_step, '').slice(0, 96)}`,
      ].filter(Boolean),
      blockedReason: progressRows.length
        ? `Next actions remain for ${progressRows.length} capability(s).`
        : `Roadmap progress snapshot has no explicit next actions. ${str(progressSummary.next_global_step, 'Keep adding proof and safety evidence.')}`,
      nextStep: asArray(progressRows).slice(0, 3).map(progressRow => {
        const capabilityId = str(progressRow.capability_id, '');
        if (!capabilityId) return '';
        const area = str(progressRow.goal_area, 'unknown');
        const target = str(progressRow.autonomy_mode_ceiling, 'unknown');
        const summary = str(progressRow.next_step, 'Next step available in pipeline snapshot.');
        return `${capabilityId} (${area}/${target}): ${summary}`;
      }).filter(Boolean).join(' / ') || str(progressSummary.next_global_step, ''),
      tone: stringList(progressSummary.blocked_live_capability_ids).length > 0
        ? 'red'
        : stringList(progressSummary.missing_capability_ids).length > 0
          ? 'yellow'
          : 'blue',
    }]
    : [];

  const objectiveAlignmentSnapshot = asRecord(data.project_goal_capability_objective_alignment_snapshot);
  const objectiveAlignmentSummary = asRecord(objectiveAlignmentSnapshot.summary);
  const missingObjectiveRows = stringList(objectiveAlignmentSummary.missing_rows);
  const blockedLiveRows = stringList(objectiveAlignmentSummary.blocked_live_rows);
  const currentlyLiveWritableRows = stringList(objectiveAlignmentSummary.currently_live_writable_rows);
  const proposalOnlyRows = stringList(objectiveAlignmentSummary.proposal_only_row_ids);
  const supervisedCandidateRows = stringList(objectiveAlignmentSummary.supervised_candidate_row_ids);
  const blockedLiveActions = stringList(objectiveAlignmentSummary.blocked_live_row_actions);
  const objectiveAlignmentRows: ModuleStatusRecord[] = Object.keys(objectiveAlignmentSummary).length
    ? [{
      moduleId: 'project_goal_capability_objective_alignment_summary',
      label: 'Project goal objective alignment',
      domain: 'project_goal_capability',
      status: str(objectiveAlignmentSnapshot.status, 'available'),
      entrypoint: 'project_goal_capability_objective_alignment_snapshot.summary',
      module: `required:${Math.trunc(num(objectiveAlignmentSummary.required_row_count, 0))}`,
      uiSurface: 'Project goal objective alignment summary',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `required:${Math.trunc(num(objectiveAlignmentSummary.required_row_count, 0))}`,
        `present:${Math.trunc(num(objectiveAlignmentSummary.present_row_count, 0))}`,
        `missing:${Math.trunc(num(objectiveAlignmentSummary.missing_row_count, 0))}`,
        `proposal_only:${Math.trunc(num(objectiveAlignmentSummary.proposal_only_row_count, 0))}`,
        `blocked_live_rows:${Math.trunc(num(objectiveAlignmentSummary.blocked_live_row_count, 0))}`,
        `blocked_live_row_ids:${blockedLiveRows.slice(0, 4).join(',')}`,
        `currently_live_writable:${Math.trunc(num(objectiveAlignmentSummary.currently_live_writable_row_count, 0))}`,
        `currently_live_writable_ids:${currentlyLiveWritableRows.slice(0, 4).join(',')}`,
        `proposal_only_row_ids:${proposalOnlyRows.slice(0, 4).join(',')}`,
        `supervised_candidates:${Math.trunc(num(objectiveAlignmentSummary.supervised_candidate_row_count, 0))}`,
        `supervised_candidate_row_ids:${supervisedCandidateRows.slice(0, 4).join(',')}`,
        `blocked_live_count:${blockedLiveRows.length}`,
        `blocked_live_actions:${blockedLiveActions.slice(0, 2).join(' | ')}`,
        `missing_row_ids:${missingObjectiveRows.slice(0, 4).join(',')}`,
        `full_autonomous_status:${str(objectiveAlignmentSummary.full_autonomous_status, 'blocked')}`,
      ].filter((entry) => entry),
      blockedReason: missingObjectiveRows.length
        ? `Missing objective capabilities: ${missingObjectiveRows.join(', ')}`
        : blockedLiveRows.length
          ? `Blocked-live objective capabilities: ${blockedLiveRows.join(', ')}`
        : 'Objective alignment summary is currently complete.',
      nextStep: [
        `full_autonomous_status:${str(objectiveAlignmentSummary.full_autonomous_status, 'blocked')}`,
        `missing_rows:${missingObjectiveRows.slice(0, 4).join(',')}`,
        `blocked_live_rows:${blockedLiveRows.slice(0, 4).join(',')}`,
        `blocked_live_actions:${blockedLiveActions.slice(0, 2).join(' | ')}`,
      ].filter((entry) => entry).join(' / ') || `full_autonomous_status:${str(objectiveAlignmentSummary.full_autonomous_status, 'blocked')}`,
      tone: missingObjectiveRows.length
        ? 'red'
        : blockedLiveRows.length
          ? 'red'
          : 'blue',
    }]
    : [];

  const proofEvidence = asRecord(data.supervised_proof_evidence);
  const proofSummary = asRecord(proofEvidence.summary);
  const proofEvidenceRows: ModuleStatusRecord[] = Object.keys(proofEvidence).length ? (() => {
    const available = Boolean(proofEvidence.available);
    const reportPath = str(proofEvidence.report_path, 'no_report_artifact');
    const overlayStatus = str(proofSummary.overlay_status, 'no_current_evidence');
    const artifactCount = Math.trunc(num(proofSummary.artifact_count, 0));
    const invalidCount = Math.trunc(num(proofSummary.invalid_summary_count, 0));
    const stale = Boolean(proofSummary.latest_report_stale);
    const ageSec = Math.max(0, num(proofSummary.latest_report_age_sec, 0));
    const staleAfterSec = Math.max(0, num(proofSummary.stale_after_sec, 0));
    const proofQuality = asRecord(proofSummary.read_surface_proof_quality);
    const proofQualityAvailable = Boolean(proofQuality.available);
    const physicalGapSurfaceCount = Math.trunc(num(proofQuality.physical_gap_surface_count, 0));
    const declaredFixturePathCount = Math.trunc(num(proofQuality.declared_fixture_path_count, 0));
    const declaredTestNameCount = Math.trunc(num(proofQuality.declared_test_name_count, 0));
    const summaryRecomputed = Boolean(proofQuality.summary_recomputed_from_rows);
    const fixtureDeclarationValidation = Boolean(proofQuality.fixture_declaration_validation);
    const contractGapConsistency = Boolean(proofQuality.contract_gap_consistency_enforced);
    const physicalGapSurfaceIds = stringList(proofQuality.physical_fixture_missing_surface_ids);
    const physicalGapSurfaceText = physicalGapSurfaceIds.join('|');
    const nextPhysicalReadbackSurfaceId = str(
      proofQuality.next_physical_readback_surface_id,
      physicalGapSurfaceIds[0] ?? '',
    );
    const sourceNextPhysicalReadbackSurfaceId = str(proofQuality.source_next_physical_readback_surface_id, '');
    const nextPhysicalReadbackTargetConsistent = proofQuality.next_physical_readback_target_consistent === true;
    const readSurfaceCapturePlan = asRecord(proofQuality.read_surface_physical_readback_capture_plan);
    const readSurfaceCapturePlanAvailable = Boolean(readSurfaceCapturePlan.available);
    const readSurfaceCapturePlanStatus = str(readSurfaceCapturePlan.plan_status, str(readSurfaceCapturePlan.status, 'missing'));
    const readSurfaceCapturePlanStepCount = Math.trunc(num(readSurfaceCapturePlan.capture_step_count, 0));
    const readSurfaceCapturePlanSurfaceId = str(readSurfaceCapturePlan.next_physical_readback_surface_id, '');
    const readSurfaceCapturePlanActionId = str(readSurfaceCapturePlan.next_physical_readback_action_id, '');
    const readSurfaceCapturePlanSourceSurfaceId = str(readSurfaceCapturePlan.source_next_physical_readback_surface_id, '');
    const readSurfaceCapturePlanSourceActionId = str(readSurfaceCapturePlan.source_next_physical_readback_action_id, '');
    const readSurfaceCapturePlanExecutionAllowed = readSurfaceCapturePlan.execution_allowed_by_plan === true;
    const readSurfaceCapturePlanOperatorRequired = readSurfaceCapturePlan.operator_required === true;
    const readSurfaceCapturePlanLiveAuthority = readSurfaceCapturePlan.live_authority_granted === true;
    const readSurfaceCapturePlanTargetConsistency = readSurfaceCapturePlan.target_consistency === true;
    const readSurfaceCapturePlanRequiredContracts = stringList(readSurfaceCapturePlan.required_artifact_contracts);
    const readSurfaceCapturePlanBlockedBy = stringList(readSurfaceCapturePlan.blocked_by);
    const readSurfaceCapturePlanForbidden = stringList(readSurfaceCapturePlan.step_forbidden_actions);
    const rawMixerContract = asRecord(proofQuality.raw_mixer_state_physical_readback_contract);
    const rawMixerContractAvailable = Boolean(rawMixerContract.available);
    const rawMixerContractStatus = str(rawMixerContract.contract_status, str(rawMixerContract.status, 'missing'));
    const rawMixerContractSchemaId = str(rawMixerContract.schema_id, '');
    const rawMixerContractSurfaceId = str(rawMixerContract.surface_id, '');
    const rawMixerContractActionId = str(rawMixerContract.remediation_action_id, '');
    const rawMixerContractNextSurfaceId = str(rawMixerContract.next_physical_readback_surface_id, '');
    const rawMixerContractNextActionId = str(rawMixerContract.next_physical_readback_action_id, '');
    const rawMixerContractGapRemains = rawMixerContract.physical_fixture_gap_remains === true;
    const rawMixerContractLiveAuthority = rawMixerContract.live_authority_granted === true;
    const rawMixerContractTargetConsistency = rawMixerContract.target_consistency === true;
    const rawMixerIntake = asRecord(proofQuality.raw_mixer_state_physical_readback_artifact_intake);
    const rawMixerIntakeAvailable = Boolean(rawMixerIntake.available);
    const rawMixerIntakeStatus = str(rawMixerIntake.intake_status, str(rawMixerIntake.status, 'missing'));
    const rawMixerIntakeSchemaId = str(rawMixerIntake.schema_id, '');
    const rawMixerIntakeSurfaceId = str(rawMixerIntake.surface_id, '');
    const rawMixerIntakeActionId = str(rawMixerIntake.remediation_action_id, '');
    const rawMixerIntakeArtifactCount = Math.trunc(num(rawMixerIntake.artifact_count, 0));
    const rawMixerIntakeValidCount = Math.trunc(num(rawMixerIntake.valid_candidate_count, 0));
    const rawMixerIntakeInvalidCount = Math.trunc(num(rawMixerIntake.invalid_candidate_count, 0));
    const rawMixerIntakeCandidate = rawMixerIntake.accepted_as_physical_fixture_candidate === true;
    const rawMixerIntakeOperatorReview = rawMixerIntake.operator_review_required === true;
    const rawMixerIntakeGapRemains = rawMixerIntake.physical_fixture_gap_remains === true;
    const rawMixerIntakeClosureAllowed = rawMixerIntake.fixture_closure_allowed_by_intake === true;
    const rawMixerIntakeLiveAuthority = rawMixerIntake.live_authority_granted === true;
    const rawMixerIntakeTargetConsistency = rawMixerIntake.target_consistency === true;
    const rawMixerIntakeBlockedBy = stringList(rawMixerIntake.blocked_by);
    const rawMixerIntakeSources = stringList(rawMixerIntake.artifact_source_ids);
    const channelBasicContract = asRecord(proofQuality.channel_basic_controls_physical_readback_contract);
    const channelBasicContractAvailable = Boolean(channelBasicContract.available);
    const channelBasicContractStatus = str(channelBasicContract.contract_status, str(channelBasicContract.status, 'missing'));
    const channelBasicContractSchemaId = str(channelBasicContract.schema_id, '');
    const channelBasicContractSurfaceId = str(channelBasicContract.surface_id, '');
    const channelBasicContractActionId = str(channelBasicContract.remediation_action_id, '');
    const channelBasicContractPhysicalSurfaceId = str(channelBasicContract.physical_readback_surface_id, '');
    const channelBasicContractPhysicalActionId = str(channelBasicContract.physical_readback_action_id, '');
    const channelBasicContractGapRemains = channelBasicContract.physical_fixture_gap_remains === true;
    const channelBasicContractLiveAuthority = channelBasicContract.live_authority_granted === true;
    const channelBasicContractTargetConsistency = channelBasicContract.target_consistency === true;
    const channelBasicContractPromotionFamilies = stringList(channelBasicContract.promotion_families_requiring_surface);
    const channelBasicContractControls = stringList(channelBasicContract.required_control_fields);
    const channelBasicIntake = asRecord(proofQuality.channel_basic_controls_physical_readback_artifact_intake);
    const channelBasicIntakeAvailable = Boolean(channelBasicIntake.available);
    const channelBasicIntakeStatus = str(channelBasicIntake.intake_status, str(channelBasicIntake.status, 'missing'));
    const channelBasicIntakeSchemaId = str(channelBasicIntake.schema_id, '');
    const channelBasicIntakeSurfaceId = str(channelBasicIntake.surface_id, '');
    const channelBasicIntakeActionId = str(channelBasicIntake.remediation_action_id, '');
    const channelBasicIntakeArtifactCount = Math.trunc(num(channelBasicIntake.artifact_count, 0));
    const channelBasicIntakeValidCount = Math.trunc(num(channelBasicIntake.valid_candidate_count, 0));
    const channelBasicIntakeInvalidCount = Math.trunc(num(channelBasicIntake.invalid_candidate_count, 0));
    const channelBasicIntakeValidChannelCount = Math.trunc(num(channelBasicIntake.valid_channel_count, 0));
    const channelBasicIntakeCandidate = channelBasicIntake.accepted_as_physical_fixture_candidate === true;
    const channelBasicIntakeOperatorReview = channelBasicIntake.operator_review_required === true;
    const channelBasicIntakeGapRemains = channelBasicIntake.physical_fixture_gap_remains === true;
    const channelBasicIntakeClosureAllowed = channelBasicIntake.fixture_closure_allowed_by_intake === true;
    const channelBasicIntakeLiveAuthority = channelBasicIntake.live_authority_granted === true;
    const channelBasicIntakeTargetConsistency = channelBasicIntake.target_consistency === true;
    const channelBasicIntakePromotionFamilies = stringList(channelBasicIntake.promotion_families_requiring_surface);
    const channelBasicIntakeBlockedBy = stringList(channelBasicIntake.blocked_by);
    const channelBasicIntakeSources = stringList(channelBasicIntake.artifact_source_ids);
    const channelProcessingContract = asRecord(proofQuality.channel_processing_physical_readback_contract);
    const channelProcessingContractAvailable = Boolean(channelProcessingContract.available);
    const channelProcessingContractStatus = str(channelProcessingContract.contract_status, str(channelProcessingContract.status, 'missing'));
    const channelProcessingContractSchemaId = str(channelProcessingContract.schema_id, '');
    const channelProcessingContractSurfaceId = str(channelProcessingContract.surface_id, '');
    const channelProcessingContractActionId = str(channelProcessingContract.remediation_action_id, '');
    const channelProcessingContractPhysicalSurfaceId = str(channelProcessingContract.physical_readback_surface_id, '');
    const channelProcessingContractPhysicalActionId = str(channelProcessingContract.physical_readback_action_id, '');
    const channelProcessingContractGapRemains = channelProcessingContract.physical_fixture_gap_remains === true;
    const channelProcessingContractLiveAuthority = channelProcessingContract.live_authority_granted === true;
    const channelProcessingContractTargetConsistency = channelProcessingContract.target_consistency === true;
    const channelProcessingContractPromotionFamilies = stringList(channelProcessingContract.promotion_families_requiring_surface);
    const channelProcessingContractEqBands = stringList(channelProcessingContract.required_eq_band_fields);
    const channelProcessingContractCompressor = stringList(channelProcessingContract.required_compressor_fields);
    const channelProcessingContractGate = stringList(channelProcessingContract.required_gate_fields);
    const channelProcessingIntake = asRecord(proofQuality.channel_processing_physical_readback_artifact_intake);
    const channelProcessingIntakeAvailable = Boolean(channelProcessingIntake.available);
    const channelProcessingIntakeStatus = str(channelProcessingIntake.intake_status, str(channelProcessingIntake.status, 'missing'));
    const channelProcessingIntakeSchemaId = str(channelProcessingIntake.schema_id, '');
    const channelProcessingIntakeSurfaceId = str(channelProcessingIntake.surface_id, '');
    const channelProcessingIntakeActionId = str(channelProcessingIntake.remediation_action_id, '');
    const channelProcessingIntakeArtifactCount = Math.trunc(num(channelProcessingIntake.artifact_count, 0));
    const channelProcessingIntakeValidCount = Math.trunc(num(channelProcessingIntake.valid_candidate_count, 0));
    const channelProcessingIntakeInvalidCount = Math.trunc(num(channelProcessingIntake.invalid_candidate_count, 0));
    const channelProcessingIntakeValidChannelCount = Math.trunc(num(channelProcessingIntake.valid_channel_count, 0));
    const channelProcessingIntakeCandidate = channelProcessingIntake.accepted_as_physical_fixture_candidate === true;
    const channelProcessingIntakeOperatorReview = channelProcessingIntake.operator_review_required === true;
    const channelProcessingIntakeGapRemains = channelProcessingIntake.physical_fixture_gap_remains === true;
    const channelProcessingIntakeClosureAllowed = channelProcessingIntake.fixture_closure_allowed_by_intake === true;
    const channelProcessingIntakeLiveAuthority = channelProcessingIntake.live_authority_granted === true;
    const channelProcessingIntakeTargetConsistency = channelProcessingIntake.target_consistency === true;
    const channelProcessingIntakePromotionFamilies = stringList(channelProcessingIntake.promotion_families_requiring_surface);
    const channelProcessingIntakeBlockedBy = stringList(channelProcessingIntake.blocked_by);
    const channelProcessingIntakeSources = stringList(channelProcessingIntake.artifact_source_ids);
    const busMainDcaContract = asRecord(proofQuality.bus_group_main_dca_physical_readback_contract);
    const busMainDcaContractAvailable = Boolean(busMainDcaContract.available);
    const busMainDcaContractStatus = str(busMainDcaContract.contract_status, str(busMainDcaContract.status, 'missing'));
    const busMainDcaContractSchemaId = str(busMainDcaContract.schema_id, '');
    const busMainDcaContractSurfaceId = str(busMainDcaContract.surface_id, '');
    const busMainDcaContractActionId = str(busMainDcaContract.remediation_action_id, '');
    const busMainDcaContractPhysicalSurfaceId = str(busMainDcaContract.physical_readback_surface_id, '');
    const busMainDcaContractPhysicalActionId = str(busMainDcaContract.physical_readback_action_id, '');
    const busMainDcaContractGapRemains = busMainDcaContract.physical_fixture_gap_remains === true;
    const busMainDcaContractLiveAuthority = busMainDcaContract.live_authority_granted === true;
    const busMainDcaContractTargetConsistency = busMainDcaContract.target_consistency === true;
    const busMainDcaContractPromotionFamilies = stringList(busMainDcaContract.promotion_families_requiring_surface);
    const busMainDcaContractBusControls = stringList(busMainDcaContract.required_bus_control_fields);
    const busMainDcaContractMainControls = stringList(busMainDcaContract.required_main_control_fields);
    const busMainDcaContractDcaControls = stringList(busMainDcaContract.required_dca_control_fields);
    const busMainDcaContractBusAssignments = stringList(busMainDcaContract.required_bus_assignment_fields);
    const busMainDcaContractDcaAssignments = stringList(busMainDcaContract.required_dca_assignment_fields);
    const busMainDcaContractMetering = stringList(busMainDcaContract.required_metering_fields);
    const busMainDcaIntake = asRecord(proofQuality.bus_group_main_dca_physical_readback_artifact_intake);
    const busMainDcaIntakeAvailable = Boolean(busMainDcaIntake.available);
    const busMainDcaIntakeStatus = str(busMainDcaIntake.intake_status, str(busMainDcaIntake.status, 'missing'));
    const busMainDcaIntakeSchemaId = str(busMainDcaIntake.schema_id, '');
    const busMainDcaIntakeSurfaceId = str(busMainDcaIntake.surface_id, '');
    const busMainDcaIntakeActionId = str(busMainDcaIntake.remediation_action_id, '');
    const busMainDcaIntakeArtifactCount = Math.trunc(num(busMainDcaIntake.artifact_count, 0));
    const busMainDcaIntakeValidCount = Math.trunc(num(busMainDcaIntake.valid_candidate_count, 0));
    const busMainDcaIntakeInvalidCount = Math.trunc(num(busMainDcaIntake.invalid_candidate_count, 0));
    const busMainDcaIntakeValidBusCount = Math.trunc(num(busMainDcaIntake.valid_bus_count, 0));
    const busMainDcaIntakeValidMainCount = Math.trunc(num(busMainDcaIntake.valid_main_count, 0));
    const busMainDcaIntakeValidDcaCount = Math.trunc(num(busMainDcaIntake.valid_dca_count, 0));
    const busMainDcaIntakeCandidate = busMainDcaIntake.accepted_as_physical_fixture_candidate === true;
    const busMainDcaIntakeOperatorReview = busMainDcaIntake.operator_review_required === true;
    const busMainDcaIntakeGapRemains = busMainDcaIntake.physical_fixture_gap_remains === true;
    const busMainDcaIntakeClosureAllowed = busMainDcaIntake.fixture_closure_allowed_by_intake === true;
    const busMainDcaIntakeLiveAuthority = busMainDcaIntake.live_authority_granted === true;
    const busMainDcaIntakeTargetConsistency = busMainDcaIntake.target_consistency === true;
    const busMainDcaIntakePromotionFamilies = stringList(busMainDcaIntake.promotion_families_requiring_surface);
    const busMainDcaIntakeBlockedBy = stringList(busMainDcaIntake.blocked_by);
    const busMainDcaIntakeSources = stringList(busMainDcaIntake.artifact_source_ids);
    const matrixStateContract = asRecord(proofQuality.matrix_state_physical_readback_contract);
    const matrixStateContractAvailable = Boolean(matrixStateContract.available);
    const matrixStateContractStatus = str(matrixStateContract.contract_status, str(matrixStateContract.status, 'missing'));
    const matrixStateContractSchemaId = str(matrixStateContract.schema_id, '');
    const matrixStateContractSurfaceId = str(matrixStateContract.surface_id, '');
    const matrixStateContractActionId = str(matrixStateContract.remediation_action_id, '');
    const matrixStateContractPhysicalSurfaceId = str(matrixStateContract.physical_readback_surface_id, '');
    const matrixStateContractPhysicalActionId = str(matrixStateContract.physical_readback_action_id, '');
    const matrixStateContractGapRemains = matrixStateContract.physical_fixture_gap_remains === true;
    const matrixStateContractLiveAuthority = matrixStateContract.live_authority_granted === true;
    const matrixStateContractTargetConsistency = matrixStateContract.target_consistency === true;
    const matrixStateContractPromotionFamilies = stringList(matrixStateContract.promotion_families_requiring_surface);
    const matrixStateContractControls = stringList(matrixStateContract.required_control_fields);
    const matrixStateContractEqBands = stringList(matrixStateContract.required_eq_band_fields);
    const matrixStateContractDynamics = stringList(matrixStateContract.required_dynamics_fields);
    const matrixStateContractInserts = stringList(matrixStateContract.required_insert_fields);
    const matrixStateContractMetering = stringList(matrixStateContract.required_metering_fields);
    const matrixStateIntake = asRecord(proofQuality.matrix_state_physical_readback_artifact_intake);
    const matrixStateIntakeAvailable = Boolean(matrixStateIntake.available);
    const matrixStateIntakeStatus = str(matrixStateIntake.intake_status, str(matrixStateIntake.status, 'missing'));
    const matrixStateIntakeSchemaId = str(matrixStateIntake.schema_id, '');
    const matrixStateIntakeSurfaceId = str(matrixStateIntake.surface_id, '');
    const matrixStateIntakeActionId = str(matrixStateIntake.remediation_action_id, '');
    const matrixStateIntakeArtifactCount = Math.trunc(num(matrixStateIntake.artifact_count, 0));
    const matrixStateIntakeValidCount = Math.trunc(num(matrixStateIntake.valid_candidate_count, 0));
    const matrixStateIntakeInvalidCount = Math.trunc(num(matrixStateIntake.invalid_candidate_count, 0));
    const matrixStateIntakeValidMatrixCount = Math.trunc(num(matrixStateIntake.valid_matrix_count, 0));
    const matrixStateIntakeCandidate = matrixStateIntake.accepted_as_physical_fixture_candidate === true;
    const matrixStateIntakeOperatorReview = matrixStateIntake.operator_review_required === true;
    const matrixStateIntakeGapRemains = matrixStateIntake.physical_fixture_gap_remains === true;
    const matrixStateIntakeClosureAllowed = matrixStateIntake.fixture_closure_allowed_by_intake === true;
    const matrixStateIntakeLiveAuthority = matrixStateIntake.live_authority_granted === true;
    const matrixStateIntakeTargetConsistency = matrixStateIntake.target_consistency === true;
    const matrixStateIntakePromotionFamilies = stringList(matrixStateIntake.promotion_families_requiring_surface);
    const matrixStateIntakeBlockedBy = stringList(matrixStateIntake.blocked_by);
    const matrixStateIntakeSources = stringList(matrixStateIntake.artifact_source_ids);
    const fxSlotsInsertsContract = asRecord(proofQuality.fx_slots_inserts_physical_readback_contract);
    const fxSlotsInsertsContractAvailable = Boolean(fxSlotsInsertsContract.available);
    const fxSlotsInsertsContractStatus = str(fxSlotsInsertsContract.contract_status, str(fxSlotsInsertsContract.status, 'missing'));
    const fxSlotsInsertsContractSchemaId = str(fxSlotsInsertsContract.schema_id, '');
    const fxSlotsInsertsContractSurfaceId = str(fxSlotsInsertsContract.surface_id, '');
    const fxSlotsInsertsContractActionId = str(fxSlotsInsertsContract.remediation_action_id, '');
    const fxSlotsInsertsContractPhysicalSurfaceId = str(fxSlotsInsertsContract.physical_readback_surface_id, '');
    const fxSlotsInsertsContractPhysicalActionId = str(fxSlotsInsertsContract.physical_readback_action_id, '');
    const fxSlotsInsertsContractGapRemains = fxSlotsInsertsContract.physical_fixture_gap_remains === true;
    const fxSlotsInsertsContractLiveAuthority = fxSlotsInsertsContract.live_authority_granted === true;
    const fxSlotsInsertsContractTargetConsistency = fxSlotsInsertsContract.target_consistency === true;
    const fxSlotsInsertsContractPromotionFamilies = stringList(fxSlotsInsertsContract.promotion_families_requiring_surface);
    const fxSlotsInsertsContractSlotFields = stringList(fxSlotsInsertsContract.required_fx_slot_fields);
    const fxSlotsInsertsContractParameterFields = stringList(fxSlotsInsertsContract.required_parameter_fields);
    const fxSlotsInsertsContractParameterSchemaFields = stringList(fxSlotsInsertsContract.required_parameter_schema_fields);
    const fxSlotsInsertsContractInsertFields = stringList(fxSlotsInsertsContract.required_insert_fields);
    const fxSlotsInsertsContractInsertCoverage = stringList(fxSlotsInsertsContract.required_insert_coverage_flags);
    const fxSlotsInsertsContractFalseSafetyFlags = stringList(fxSlotsInsertsContract.required_false_safety_flags);
    const fxSlotsInsertsIntake = asRecord(proofQuality.fx_slots_inserts_physical_readback_artifact_intake);
    const fxSlotsInsertsIntakeAvailable = Boolean(fxSlotsInsertsIntake.available);
    const fxSlotsInsertsIntakeStatus = str(fxSlotsInsertsIntake.intake_status, str(fxSlotsInsertsIntake.status, 'missing'));
    const fxSlotsInsertsIntakeSchemaId = str(fxSlotsInsertsIntake.schema_id, '');
    const fxSlotsInsertsIntakeSurfaceId = str(fxSlotsInsertsIntake.surface_id, '');
    const fxSlotsInsertsIntakeActionId = str(fxSlotsInsertsIntake.remediation_action_id, '');
    const fxSlotsInsertsIntakeArtifactCount = Math.trunc(num(fxSlotsInsertsIntake.artifact_count, 0));
    const fxSlotsInsertsIntakeValidCount = Math.trunc(num(fxSlotsInsertsIntake.valid_candidate_count, 0));
    const fxSlotsInsertsIntakeInvalidCount = Math.trunc(num(fxSlotsInsertsIntake.invalid_candidate_count, 0));
    const fxSlotsInsertsIntakeValidFxSlotCount = Math.trunc(num(fxSlotsInsertsIntake.valid_fx_slot_count, 0));
    const fxSlotsInsertsIntakeValidInsertCount = Math.trunc(num(fxSlotsInsertsIntake.valid_insert_count, 0));
    const fxSlotsInsertsIntakeCandidate = fxSlotsInsertsIntake.accepted_as_physical_fixture_candidate === true;
    const fxSlotsInsertsIntakeOperatorReview = fxSlotsInsertsIntake.operator_review_required === true;
    const fxSlotsInsertsIntakeGapRemains = fxSlotsInsertsIntake.physical_fixture_gap_remains === true;
    const fxSlotsInsertsIntakeClosureAllowed = fxSlotsInsertsIntake.fixture_closure_allowed_by_intake === true;
    const fxSlotsInsertsIntakeLiveAuthority = fxSlotsInsertsIntake.live_authority_granted === true;
    const fxSlotsInsertsIntakeTargetConsistency = fxSlotsInsertsIntake.target_consistency === true;
    const fxSlotsInsertsIntakePromotionFamilies = stringList(fxSlotsInsertsIntake.promotion_families_requiring_surface);
    const fxSlotsInsertsIntakeBlockedBy = stringList(fxSlotsInsertsIntake.blocked_by);
    const fxSlotsInsertsIntakeSources = stringList(fxSlotsInsertsIntake.artifact_source_ids);
    const routingSendsContract = asRecord(proofQuality.routing_sends_physical_readback_contract);
    const routingSendsContractAvailable = Boolean(routingSendsContract.available);
    const routingSendsContractStatus = str(routingSendsContract.contract_status, str(routingSendsContract.status, 'missing'));
    const routingSendsContractSchemaId = str(routingSendsContract.schema_id, '');
    const routingSendsContractSurfaceId = str(routingSendsContract.surface_id, '');
    const routingSendsContractActionId = str(routingSendsContract.remediation_action_id, '');
    const routingSendsContractPhysicalSurfaceId = str(routingSendsContract.physical_readback_surface_id, '');
    const routingSendsContractPhysicalActionId = str(routingSendsContract.physical_readback_action_id, '');
    const routingSendsContractGapRemains = routingSendsContract.physical_fixture_gap_remains === true;
    const routingSendsContractLiveAuthority = routingSendsContract.live_authority_granted === true;
    const routingSendsContractTargetConsistency = routingSendsContract.target_consistency === true;
    const routingSendsContractPromotionFamilies = stringList(routingSendsContract.promotion_families_requiring_surface);
    const routingSendsContractStaticFields = stringList(routingSendsContract.required_static_route_fields);
    const routingSendsContractBusSendFields = stringList(routingSendsContract.required_bus_send_fields);
    const routingSendsContractMainSendFields = stringList(routingSendsContract.required_main_send_fields);
    const routingSendsContractMatrixSendFields = stringList(routingSendsContract.required_matrix_send_fields);
    const routingSendsContractOutputRouteFields = stringList(routingSendsContract.required_output_route_fields);
    const routingSendsContractSendCoverage = stringList(routingSendsContract.required_send_coverage_flags);
    const routingSendsContractOutputCoverage = stringList(routingSendsContract.required_output_route_coverage_flags);
    const routingSendsContractFalseSafetyFlags = stringList(routingSendsContract.required_false_safety_flags);
    const routingSendsIntake = asRecord(proofQuality.routing_sends_physical_readback_artifact_intake);
    const routingSendsIntakeAvailable = Boolean(routingSendsIntake.available);
    const routingSendsIntakeStatus = str(routingSendsIntake.intake_status, str(routingSendsIntake.status, 'missing'));
    const routingSendsIntakeSchemaId = str(routingSendsIntake.schema_id, '');
    const routingSendsIntakeSurfaceId = str(routingSendsIntake.surface_id, '');
    const routingSendsIntakeActionId = str(routingSendsIntake.remediation_action_id, '');
    const routingSendsIntakeArtifactCount = Math.trunc(num(routingSendsIntake.artifact_count, 0));
    const routingSendsIntakeValidCount = Math.trunc(num(routingSendsIntake.valid_candidate_count, 0));
    const routingSendsIntakeInvalidCount = Math.trunc(num(routingSendsIntake.invalid_candidate_count, 0));
    const routingSendsIntakeValidStaticRouteCount = Math.trunc(num(routingSendsIntake.valid_static_dante_route_count, 0));
    const routingSendsIntakeValidBusSendCount = Math.trunc(num(routingSendsIntake.valid_channel_bus_send_count, 0));
    const routingSendsIntakeValidMainSendCount = Math.trunc(num(routingSendsIntake.valid_channel_main_send_count, 0));
    const routingSendsIntakeValidMatrixSendCount = Math.trunc(num(routingSendsIntake.valid_channel_matrix_send_count, 0));
    const routingSendsIntakeValidOutputRouteCount = Math.trunc(num(routingSendsIntake.valid_output_route_count, 0));
    const routingSendsIntakeCandidate = routingSendsIntake.accepted_as_physical_fixture_candidate === true;
    const routingSendsIntakeOperatorReview = routingSendsIntake.operator_review_required === true;
    const routingSendsIntakeGapRemains = routingSendsIntake.physical_fixture_gap_remains === true;
    const routingSendsIntakeClosureAllowed = routingSendsIntake.fixture_closure_allowed_by_intake === true;
    const routingSendsIntakeLiveAuthority = routingSendsIntake.live_authority_granted === true;
    const routingSendsIntakeTargetConsistency = routingSendsIntake.target_consistency === true;
    const routingSendsIntakePromotionFamilies = stringList(routingSendsIntake.promotion_families_requiring_surface);
    const routingSendsIntakeBlockedBy = stringList(routingSendsIntake.blocked_by);
    const routingSendsIntakeSources = stringList(routingSendsIntake.artifact_source_ids);
    const snapshotsScenesContract = asRecord(proofQuality.snapshots_scenes_physical_readback_contract);
    const snapshotsScenesContractAvailable = Boolean(snapshotsScenesContract.available);
    const snapshotsScenesContractStatus = str(snapshotsScenesContract.contract_status, str(snapshotsScenesContract.status, 'missing'));
    const snapshotsScenesContractSchemaId = str(snapshotsScenesContract.schema_id, '');
    const snapshotsScenesContractSurfaceId = str(snapshotsScenesContract.surface_id, '');
    const snapshotsScenesContractActionId = str(snapshotsScenesContract.remediation_action_id, '');
    const snapshotsScenesContractPhysicalSurfaceId = str(snapshotsScenesContract.physical_readback_surface_id, '');
    const snapshotsScenesContractPhysicalActionId = str(snapshotsScenesContract.physical_readback_action_id, '');
    const snapshotsScenesContractGapRemains = snapshotsScenesContract.physical_fixture_gap_remains === true;
    const snapshotsScenesContractLiveAuthority = snapshotsScenesContract.live_authority_granted === true;
    const snapshotsScenesContractTargetConsistency = snapshotsScenesContract.target_consistency === true;
    const snapshotsScenesContractSceneLoadAllowed = snapshotsScenesContract.scene_load_allowed === true;
    const snapshotsScenesContractSceneSaveAllowed = snapshotsScenesContract.scene_save_allowed === true;
    const snapshotsScenesContractPromotionFamilies = stringList(snapshotsScenesContract.promotion_families_requiring_surface);
    const snapshotsScenesContractActiveFields = stringList(snapshotsScenesContract.required_active_scene_fields);
    const snapshotsScenesContractSnapshotFields = stringList(snapshotsScenesContract.required_snapshot_fields);
    const snapshotsScenesContractCoverage = stringList(snapshotsScenesContract.required_coverage_flags);
    const snapshotsScenesContractFalseSafetyFlags = stringList(snapshotsScenesContract.required_false_safety_flags);
    const snapshotsScenesIntake = asRecord(proofQuality.snapshots_scenes_physical_readback_artifact_intake);
    const snapshotsScenesIntakeAvailable = Boolean(snapshotsScenesIntake.available);
    const snapshotsScenesIntakeStatus = str(snapshotsScenesIntake.intake_status, str(snapshotsScenesIntake.status, 'missing'));
    const snapshotsScenesIntakeSchemaId = str(snapshotsScenesIntake.schema_id, '');
    const snapshotsScenesIntakeSurfaceId = str(snapshotsScenesIntake.surface_id, '');
    const snapshotsScenesIntakeActionId = str(snapshotsScenesIntake.remediation_action_id, '');
    const snapshotsScenesIntakeArtifactCount = Math.trunc(num(snapshotsScenesIntake.artifact_count, 0));
    const snapshotsScenesIntakeValidCount = Math.trunc(num(snapshotsScenesIntake.valid_candidate_count, 0));
    const snapshotsScenesIntakeInvalidCount = Math.trunc(num(snapshotsScenesIntake.invalid_candidate_count, 0));
    const snapshotsScenesIntakeValidSnapshotCount = Math.trunc(num(snapshotsScenesIntake.valid_snapshot_count, 0));
    const snapshotsScenesIntakeCandidate = snapshotsScenesIntake.accepted_as_physical_fixture_candidate === true;
    const snapshotsScenesIntakeOperatorReview = snapshotsScenesIntake.operator_review_required === true;
    const snapshotsScenesIntakeGapRemains = snapshotsScenesIntake.physical_fixture_gap_remains === true;
    const snapshotsScenesIntakeClosureAllowed = snapshotsScenesIntake.fixture_closure_allowed_by_intake === true;
    const snapshotsScenesIntakeSceneLoadAllowed = snapshotsScenesIntake.scene_load_allowed === true;
    const snapshotsScenesIntakeSceneSaveAllowed = snapshotsScenesIntake.scene_save_allowed === true;
    const snapshotsScenesIntakeLiveAuthority = snapshotsScenesIntake.live_authority_granted === true;
    const snapshotsScenesIntakeTargetConsistency = snapshotsScenesIntake.target_consistency === true;
    const snapshotsScenesIntakePromotionFamilies = stringList(snapshotsScenesIntake.promotion_families_requiring_surface);
    const snapshotsScenesIntakeBlockedBy = stringList(snapshotsScenesIntake.blocked_by);
    const snapshotsScenesIntakeSources = stringList(snapshotsScenesIntake.artifact_source_ids);
    const userPresetsContract = asRecord(proofQuality.user_presets_physical_readback_contract);
    const userPresetsContractAvailable = Boolean(userPresetsContract.available);
    const userPresetsContractStatus = str(userPresetsContract.contract_status, str(userPresetsContract.status, 'missing'));
    const userPresetsContractSchemaId = str(userPresetsContract.schema_id, '');
    const userPresetsContractSurfaceId = str(userPresetsContract.surface_id, '');
    const userPresetsContractActionId = str(userPresetsContract.remediation_action_id, '');
    const userPresetsContractPhysicalSurfaceId = str(userPresetsContract.physical_readback_surface_id, '');
    const userPresetsContractPhysicalActionId = str(userPresetsContract.physical_readback_action_id, '');
    const userPresetsContractGapRemains = userPresetsContract.physical_fixture_gap_remains === true;
    const userPresetsContractLiveAuthority = userPresetsContract.live_authority_granted === true;
    const userPresetsContractTargetConsistency = userPresetsContract.target_consistency === true;
    const userPresetsContractPresetRecallAllowed = userPresetsContract.preset_recall_allowed === true;
    const userPresetsContractPresetStoreAllowed = userPresetsContract.preset_store_allowed === true;
    const userPresetsContractPromotionFamilies = stringList(userPresetsContract.promotion_families_requiring_surface);
    const userPresetsContractPresetFields = stringList(userPresetsContract.required_preset_fields);
    const userPresetsContractCoverage = stringList(userPresetsContract.required_coverage_flags);
    const userPresetsContractFalseSafetyFlags = stringList(userPresetsContract.required_false_safety_flags);
    const userPresetsIntake = asRecord(proofQuality.user_presets_physical_readback_artifact_intake);
    const userPresetsIntakeAvailable = Boolean(userPresetsIntake.available);
    const userPresetsIntakeStatus = str(userPresetsIntake.intake_status, str(userPresetsIntake.status, 'missing'));
    const userPresetsIntakeSchemaId = str(userPresetsIntake.schema_id, '');
    const userPresetsIntakeSurfaceId = str(userPresetsIntake.surface_id, '');
    const userPresetsIntakeActionId = str(userPresetsIntake.remediation_action_id, '');
    const userPresetsIntakeArtifactCount = Math.trunc(num(userPresetsIntake.artifact_count, 0));
    const userPresetsIntakeValidCount = Math.trunc(num(userPresetsIntake.valid_candidate_count, 0));
    const userPresetsIntakeInvalidCount = Math.trunc(num(userPresetsIntake.invalid_candidate_count, 0));
    const userPresetsIntakeValidPresetCount = Math.trunc(num(userPresetsIntake.valid_preset_count, 0));
    const userPresetsIntakeCandidate = userPresetsIntake.accepted_as_physical_fixture_candidate === true;
    const userPresetsIntakeOperatorReview = userPresetsIntake.operator_review_required === true;
    const userPresetsIntakeGapRemains = userPresetsIntake.physical_fixture_gap_remains === true;
    const userPresetsIntakeClosureAllowed = userPresetsIntake.fixture_closure_allowed_by_intake === true;
    const userPresetsIntakePresetRecallAllowed = userPresetsIntake.preset_recall_allowed === true;
    const userPresetsIntakePresetStoreAllowed = userPresetsIntake.preset_store_allowed === true;
    const userPresetsIntakeLiveAuthority = userPresetsIntake.live_authority_granted === true;
    const userPresetsIntakeTargetConsistency = userPresetsIntake.target_consistency === true;
    const userPresetsIntakePromotionFamilies = stringList(userPresetsIntake.promotion_families_requiring_surface);
    const userPresetsIntakeBlockedBy = stringList(userPresetsIntake.blocked_by);
    const userPresetsIntakeSources = stringList(userPresetsIntake.artifact_source_ids);
    const dliveStateContract = asRecord(proofQuality.dlive_state_physical_readback_contract);
    const dliveStateContractAvailable = Boolean(dliveStateContract.available);
    const dliveStateContractStatus = str(dliveStateContract.contract_status, str(dliveStateContract.status, 'missing'));
    const dliveStateContractSchemaId = str(dliveStateContract.schema_id, '');
    const dliveStateContractSurfaceId = str(dliveStateContract.surface_id, '');
    const dliveStateContractActionId = str(dliveStateContract.remediation_action_id, '');
    const dliveStateContractPhysicalSurfaceId = str(dliveStateContract.physical_readback_surface_id, '');
    const dliveStateContractPhysicalActionId = str(dliveStateContract.physical_readback_action_id, '');
    const dliveStateContractGapRemains = dliveStateContract.physical_fixture_gap_remains === true;
    const dliveStateContractLiveAuthority = dliveStateContract.live_authority_granted === true;
    const dliveStateContractTargetConsistency = dliveStateContract.target_consistency === true;
    const dliveStateContractMidiSendAllowed = dliveStateContract.midi_send_allowed === true;
    const dliveStateContractTcpSendAllowed = dliveStateContract.tcp_send_allowed === true;
    const dliveStateContractDliveWriteAllowed = dliveStateContract.dlive_write_allowed === true;
    const dliveStateContractDliveApplyAllowed = dliveStateContract.dlive_apply_allowed === true;
    const dliveStateContractPromotionFamilies = stringList(dliveStateContract.promotion_families_requiring_surface);
    const dliveStateContractChannelFields = stringList(dliveStateContract.required_channel_fields);
    const dliveStateContractMidiFields = stringList(dliveStateContract.required_midi_state_fields);
    const dliveStateContractParityFields = stringList(dliveStateContract.required_parity_fields);
    const dliveStateContractCoverage = stringList(dliveStateContract.required_coverage_flags);
    const dliveStateContractFalseSafetyFlags = stringList(dliveStateContract.required_false_safety_flags);
    const dliveStateIntake = asRecord(proofQuality.dlive_state_physical_readback_artifact_intake);
    const dliveStateIntakeAvailable = Boolean(dliveStateIntake.available);
    const dliveStateIntakeStatus = str(dliveStateIntake.intake_status, str(dliveStateIntake.status, 'missing'));
    const dliveStateIntakeSchemaId = str(dliveStateIntake.schema_id, '');
    const dliveStateIntakeSurfaceId = str(dliveStateIntake.surface_id, '');
    const dliveStateIntakeActionId = str(dliveStateIntake.remediation_action_id, '');
    const dliveStateIntakeArtifactCount = Math.trunc(num(dliveStateIntake.artifact_count, 0));
    const dliveStateIntakeValidCount = Math.trunc(num(dliveStateIntake.valid_candidate_count, 0));
    const dliveStateIntakeInvalidCount = Math.trunc(num(dliveStateIntake.invalid_candidate_count, 0));
    const dliveStateIntakeValidChannelCount = Math.trunc(num(dliveStateIntake.valid_channel_count, 0));
    const dliveStateIntakeCandidate = dliveStateIntake.accepted_as_physical_fixture_candidate === true;
    const dliveStateIntakeOperatorReview = dliveStateIntake.operator_review_required === true;
    const dliveStateIntakeGapRemains = dliveStateIntake.physical_fixture_gap_remains === true;
    const dliveStateIntakeClosureAllowed = dliveStateIntake.fixture_closure_allowed_by_intake === true;
    const dliveStateIntakeMidiSendAllowed = dliveStateIntake.midi_send_allowed === true;
    const dliveStateIntakeTcpSendAllowed = dliveStateIntake.tcp_send_allowed === true;
    const dliveStateIntakeDliveWriteAllowed = dliveStateIntake.dlive_write_allowed === true;
    const dliveStateIntakeDliveApplyAllowed = dliveStateIntake.dlive_apply_allowed === true;
    const dliveStateIntakeLiveAuthority = dliveStateIntake.live_authority_granted === true;
    const dliveStateIntakeTargetConsistency = dliveStateIntake.target_consistency === true;
    const dliveStateIntakePromotionFamilies = stringList(dliveStateIntake.promotion_families_requiring_surface);
    const dliveStateIntakeBlockedBy = stringList(dliveStateIntake.blocked_by);
    const dliveStateIntakeSources = stringList(dliveStateIntake.artifact_source_ids);
    const artifactReviewQueue = asRecord(proofQuality.read_surface_physical_readback_artifact_review_queue);
    const artifactReviewQueueAvailable = Boolean(artifactReviewQueue.available);
    const artifactReviewQueueStatus = str(artifactReviewQueue.queue_status, str(artifactReviewQueue.status, 'missing'));
    const artifactReviewQueueSurfaceCount = Math.trunc(num(artifactReviewQueue.intake_surface_count, 0));
    const artifactReviewQueueAvailableIntakeCount = Math.trunc(num(artifactReviewQueue.available_intake_count, 0));
    const artifactReviewQueueArtifactCount = Math.trunc(num(artifactReviewQueue.artifact_count, 0));
    const artifactReviewQueueValidCount = Math.trunc(num(artifactReviewQueue.valid_candidate_count, 0));
    const artifactReviewQueueInvalidCount = Math.trunc(num(artifactReviewQueue.invalid_candidate_count, 0));
    const artifactReviewQueueReviewCandidateCount = Math.trunc(num(artifactReviewQueue.review_candidate_count, 0));
    const artifactReviewQueueMissingCount = Math.trunc(num(artifactReviewQueue.missing_artifact_surface_count, 0));
    const artifactReviewQueueUnsafeCount = Math.trunc(num(artifactReviewQueue.unsafe_authority_claim_count, 0));
    const artifactReviewQueueTargetInconsistentCount = Math.trunc(num(artifactReviewQueue.target_inconsistent_surface_count, 0));
    const artifactReviewQueueReviewAvailable = artifactReviewQueue.review_candidates_available === true;
    const artifactReviewQueueOperatorReview = artifactReviewQueue.operator_review_required === true;
    const artifactReviewQueueGapRemains = artifactReviewQueue.physical_fixture_gap_remains === true;
    const artifactReviewQueueClosureAllowed = artifactReviewQueue.fixture_closure_allowed_by_queue === true;
    const artifactReviewQueueLiveAuthority = artifactReviewQueue.live_authority_granted === true;
    const artifactReviewQueueFullAuto = artifactReviewQueue.full_autonomous_ready === true;
    const artifactReviewQueueMissingSurfaces = stringList(artifactReviewQueue.missing_artifact_surface_ids);
    const artifactReviewQueueCandidateSurfaces = stringList(artifactReviewQueue.review_candidate_surface_ids);
    const artifactReviewQueueInvalidSurfaces = stringList(artifactReviewQueue.invalid_surface_ids);
    const artifactReviewQueueUnsafeSurfaces = stringList(artifactReviewQueue.unsafe_authority_surface_ids);
    const artifactReviewQueueTargetInconsistentSurfaces = stringList(artifactReviewQueue.target_inconsistent_surface_ids);
    const artifactReviewQueueMethodIds = stringList(artifactReviewQueue.queue_item_method_ids);
    const artifactReviewQueueBlockedBy = stringList(artifactReviewQueue.blocked_by);
    const fixtureClosurePlan = asRecord(proofQuality.read_surface_physical_readback_fixture_closure_plan);
    const fixtureClosurePlanAvailable = Boolean(fixtureClosurePlan.available);
    const fixtureClosurePlanStatus = str(fixtureClosurePlan.plan_status, str(fixtureClosurePlan.status, 'missing'));
    const fixtureClosurePlanReviewCandidateCount = Math.trunc(num(fixtureClosurePlan.review_candidate_count, 0));
    const fixtureClosurePlanAcceptanceRecordCount = Math.trunc(num(fixtureClosurePlan.acceptance_record_count, 0));
    const fixtureClosurePlanAcceptedRecordCount = Math.trunc(num(fixtureClosurePlan.accepted_record_count, 0));
    const fixtureClosurePlanRejectedRecordCount = Math.trunc(num(fixtureClosurePlan.rejected_record_count, 0));
    const fixtureClosurePlanInvalidRecordCount = Math.trunc(num(fixtureClosurePlan.invalid_record_count, 0));
    const fixtureClosurePlanUnreviewedCount = Math.trunc(num(fixtureClosurePlan.unreviewed_candidate_count, 0));
    const fixtureClosurePlanCandidateCount = Math.trunc(num(fixtureClosurePlan.closure_candidate_count, 0));
    const fixtureClosurePlanCandidateAvailable = fixtureClosurePlan.manual_fixture_closure_candidate_available === true;
    const fixtureClosurePlanManualRequired = fixtureClosurePlan.manual_fixture_closure_required === true;
    const fixtureClosurePlanOperatorReview = fixtureClosurePlan.operator_review_required === true;
    const fixtureClosurePlanGapRemains = fixtureClosurePlan.physical_fixture_gap_remains === true;
    const fixtureClosurePlanUpdateAllowed = fixtureClosurePlan.fixture_update_allowed_by_plan === true;
    const fixtureClosurePlanClosureAllowed = fixtureClosurePlan.fixture_closure_allowed_by_plan === true;
    const fixtureClosurePlanLiveAuthority = fixtureClosurePlan.live_authority_granted === true;
    const fixtureClosurePlanFullAuto = fixtureClosurePlan.full_autonomous_ready === true;
    const fixtureClosurePlanCandidateSurfaces = stringList(fixtureClosurePlan.closure_candidate_surface_ids);
    const fixtureClosurePlanRejectedSurfaces = stringList(fixtureClosurePlan.rejected_surface_ids);
    const fixtureClosurePlanInvalidRecordIds = stringList(fixtureClosurePlan.invalid_record_ids);
    const fixtureClosurePlanUnreviewedSurfaces = stringList(fixtureClosurePlan.unreviewed_candidate_surface_ids);
    const fixtureClosurePlanBlockedBy = stringList(fixtureClosurePlan.blocked_by);
    const freshnessText = available
      ? `age:${Math.round(ageSec)}s stale_after:${Math.round(staleAfterSec)}s stale:${stale}`
      : '';
    const proofQualityText = proofQualityAvailable
      ? `read_surface_gaps:${physicalGapSurfaceCount}; fixture_paths:${declaredFixturePathCount}; test_names:${declaredTestNameCount}; summary_recomputed:${summaryRecomputed}; declarations:${fixtureDeclarationValidation}; contract_gap:${contractGapConsistency}; gap_surfaces:${physicalGapSurfaceIds.join(',') || 'none'}; next_readback:${nextPhysicalReadbackSurfaceId || 'none'}; source_next_readback:${sourceNextPhysicalReadbackSurfaceId || 'none'}; next_readback_gate:${nextPhysicalReadbackTargetConsistent}; capture_plan:${readSurfaceCapturePlanStatus || 'missing'}; capture_steps:${readSurfaceCapturePlanStepCount}; capture_execution:${readSurfaceCapturePlanExecutionAllowed}; capture_target_gate:${readSurfaceCapturePlanTargetConsistency}; raw_contract:${rawMixerContractStatus || 'missing'}; raw_contract_gap:${rawMixerContractGapRemains}; raw_contract_live_authority:${rawMixerContractLiveAuthority}; raw_contract_target_gate:${rawMixerContractTargetConsistency}; raw_intake:${rawMixerIntakeStatus || 'missing'}; raw_intake_artifacts:${rawMixerIntakeArtifactCount}; raw_intake_valid:${rawMixerIntakeValidCount}; raw_intake_invalid:${rawMixerIntakeInvalidCount}; raw_intake_candidate:${rawMixerIntakeCandidate}; raw_intake_closure:${rawMixerIntakeClosureAllowed}; raw_intake_target_gate:${rawMixerIntakeTargetConsistency}; channel_contract:${channelBasicContractStatus || 'missing'}; channel_contract_gap:${channelBasicContractGapRemains}; channel_contract_live_authority:${channelBasicContractLiveAuthority}; channel_contract_target_gate:${channelBasicContractTargetConsistency}; channel_intake:${channelBasicIntakeStatus || 'missing'}; channel_intake_artifacts:${channelBasicIntakeArtifactCount}; channel_intake_valid:${channelBasicIntakeValidCount}; channel_intake_invalid:${channelBasicIntakeInvalidCount}; channel_intake_valid_channels:${channelBasicIntakeValidChannelCount}; channel_intake_candidate:${channelBasicIntakeCandidate}; channel_intake_gap:${channelBasicIntakeGapRemains}; channel_intake_closure:${channelBasicIntakeClosureAllowed}; channel_intake_live_authority:${channelBasicIntakeLiveAuthority}; channel_intake_target_gate:${channelBasicIntakeTargetConsistency}; processing_contract:${channelProcessingContractStatus || 'missing'}; processing_contract_gap:${channelProcessingContractGapRemains}; processing_contract_live_authority:${channelProcessingContractLiveAuthority}; processing_contract_target_gate:${channelProcessingContractTargetConsistency}; processing_intake:${channelProcessingIntakeStatus || 'missing'}; processing_intake_artifacts:${channelProcessingIntakeArtifactCount}; processing_intake_valid:${channelProcessingIntakeValidCount}; processing_intake_invalid:${channelProcessingIntakeInvalidCount}; processing_intake_valid_channels:${channelProcessingIntakeValidChannelCount}; processing_intake_candidate:${channelProcessingIntakeCandidate}; processing_intake_gap:${channelProcessingIntakeGapRemains}; processing_intake_closure:${channelProcessingIntakeClosureAllowed}; processing_intake_live_authority:${channelProcessingIntakeLiveAuthority}; processing_intake_target_gate:${channelProcessingIntakeTargetConsistency}; bus_contract:${busMainDcaContractStatus || 'missing'}; bus_contract_gap:${busMainDcaContractGapRemains}; bus_contract_live_authority:${busMainDcaContractLiveAuthority}; bus_contract_target_gate:${busMainDcaContractTargetConsistency}; bus_intake:${busMainDcaIntakeStatus || 'missing'}; bus_intake_artifacts:${busMainDcaIntakeArtifactCount}; bus_intake_valid:${busMainDcaIntakeValidCount}; bus_intake_invalid:${busMainDcaIntakeInvalidCount}; bus_intake_valid_buses:${busMainDcaIntakeValidBusCount}; bus_intake_valid_mains:${busMainDcaIntakeValidMainCount}; bus_intake_valid_dcas:${busMainDcaIntakeValidDcaCount}; bus_intake_candidate:${busMainDcaIntakeCandidate}; bus_intake_gap:${busMainDcaIntakeGapRemains}; bus_intake_closure:${busMainDcaIntakeClosureAllowed}; bus_intake_live_authority:${busMainDcaIntakeLiveAuthority}; bus_intake_target_gate:${busMainDcaIntakeTargetConsistency}; matrix_contract:${matrixStateContractStatus || 'missing'}; matrix_contract_gap:${matrixStateContractGapRemains}; matrix_contract_live_authority:${matrixStateContractLiveAuthority}; matrix_contract_target_gate:${matrixStateContractTargetConsistency}; matrix_intake:${matrixStateIntakeStatus || 'missing'}; matrix_intake_valid_matrices:${matrixStateIntakeValidMatrixCount}; matrix_intake_gap:${matrixStateIntakeGapRemains}; matrix_intake_live_authority:${matrixStateIntakeLiveAuthority}; matrix_intake_target_gate:${matrixStateIntakeTargetConsistency}; fx_contract:${fxSlotsInsertsContractStatus || 'missing'}; fx_contract_gap:${fxSlotsInsertsContractGapRemains}; fx_contract_live_authority:${fxSlotsInsertsContractLiveAuthority}; fx_contract_target_gate:${fxSlotsInsertsContractTargetConsistency}; fx_intake:${fxSlotsInsertsIntakeStatus || 'missing'}; fx_intake_valid_slots:${fxSlotsInsertsIntakeValidFxSlotCount}; fx_intake_valid_inserts:${fxSlotsInsertsIntakeValidInsertCount}; fx_intake_gap:${fxSlotsInsertsIntakeGapRemains}; fx_intake_live_authority:${fxSlotsInsertsIntakeLiveAuthority}; fx_intake_target_gate:${fxSlotsInsertsIntakeTargetConsistency}; routing_contract:${routingSendsContractStatus || 'missing'}; routing_contract_gap:${routingSendsContractGapRemains}; routing_contract_live_authority:${routingSendsContractLiveAuthority}; routing_contract_target_gate:${routingSendsContractTargetConsistency}; scenes_contract:${snapshotsScenesContractStatus || 'missing'}; scenes_contract_gap:${snapshotsScenesContractGapRemains}; scenes_contract_live_authority:${snapshotsScenesContractLiveAuthority}; scenes_contract_target_gate:${snapshotsScenesContractTargetConsistency}; presets_contract:${userPresetsContractStatus || 'missing'}; presets_contract_gap:${userPresetsContractGapRemains}; presets_contract_live_authority:${userPresetsContractLiveAuthority}; presets_contract_target_gate:${userPresetsContractTargetConsistency}; presets_intake:${userPresetsIntakeStatus || 'missing'}; presets_intake_artifacts:${userPresetsIntakeArtifactCount}; presets_intake_valid:${userPresetsIntakeValidCount}; presets_intake_invalid:${userPresetsIntakeInvalidCount}; presets_intake_valid_presets:${userPresetsIntakeValidPresetCount}; presets_intake_candidate:${userPresetsIntakeCandidate}; presets_intake_gap:${userPresetsIntakeGapRemains}; presets_intake_closure:${userPresetsIntakeClosureAllowed}; presets_intake_preset_recall:${userPresetsIntakePresetRecallAllowed}; presets_intake_preset_store:${userPresetsIntakePresetStoreAllowed}; presets_intake_live_authority:${userPresetsIntakeLiveAuthority}; presets_intake_target_gate:${userPresetsIntakeTargetConsistency}; dlive_contract:${dliveStateContractStatus || 'missing'}; dlive_contract_gap:${dliveStateContractGapRemains}; dlive_contract_live_authority:${dliveStateContractLiveAuthority}; dlive_contract_target_gate:${dliveStateContractTargetConsistency}; dlive_intake:${dliveStateIntakeStatus || 'missing'}; dlive_intake_artifacts:${dliveStateIntakeArtifactCount}; dlive_intake_valid:${dliveStateIntakeValidCount}; dlive_intake_invalid:${dliveStateIntakeInvalidCount}; dlive_intake_valid_channels:${dliveStateIntakeValidChannelCount}; dlive_intake_candidate:${dliveStateIntakeCandidate}; dlive_intake_gap:${dliveStateIntakeGapRemains}; dlive_intake_closure:${dliveStateIntakeClosureAllowed}; dlive_intake_midi_send:${dliveStateIntakeMidiSendAllowed}; dlive_intake_tcp_send:${dliveStateIntakeTcpSendAllowed}; dlive_intake_dlive_write:${dliveStateIntakeDliveWriteAllowed}; dlive_intake_dlive_apply:${dliveStateIntakeDliveApplyAllowed}; dlive_intake_live_authority:${dliveStateIntakeLiveAuthority}; dlive_intake_target_gate:${dliveStateIntakeTargetConsistency}`
      : 'read_surface_quality:missing';
    const artifactReviewQueueText = proofQualityAvailable
      ? `artifact_queue:${artifactReviewQueueStatus || 'missing'}; artifact_queue_surfaces:${artifactReviewQueueSurfaceCount}; artifact_queue_available_intakes:${artifactReviewQueueAvailableIntakeCount}; artifact_queue_artifacts:${artifactReviewQueueArtifactCount}; artifact_queue_valid:${artifactReviewQueueValidCount}; artifact_queue_invalid:${artifactReviewQueueInvalidCount}; artifact_queue_review_candidates:${artifactReviewQueueReviewCandidateCount}; artifact_queue_missing:${artifactReviewQueueMissingCount}; artifact_queue_unsafe:${artifactReviewQueueUnsafeCount}; artifact_queue_target_inconsistent:${artifactReviewQueueTargetInconsistentCount}; artifact_queue_review_available:${artifactReviewQueueReviewAvailable}; artifact_queue_operator_review:${artifactReviewQueueOperatorReview}; artifact_queue_gap:${artifactReviewQueueGapRemains}; artifact_queue_closure:${artifactReviewQueueClosureAllowed}; artifact_queue_live_authority:${artifactReviewQueueLiveAuthority}; artifact_queue_full_auto:${artifactReviewQueueFullAuto}`
      : 'artifact_queue:missing';
    const fixtureClosurePlanText = proofQualityAvailable
      ? `fixture_closure_plan:${fixtureClosurePlanStatus || 'missing'}; fixture_closure_candidates:${fixtureClosurePlanCandidateCount}; fixture_closure_accepted:${fixtureClosurePlanAcceptedRecordCount}; fixture_closure_rejected:${fixtureClosurePlanRejectedRecordCount}; fixture_closure_invalid:${fixtureClosurePlanInvalidRecordCount}; fixture_closure_unreviewed:${fixtureClosurePlanUnreviewedCount}; fixture_closure_candidate_available:${fixtureClosurePlanCandidateAvailable}; fixture_update_allowed:${fixtureClosurePlanUpdateAllowed}; fixture_closure_allowed:${fixtureClosurePlanClosureAllowed}; fixture_closure_gap:${fixtureClosurePlanGapRemains}; fixture_closure_live_authority:${fixtureClosurePlanLiveAuthority}; fixture_closure_full_auto:${fixtureClosurePlanFullAuto}`
      : 'fixture_closure_plan:missing';
    const rowStatus = available
      ? invalidCount > 0 ? 'artifact_review_required' : stale ? 'artifact_stale' : 'artifact_available'
      : 'no_current_evidence';
    const rows: ModuleStatusRecord[] = [{
      moduleId: 'supervised_proof_evidence',
      label: 'Supervised proof evidence',
      domain: 'supervised_proof_evidence',
      status: rowStatus,
      entrypoint: 'supervised_proof_artifact_report',
      module: reportPath,
      uiSurface: 'Supervised proof evidence source',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${available}`,
        `overlay:${overlayStatus}`,
        `artifacts:${artifactCount}`,
        `invalid:${invalidCount}`,
        `stale:${stale}`,
        `read_surface_quality:${proofQualityAvailable}`,
        `read_surface_gaps:${physicalGapSurfaceCount}`,
        `read_surface_gap_ids:${physicalGapSurfaceText}`,
        `next_readback_target:${nextPhysicalReadbackSurfaceId}`,
        `source_next_readback_target:${sourceNextPhysicalReadbackSurfaceId}`,
        `next_readback_gate:${nextPhysicalReadbackTargetConsistent}`,
        `capture_plan_available:${readSurfaceCapturePlanAvailable}`,
        `capture_plan_status:${readSurfaceCapturePlanStatus}`,
        `capture_plan_steps:${readSurfaceCapturePlanStepCount}`,
        `capture_plan_next_surface:${readSurfaceCapturePlanSurfaceId}`,
        `capture_plan_execution_allowed:${readSurfaceCapturePlanExecutionAllowed}`,
        `capture_plan_target_gate:${readSurfaceCapturePlanTargetConsistency}`,
        `fixture_tests:${declaredTestNameCount}`,
        `contract_gap:${contractGapConsistency}`,
        `raw_contract_available:${rawMixerContractAvailable}`,
        `raw_contract_status:${rawMixerContractStatus}`,
        `raw_contract_schema:${rawMixerContractSchemaId}`,
        `raw_contract_gap:${rawMixerContractGapRemains}`,
        `raw_contract_live_authority:${rawMixerContractLiveAuthority}`,
        `raw_contract_target_gate:${rawMixerContractTargetConsistency}`,
        `raw_intake_available:${rawMixerIntakeAvailable}`,
        `raw_intake_status:${rawMixerIntakeStatus}`,
        `raw_intake_artifacts:${rawMixerIntakeArtifactCount}`,
        `raw_intake_valid:${rawMixerIntakeValidCount}`,
        `raw_intake_invalid:${rawMixerIntakeInvalidCount}`,
        `raw_intake_candidate:${rawMixerIntakeCandidate}`,
        `raw_intake_gap:${rawMixerIntakeGapRemains}`,
        `raw_intake_closure_allowed:${rawMixerIntakeClosureAllowed}`,
        `raw_intake_live_authority:${rawMixerIntakeLiveAuthority}`,
        `raw_intake_target_gate:${rawMixerIntakeTargetConsistency}`,
        `channel_contract_available:${channelBasicContractAvailable}`,
        `channel_contract_status:${channelBasicContractStatus}`,
        `channel_contract_schema:${channelBasicContractSchemaId}`,
        `channel_contract_gap:${channelBasicContractGapRemains}`,
        `channel_contract_live_authority:${channelBasicContractLiveAuthority}`,
        `channel_contract_target_gate:${channelBasicContractTargetConsistency}`,
        `channel_intake_available:${channelBasicIntakeAvailable}`,
        `channel_intake_status:${channelBasicIntakeStatus}`,
        `channel_intake_artifacts:${channelBasicIntakeArtifactCount}`,
        `channel_intake_valid:${channelBasicIntakeValidCount}`,
        `channel_intake_invalid:${channelBasicIntakeInvalidCount}`,
        `channel_intake_valid_channels:${channelBasicIntakeValidChannelCount}`,
        `channel_intake_candidate:${channelBasicIntakeCandidate}`,
        `channel_intake_gap:${channelBasicIntakeGapRemains}`,
        `channel_intake_closure_allowed:${channelBasicIntakeClosureAllowed}`,
        `channel_intake_live_authority:${channelBasicIntakeLiveAuthority}`,
        `channel_intake_target_gate:${channelBasicIntakeTargetConsistency}`,
        `processing_contract_available:${channelProcessingContractAvailable}`,
        `processing_contract_status:${channelProcessingContractStatus}`,
        `processing_contract_schema:${channelProcessingContractSchemaId}`,
        `processing_contract_gap:${channelProcessingContractGapRemains}`,
        `processing_contract_live_authority:${channelProcessingContractLiveAuthority}`,
        `processing_contract_target_gate:${channelProcessingContractTargetConsistency}`,
        `processing_intake_available:${channelProcessingIntakeAvailable}`,
        `processing_intake_status:${channelProcessingIntakeStatus}`,
        `processing_intake_artifacts:${channelProcessingIntakeArtifactCount}`,
        `processing_intake_valid:${channelProcessingIntakeValidCount}`,
        `processing_intake_invalid:${channelProcessingIntakeInvalidCount}`,
        `processing_intake_valid_channels:${channelProcessingIntakeValidChannelCount}`,
        `processing_intake_candidate:${channelProcessingIntakeCandidate}`,
        `processing_intake_gap:${channelProcessingIntakeGapRemains}`,
        `processing_intake_closure_allowed:${channelProcessingIntakeClosureAllowed}`,
        `processing_intake_live_authority:${channelProcessingIntakeLiveAuthority}`,
        `processing_intake_target_gate:${channelProcessingIntakeTargetConsistency}`,
        `bus_contract_available:${busMainDcaContractAvailable}`,
        `bus_contract_status:${busMainDcaContractStatus}`,
        `bus_contract_schema:${busMainDcaContractSchemaId}`,
        `bus_contract_gap:${busMainDcaContractGapRemains}`,
        `bus_contract_live_authority:${busMainDcaContractLiveAuthority}`,
        `bus_contract_target_gate:${busMainDcaContractTargetConsistency}`,
        `bus_intake_available:${busMainDcaIntakeAvailable}`,
        `bus_intake_status:${busMainDcaIntakeStatus}`,
        `bus_intake_artifacts:${busMainDcaIntakeArtifactCount}`,
        `bus_intake_valid:${busMainDcaIntakeValidCount}`,
        `bus_intake_invalid:${busMainDcaIntakeInvalidCount}`,
        `bus_intake_valid_buses:${busMainDcaIntakeValidBusCount}`,
        `bus_intake_valid_mains:${busMainDcaIntakeValidMainCount}`,
        `bus_intake_valid_dcas:${busMainDcaIntakeValidDcaCount}`,
        `bus_intake_candidate:${busMainDcaIntakeCandidate}`,
        `bus_intake_gap:${busMainDcaIntakeGapRemains}`,
        `bus_intake_closure_allowed:${busMainDcaIntakeClosureAllowed}`,
        `bus_intake_live_authority:${busMainDcaIntakeLiveAuthority}`,
        `bus_intake_target_gate:${busMainDcaIntakeTargetConsistency}`,
        `matrix_contract_available:${matrixStateContractAvailable}`,
        `matrix_contract_status:${matrixStateContractStatus}`,
        `matrix_contract_schema:${matrixStateContractSchemaId}`,
        `matrix_contract_gap:${matrixStateContractGapRemains}`,
        `matrix_contract_live_authority:${matrixStateContractLiveAuthority}`,
        `matrix_contract_target_gate:${matrixStateContractTargetConsistency}`,
        `matrix_intake_available:${matrixStateIntakeAvailable}`,
        `matrix_intake_status:${matrixStateIntakeStatus}`,
        `matrix_intake_artifacts:${matrixStateIntakeArtifactCount}`,
        `matrix_intake_valid:${matrixStateIntakeValidCount}`,
        `matrix_intake_invalid:${matrixStateIntakeInvalidCount}`,
        `matrix_intake_valid_matrices:${matrixStateIntakeValidMatrixCount}`,
        `matrix_intake_candidate:${matrixStateIntakeCandidate}`,
        `matrix_intake_gap:${matrixStateIntakeGapRemains}`,
        `matrix_intake_closure_allowed:${matrixStateIntakeClosureAllowed}`,
        `matrix_intake_live_authority:${matrixStateIntakeLiveAuthority}`,
        `matrix_intake_target_gate:${matrixStateIntakeTargetConsistency}`,
        `fx_contract_available:${fxSlotsInsertsContractAvailable}`,
        `fx_contract_status:${fxSlotsInsertsContractStatus}`,
        `fx_contract_schema:${fxSlotsInsertsContractSchemaId}`,
        `fx_contract_gap:${fxSlotsInsertsContractGapRemains}`,
        `fx_contract_live_authority:${fxSlotsInsertsContractLiveAuthority}`,
        `fx_contract_target_gate:${fxSlotsInsertsContractTargetConsistency}`,
        `fx_intake_available:${fxSlotsInsertsIntakeAvailable}`,
        `fx_intake_status:${fxSlotsInsertsIntakeStatus}`,
        `fx_intake_artifacts:${fxSlotsInsertsIntakeArtifactCount}`,
        `fx_intake_valid:${fxSlotsInsertsIntakeValidCount}`,
        `fx_intake_invalid:${fxSlotsInsertsIntakeInvalidCount}`,
        `fx_intake_valid_slots:${fxSlotsInsertsIntakeValidFxSlotCount}`,
        `fx_intake_valid_inserts:${fxSlotsInsertsIntakeValidInsertCount}`,
        `fx_intake_candidate:${fxSlotsInsertsIntakeCandidate}`,
        `fx_intake_gap:${fxSlotsInsertsIntakeGapRemains}`,
        `fx_intake_closure_allowed:${fxSlotsInsertsIntakeClosureAllowed}`,
        `fx_intake_live_authority:${fxSlotsInsertsIntakeLiveAuthority}`,
        `fx_intake_target_gate:${fxSlotsInsertsIntakeTargetConsistency}`,
        `routing_contract_available:${routingSendsContractAvailable}`,
        `routing_contract_status:${routingSendsContractStatus}`,
        `routing_contract_schema:${routingSendsContractSchemaId}`,
        `routing_contract_gap:${routingSendsContractGapRemains}`,
        `routing_contract_live_authority:${routingSendsContractLiveAuthority}`,
        `routing_contract_target_gate:${routingSendsContractTargetConsistency}`,
        `routing_intake_available:${routingSendsIntakeAvailable}`,
        `routing_intake_status:${routingSendsIntakeStatus}`,
        `routing_intake_artifacts:${routingSendsIntakeArtifactCount}`,
        `routing_intake_valid:${routingSendsIntakeValidCount}`,
        `routing_intake_invalid:${routingSendsIntakeInvalidCount}`,
        `routing_intake_valid_static_routes:${routingSendsIntakeValidStaticRouteCount}`,
        `routing_intake_valid_bus_sends:${routingSendsIntakeValidBusSendCount}`,
        `routing_intake_valid_main_sends:${routingSendsIntakeValidMainSendCount}`,
        `routing_intake_valid_matrix_sends:${routingSendsIntakeValidMatrixSendCount}`,
        `routing_intake_valid_output_routes:${routingSendsIntakeValidOutputRouteCount}`,
        `routing_intake_candidate:${routingSendsIntakeCandidate}`,
        `routing_intake_gap:${routingSendsIntakeGapRemains}`,
        `routing_intake_closure_allowed:${routingSendsIntakeClosureAllowed}`,
        `routing_intake_live_authority:${routingSendsIntakeLiveAuthority}`,
        `routing_intake_target_gate:${routingSendsIntakeTargetConsistency}`,
        `scenes_contract_available:${snapshotsScenesContractAvailable}`,
        `scenes_contract_status:${snapshotsScenesContractStatus}`,
        `scenes_contract_schema:${snapshotsScenesContractSchemaId}`,
        `scenes_contract_gap:${snapshotsScenesContractGapRemains}`,
        `scenes_contract_live_authority:${snapshotsScenesContractLiveAuthority}`,
        `scenes_contract_target_gate:${snapshotsScenesContractTargetConsistency}`,
        `scenes_intake_available:${snapshotsScenesIntakeAvailable}`,
        `scenes_intake_status:${snapshotsScenesIntakeStatus}`,
        `scenes_intake_artifacts:${snapshotsScenesIntakeArtifactCount}`,
        `scenes_intake_valid:${snapshotsScenesIntakeValidCount}`,
        `scenes_intake_invalid:${snapshotsScenesIntakeInvalidCount}`,
        `scenes_intake_valid_snapshots:${snapshotsScenesIntakeValidSnapshotCount}`,
        `scenes_intake_candidate:${snapshotsScenesIntakeCandidate}`,
        `scenes_intake_gap:${snapshotsScenesIntakeGapRemains}`,
        `scenes_intake_closure_allowed:${snapshotsScenesIntakeClosureAllowed}`,
        `scenes_intake_scene_load_allowed:${snapshotsScenesIntakeSceneLoadAllowed}`,
        `scenes_intake_scene_save_allowed:${snapshotsScenesIntakeSceneSaveAllowed}`,
        `scenes_intake_live_authority:${snapshotsScenesIntakeLiveAuthority}`,
        `scenes_intake_target_gate:${snapshotsScenesIntakeTargetConsistency}`,
        `presets_contract_available:${userPresetsContractAvailable}`,
        `presets_contract_status:${userPresetsContractStatus}`,
        `presets_contract_schema:${userPresetsContractSchemaId}`,
        `presets_contract_gap:${userPresetsContractGapRemains}`,
        `presets_contract_live_authority:${userPresetsContractLiveAuthority}`,
        `presets_contract_target_gate:${userPresetsContractTargetConsistency}`,
        `presets_intake_available:${userPresetsIntakeAvailable}`,
        `presets_intake_status:${userPresetsIntakeStatus}`,
        `presets_intake_artifacts:${userPresetsIntakeArtifactCount}`,
        `presets_intake_valid:${userPresetsIntakeValidCount}`,
        `presets_intake_invalid:${userPresetsIntakeInvalidCount}`,
        `presets_intake_valid_presets:${userPresetsIntakeValidPresetCount}`,
        `presets_intake_candidate:${userPresetsIntakeCandidate}`,
        `presets_intake_gap:${userPresetsIntakeGapRemains}`,
        `presets_intake_closure_allowed:${userPresetsIntakeClosureAllowed}`,
        `presets_intake_preset_recall_allowed:${userPresetsIntakePresetRecallAllowed}`,
        `presets_intake_preset_store_allowed:${userPresetsIntakePresetStoreAllowed}`,
        `presets_intake_live_authority:${userPresetsIntakeLiveAuthority}`,
        `presets_intake_target_gate:${userPresetsIntakeTargetConsistency}`,
        `dlive_contract_available:${dliveStateContractAvailable}`,
        `dlive_contract_status:${dliveStateContractStatus}`,
        `dlive_contract_schema:${dliveStateContractSchemaId}`,
        `dlive_contract_gap:${dliveStateContractGapRemains}`,
        `dlive_contract_live_authority:${dliveStateContractLiveAuthority}`,
        `dlive_contract_target_gate:${dliveStateContractTargetConsistency}`,
        `dlive_intake_available:${dliveStateIntakeAvailable}`,
        `dlive_intake_status:${dliveStateIntakeStatus}`,
        `dlive_intake_artifacts:${dliveStateIntakeArtifactCount}`,
        `dlive_intake_valid:${dliveStateIntakeValidCount}`,
        `dlive_intake_invalid:${dliveStateIntakeInvalidCount}`,
        `dlive_intake_valid_channels:${dliveStateIntakeValidChannelCount}`,
        `dlive_intake_candidate:${dliveStateIntakeCandidate}`,
        `dlive_intake_gap:${dliveStateIntakeGapRemains}`,
        `dlive_intake_closure_allowed:${dliveStateIntakeClosureAllowed}`,
        `dlive_intake_midi_send_allowed:${dliveStateIntakeMidiSendAllowed}`,
        `dlive_intake_tcp_send_allowed:${dliveStateIntakeTcpSendAllowed}`,
        `dlive_intake_dlive_write_allowed:${dliveStateIntakeDliveWriteAllowed}`,
        `dlive_intake_dlive_apply_allowed:${dliveStateIntakeDliveApplyAllowed}`,
        `dlive_intake_live_authority:${dliveStateIntakeLiveAuthority}`,
        `dlive_intake_target_gate:${dliveStateIntakeTargetConsistency}`,
        `artifact_queue_available:${artifactReviewQueueAvailable}`,
        `artifact_queue_status:${artifactReviewQueueStatus}`,
        `artifact_queue_surfaces:${artifactReviewQueueSurfaceCount}`,
        `artifact_queue_available_intakes:${artifactReviewQueueAvailableIntakeCount}`,
        `artifact_queue_artifacts:${artifactReviewQueueArtifactCount}`,
        `artifact_queue_valid:${artifactReviewQueueValidCount}`,
        `artifact_queue_invalid:${artifactReviewQueueInvalidCount}`,
        `artifact_queue_review_candidates:${artifactReviewQueueReviewCandidateCount}`,
        `artifact_queue_missing:${artifactReviewQueueMissingCount}`,
        `artifact_queue_unsafe:${artifactReviewQueueUnsafeCount}`,
        `artifact_queue_target_inconsistent:${artifactReviewQueueTargetInconsistentCount}`,
        `artifact_queue_review_available:${artifactReviewQueueReviewAvailable}`,
        `artifact_queue_closure_allowed:${artifactReviewQueueClosureAllowed}`,
        `artifact_queue_live_authority:${artifactReviewQueueLiveAuthority}`,
        `artifact_queue_full_auto:${artifactReviewQueueFullAuto}`,
        `fixture_closure_plan_available:${fixtureClosurePlanAvailable}`,
        `fixture_closure_plan_status:${fixtureClosurePlanStatus}`,
        `fixture_closure_plan_review_candidates:${fixtureClosurePlanReviewCandidateCount}`,
        `fixture_closure_plan_acceptance_records:${fixtureClosurePlanAcceptanceRecordCount}`,
        `fixture_closure_plan_accepted:${fixtureClosurePlanAcceptedRecordCount}`,
        `fixture_closure_plan_rejected:${fixtureClosurePlanRejectedRecordCount}`,
        `fixture_closure_plan_invalid:${fixtureClosurePlanInvalidRecordCount}`,
        `fixture_closure_plan_unreviewed:${fixtureClosurePlanUnreviewedCount}`,
        `fixture_closure_plan_candidates:${fixtureClosurePlanCandidateCount}`,
        `fixture_closure_plan_candidate_available:${fixtureClosurePlanCandidateAvailable}`,
        `fixture_update_allowed:${fixtureClosurePlanUpdateAllowed}`,
        `fixture_closure_allowed:${fixtureClosurePlanClosureAllowed}`,
        `fixture_closure_plan_gap:${fixtureClosurePlanGapRemains}`,
        `fixture_closure_plan_live_authority:${fixtureClosurePlanLiveAuthority}`,
        `fixture_closure_plan_full_auto:${fixtureClosurePlanFullAuto}`,
      ],
      blockedReason: available
        ? stale
          ? 'Proof report artifact is stale; rerun proof checks before promotion review.'
          : 'Read-only artifact source; safety is normalized before cockpit display.'
        : 'No supervised proof report artifact found.',
      nextStep: available
        ? `Latest report status: ${str(proofSummary.latest_report_status, 'unknown')}; overlay: ${overlayStatus}; artifacts: ${artifactCount}; invalid: ${invalidCount}; ${freshnessText}; ${proofQualityText}; ${artifactReviewQueueText}; ${fixtureClosurePlanText}.`
        : 'Run no-WING supervised proof checks or point Product Layer at an existing report artifact.',
      tone: invalidCount > 0 ? 'red' : stale || !available ? 'yellow' : 'blue'
    }];
    rows.push({
      moduleId: 'read_surface_physical_readback_capture_plan',
      label: 'Readback capture plan',
      domain: 'read_surface_capture_plan',
      status: readSurfaceCapturePlanAvailable ? readSurfaceCapturePlanStatus || 'plan_available' : 'plan_missing',
      entrypoint: 'build_read_surface_physical_readback_capture_plan',
      module: readSurfaceCapturePlanActionId || 'no_next_capture_action',
      uiSurface: 'Read-surface physical readback capture plan',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${readSurfaceCapturePlanAvailable}`,
        `steps:${readSurfaceCapturePlanStepCount}`,
        `next_surface:${readSurfaceCapturePlanSurfaceId}`,
        `source_next_surface:${readSurfaceCapturePlanSourceSurfaceId}`,
        `next_action:${readSurfaceCapturePlanActionId}`,
        `source_next_action:${readSurfaceCapturePlanSourceActionId}`,
        `operator_required:${readSurfaceCapturePlanOperatorRequired}`,
        `execution_allowed_by_plan:${readSurfaceCapturePlanExecutionAllowed}`,
        `live_authority:${readSurfaceCapturePlanLiveAuthority}`,
        `target_consistency:${readSurfaceCapturePlanTargetConsistency}`,
        readSurfaceCapturePlanRequiredContracts.length ? `contracts:${readSurfaceCapturePlanRequiredContracts.join(',')}` : '',
        readSurfaceCapturePlanForbidden.length ? `forbidden:${readSurfaceCapturePlanForbidden.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: readSurfaceCapturePlanBlockedBy.length
        ? readSurfaceCapturePlanBlockedBy.join(', ')
        : 'Capture plan is complete or missing no current blockers.',
      nextStep: readSurfaceCapturePlanAvailable
        ? [
            readSurfaceCapturePlanSurfaceId ? `next_physical_readback:${readSurfaceCapturePlanSurfaceId}` : '',
            readSurfaceCapturePlanActionId ? `next_action:${readSurfaceCapturePlanActionId}` : '',
            `operator_required:${readSurfaceCapturePlanOperatorRequired}`,
            `execution_allowed_by_plan:${readSurfaceCapturePlanExecutionAllowed}`,
            `target_consistency:${readSurfaceCapturePlanTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with read-surface physical readback capture-plan validation.',
      tone: !readSurfaceCapturePlanAvailable || readSurfaceCapturePlanExecutionAllowed || readSurfaceCapturePlanLiveAuthority || !readSurfaceCapturePlanTargetConsistency
        ? 'yellow'
        : readSurfaceCapturePlanStepCount > 0
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'raw_mixer_state_physical_readback_contract',
      label: 'Raw mixer readback contract',
      domain: 'read_surface_contract',
      status: rawMixerContractAvailable ? rawMixerContractStatus || 'contract_available' : 'contract_missing',
      entrypoint: 'raw_mixer_state_physical_readback_contract',
      module: rawMixerContractActionId || 'no_contract_action',
      uiSurface: 'Read-surface proof contract',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${rawMixerContractAvailable}`,
        `schema:${rawMixerContractSchemaId}`,
        `surface:${rawMixerContractSurfaceId}`,
        `next_readback:${rawMixerContractNextSurfaceId}`,
        `next_action:${rawMixerContractNextActionId}`,
        `physical_gap_remains:${rawMixerContractGapRemains}`,
        `live_authority:${rawMixerContractLiveAuthority}`,
        `target_consistency:${rawMixerContractTargetConsistency}`,
      ],
      blockedReason: rawMixerContractAvailable
        ? rawMixerContractGapRemains
          ? 'Contract is present, but physical readback fixture evidence is still missing.'
          : 'Contract is present; physical gap state must still be reviewed before promotion.'
        : 'Raw mixer-state physical readback contract is missing from proof summary.',
      nextStep: rawMixerContractAvailable
        ? [
            rawMixerContractNextSurfaceId ? `next_readback:${rawMixerContractNextSurfaceId}` : '',
            rawMixerContractNextActionId ? `next_action:${rawMixerContractNextActionId}` : '',
            `physical_gap_remains:${rawMixerContractGapRemains}`,
            `live_authority:${rawMixerContractLiveAuthority}`,
            `target_consistency:${rawMixerContractTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with raw mixer readback contract schema validation.',
      tone: !rawMixerContractAvailable || rawMixerContractLiveAuthority || !rawMixerContractTargetConsistency
        ? 'yellow'
        : rawMixerContractGapRemains
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'raw_mixer_state_physical_readback_artifact_intake',
      label: 'Raw mixer readback artifact intake',
      domain: 'read_surface_artifact_intake',
      status: rawMixerIntakeAvailable ? rawMixerIntakeStatus || 'intake_available' : 'intake_missing',
      entrypoint: 'build_raw_mixer_state_physical_readback_artifact_intake',
      module: rawMixerIntakeActionId || 'no_intake_action',
      uiSurface: 'Read-surface proof artifact intake',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${rawMixerIntakeAvailable}`,
        `schema:${rawMixerIntakeSchemaId}`,
        `surface:${rawMixerIntakeSurfaceId}`,
        `action:${rawMixerIntakeActionId}`,
        `artifacts:${rawMixerIntakeArtifactCount}`,
        `valid:${rawMixerIntakeValidCount}`,
        `invalid:${rawMixerIntakeInvalidCount}`,
        `candidate:${rawMixerIntakeCandidate}`,
        `operator_review:${rawMixerIntakeOperatorReview}`,
        `physical_gap_remains:${rawMixerIntakeGapRemains}`,
        `fixture_closure_allowed:${rawMixerIntakeClosureAllowed}`,
        `live_authority:${rawMixerIntakeLiveAuthority}`,
        `target_consistency:${rawMixerIntakeTargetConsistency}`,
        rawMixerIntakeSources.length ? `sources:${rawMixerIntakeSources.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: rawMixerIntakeAvailable
        ? rawMixerIntakeBlockedBy.length
          ? rawMixerIntakeBlockedBy.join(', ')
          : 'Artifact intake is read-only; operator review still gates fixture closure.'
        : 'Raw mixer-state artifact intake is missing from proof summary.',
      nextStep: rawMixerIntakeAvailable
        ? [
            rawMixerIntakeSurfaceId ? `surface:${rawMixerIntakeSurfaceId}` : '',
            rawMixerIntakeActionId ? `action:${rawMixerIntakeActionId}` : '',
            `artifacts:${rawMixerIntakeArtifactCount}`,
            `valid:${rawMixerIntakeValidCount}`,
            `invalid:${rawMixerIntakeInvalidCount}`,
            `candidate:${rawMixerIntakeCandidate}`,
            `operator_review:${rawMixerIntakeOperatorReview}`,
            `fixture_closure_allowed:${rawMixerIntakeClosureAllowed}`,
            `live_authority:${rawMixerIntakeLiveAuthority}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with raw mixer artifact-intake validation.',
      tone: !rawMixerIntakeAvailable || rawMixerIntakeInvalidCount > 0 || rawMixerIntakeClosureAllowed || rawMixerIntakeLiveAuthority || !rawMixerIntakeTargetConsistency
        ? 'yellow'
        : rawMixerIntakeCandidate
          ? 'yellow'
          : 'yellow'
    });
    rows.push({
      moduleId: 'channel_basic_controls_physical_readback_contract',
      label: 'Channel basic controls readback contract',
      domain: 'read_surface_contract',
      status: channelBasicContractAvailable ? channelBasicContractStatus || 'contract_available' : 'contract_missing',
      entrypoint: 'channel_basic_controls_physical_readback_contract',
      module: channelBasicContractActionId || 'no_contract_action',
      uiSurface: 'Read-surface proof contract',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${channelBasicContractAvailable}`,
        `schema:${channelBasicContractSchemaId}`,
        `surface:${channelBasicContractSurfaceId}`,
        `physical_readback:${channelBasicContractPhysicalSurfaceId}`,
        `physical_action:${channelBasicContractPhysicalActionId}`,
        `physical_gap_remains:${channelBasicContractGapRemains}`,
        `live_authority:${channelBasicContractLiveAuthority}`,
        `target_consistency:${channelBasicContractTargetConsistency}`,
        `promotion_families:${channelBasicContractPromotionFamilies.join('|')}`,
        `controls:${channelBasicContractControls.join('|')}`,
      ],
      blockedReason: channelBasicContractAvailable
        ? channelBasicContractGapRemains
          ? 'Channel identity/basic-control contract is present, but physical readback fixture evidence is still missing.'
          : 'Channel identity/basic-control contract is present; physical gap state must still be reviewed before promotion.'
        : 'Channel identity/basic-control physical readback contract is missing from proof summary.',
      nextStep: channelBasicContractAvailable
        ? [
            channelBasicContractPhysicalSurfaceId ? `physical_readback:${channelBasicContractPhysicalSurfaceId}` : '',
            channelBasicContractPhysicalActionId ? `physical_action:${channelBasicContractPhysicalActionId}` : '',
            channelBasicContractPromotionFamilies.length ? `families:${channelBasicContractPromotionFamilies.join(',')}` : '',
            channelBasicContractControls.length ? `controls:${channelBasicContractControls.join(',')}` : '',
            `physical_gap_remains:${channelBasicContractGapRemains}`,
            `live_authority:${channelBasicContractLiveAuthority}`,
            `target_consistency:${channelBasicContractTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with channel basic-controls readback contract schema validation.',
      tone: !channelBasicContractAvailable || channelBasicContractLiveAuthority || !channelBasicContractTargetConsistency
        ? 'yellow'
        : channelBasicContractGapRemains
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'channel_basic_controls_physical_readback_artifact_intake',
      label: 'Channel basic controls artifact intake',
      domain: 'read_surface_artifact_intake',
      status: channelBasicIntakeAvailable ? channelBasicIntakeStatus || 'intake_available' : 'intake_missing',
      entrypoint: 'build_channel_basic_controls_physical_readback_artifact_intake',
      module: channelBasicIntakeActionId || 'no_intake_action',
      uiSurface: 'Read-surface proof artifact intake',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${channelBasicIntakeAvailable}`,
        `schema:${channelBasicIntakeSchemaId}`,
        `surface:${channelBasicIntakeSurfaceId}`,
        `action:${channelBasicIntakeActionId}`,
        `artifacts:${channelBasicIntakeArtifactCount}`,
        `valid:${channelBasicIntakeValidCount}`,
        `invalid:${channelBasicIntakeInvalidCount}`,
        `valid_channels:${channelBasicIntakeValidChannelCount}`,
        `candidate:${channelBasicIntakeCandidate}`,
        `operator_review:${channelBasicIntakeOperatorReview}`,
        `physical_gap_remains:${channelBasicIntakeGapRemains}`,
        `fixture_closure_allowed:${channelBasicIntakeClosureAllowed}`,
        `live_authority:${channelBasicIntakeLiveAuthority}`,
        `target_consistency:${channelBasicIntakeTargetConsistency}`,
        channelBasicIntakePromotionFamilies.length ? `promotion_families:${channelBasicIntakePromotionFamilies.join('|')}` : '',
        channelBasicIntakeSources.length ? `sources:${channelBasicIntakeSources.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: channelBasicIntakeAvailable
        ? channelBasicIntakeBlockedBy.length
          ? channelBasicIntakeBlockedBy.join(', ')
          : 'Channel basic-controls artifact intake is read-only; operator review still gates fixture closure.'
        : 'Channel basic-controls artifact intake is missing from proof summary.',
      nextStep: channelBasicIntakeAvailable
        ? [
            channelBasicIntakeSurfaceId ? `surface:${channelBasicIntakeSurfaceId}` : '',
            channelBasicIntakeActionId ? `action:${channelBasicIntakeActionId}` : '',
            `artifacts:${channelBasicIntakeArtifactCount}`,
            `valid:${channelBasicIntakeValidCount}`,
            `invalid:${channelBasicIntakeInvalidCount}`,
            `valid_channels:${channelBasicIntakeValidChannelCount}`,
            `candidate:${channelBasicIntakeCandidate}`,
            `operator_review:${channelBasicIntakeOperatorReview}`,
            `fixture_closure_allowed:${channelBasicIntakeClosureAllowed}`,
            `live_authority:${channelBasicIntakeLiveAuthority}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with channel basic-controls artifact-intake validation.',
      tone: !channelBasicIntakeAvailable || channelBasicIntakeInvalidCount > 0 || channelBasicIntakeClosureAllowed || channelBasicIntakeLiveAuthority || !channelBasicIntakeTargetConsistency
        ? 'yellow'
        : channelBasicIntakeCandidate
          ? 'yellow'
          : 'yellow'
    });
    rows.push({
      moduleId: 'channel_processing_physical_readback_contract',
      label: 'Channel processing readback contract',
      domain: 'read_surface_contract',
      status: channelProcessingContractAvailable ? channelProcessingContractStatus || 'contract_available' : 'contract_missing',
      entrypoint: 'channel_processing_physical_readback_contract',
      module: channelProcessingContractActionId || 'no_contract_action',
      uiSurface: 'Read-surface proof contract',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${channelProcessingContractAvailable}`,
        `schema:${channelProcessingContractSchemaId}`,
        `surface:${channelProcessingContractSurfaceId}`,
        `physical_readback:${channelProcessingContractPhysicalSurfaceId}`,
        `physical_action:${channelProcessingContractPhysicalActionId}`,
        `physical_gap_remains:${channelProcessingContractGapRemains}`,
        `live_authority:${channelProcessingContractLiveAuthority}`,
        `target_consistency:${channelProcessingContractTargetConsistency}`,
        `promotion_families:${channelProcessingContractPromotionFamilies.join('|')}`,
        `eq_band_fields:${channelProcessingContractEqBands.join('|')}`,
        `compressor_fields:${channelProcessingContractCompressor.join('|')}`,
        `gate_fields:${channelProcessingContractGate.join('|')}`,
      ],
      blockedReason: channelProcessingContractAvailable
        ? channelProcessingContractGapRemains
          ? 'Channel EQ/dynamics/gate contract is present, but physical readback fixture evidence is still missing.'
          : 'Channel EQ/dynamics/gate contract is present; physical gap state must still be reviewed before promotion.'
        : 'Channel EQ/dynamics/gate physical readback contract is missing from proof summary.',
      nextStep: channelProcessingContractAvailable
        ? [
            channelProcessingContractPhysicalSurfaceId ? `physical_readback:${channelProcessingContractPhysicalSurfaceId}` : '',
            channelProcessingContractPhysicalActionId ? `physical_action:${channelProcessingContractPhysicalActionId}` : '',
            channelProcessingContractPromotionFamilies.length ? `families:${channelProcessingContractPromotionFamilies.join(',')}` : '',
            channelProcessingContractEqBands.length ? `eq_bands:${channelProcessingContractEqBands.join(',')}` : '',
            channelProcessingContractCompressor.length ? `compressor:${channelProcessingContractCompressor.join(',')}` : '',
            channelProcessingContractGate.length ? `gate:${channelProcessingContractGate.join(',')}` : '',
            `physical_gap_remains:${channelProcessingContractGapRemains}`,
            `live_authority:${channelProcessingContractLiveAuthority}`,
            `target_consistency:${channelProcessingContractTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with channel processing readback contract schema validation.',
      tone: !channelProcessingContractAvailable || channelProcessingContractLiveAuthority || !channelProcessingContractTargetConsistency
        ? 'yellow'
        : channelProcessingContractGapRemains
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'channel_processing_physical_readback_artifact_intake',
      label: 'Channel processing readback artifact intake',
      domain: 'read_surface_artifact_intake',
      status: channelProcessingIntakeAvailable ? channelProcessingIntakeStatus || 'intake_available' : 'intake_missing',
      entrypoint: 'build_channel_processing_physical_readback_artifact_intake',
      module: channelProcessingIntakeActionId || 'no_intake_action',
      uiSurface: 'Read-surface proof artifact intake',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${channelProcessingIntakeAvailable}`,
        `schema:${channelProcessingIntakeSchemaId}`,
        `surface:${channelProcessingIntakeSurfaceId}`,
        `action:${channelProcessingIntakeActionId}`,
        `artifacts:${channelProcessingIntakeArtifactCount}`,
        `valid:${channelProcessingIntakeValidCount}`,
        `invalid:${channelProcessingIntakeInvalidCount}`,
        `valid_channels:${channelProcessingIntakeValidChannelCount}`,
        `candidate:${channelProcessingIntakeCandidate}`,
        `operator_review:${channelProcessingIntakeOperatorReview}`,
        `physical_gap_remains:${channelProcessingIntakeGapRemains}`,
        `fixture_closure_allowed:${channelProcessingIntakeClosureAllowed}`,
        `live_authority:${channelProcessingIntakeLiveAuthority}`,
        `target_consistency:${channelProcessingIntakeTargetConsistency}`,
        channelProcessingIntakePromotionFamilies.length ? `promotion_families:${channelProcessingIntakePromotionFamilies.join('|')}` : '',
        channelProcessingIntakeSources.length ? `sources:${channelProcessingIntakeSources.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: channelProcessingIntakeAvailable
        ? channelProcessingIntakeBlockedBy.length
          ? channelProcessingIntakeBlockedBy.join(', ')
          : 'Channel processing artifact intake is read-only; operator review still gates fixture closure.'
        : 'Channel processing artifact intake is missing from proof summary.',
      nextStep: channelProcessingIntakeAvailable
        ? [
            channelProcessingIntakeSurfaceId ? `surface:${channelProcessingIntakeSurfaceId}` : '',
            channelProcessingIntakeActionId ? `action:${channelProcessingIntakeActionId}` : '',
            `artifacts:${channelProcessingIntakeArtifactCount}`,
            `valid:${channelProcessingIntakeValidCount}`,
            `invalid:${channelProcessingIntakeInvalidCount}`,
            `valid_channels:${channelProcessingIntakeValidChannelCount}`,
            `candidate:${channelProcessingIntakeCandidate}`,
            `operator_review:${channelProcessingIntakeOperatorReview}`,
            `fixture_closure_allowed:${channelProcessingIntakeClosureAllowed}`,
            `live_authority:${channelProcessingIntakeLiveAuthority}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with channel processing artifact-intake validation.',
      tone: !channelProcessingIntakeAvailable || channelProcessingIntakeInvalidCount > 0 || channelProcessingIntakeClosureAllowed || channelProcessingIntakeLiveAuthority || !channelProcessingIntakeTargetConsistency
        ? 'yellow'
        : channelProcessingIntakeCandidate
          ? 'yellow'
          : 'yellow'
    });
    rows.push({
      moduleId: 'bus_group_main_dca_physical_readback_contract',
      label: 'Bus/main/DCA readback contract',
      domain: 'read_surface_contract',
      status: busMainDcaContractAvailable ? busMainDcaContractStatus || 'contract_available' : 'contract_missing',
      entrypoint: 'bus_group_main_dca_physical_readback_contract',
      module: busMainDcaContractActionId || 'no_contract_action',
      uiSurface: 'Read-surface proof contract',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${busMainDcaContractAvailable}`,
        `schema:${busMainDcaContractSchemaId}`,
        `surface:${busMainDcaContractSurfaceId}`,
        `physical_readback:${busMainDcaContractPhysicalSurfaceId}`,
        `physical_action:${busMainDcaContractPhysicalActionId}`,
        `physical_gap_remains:${busMainDcaContractGapRemains}`,
        `live_authority:${busMainDcaContractLiveAuthority}`,
        `target_consistency:${busMainDcaContractTargetConsistency}`,
        `promotion_families:${busMainDcaContractPromotionFamilies.join('|')}`,
        `bus_controls:${busMainDcaContractBusControls.join('|')}`,
        `main_controls:${busMainDcaContractMainControls.join('|')}`,
        `dca_controls:${busMainDcaContractDcaControls.join('|')}`,
        `bus_assignments:${busMainDcaContractBusAssignments.join('|')}`,
        `dca_assignments:${busMainDcaContractDcaAssignments.join('|')}`,
        `metering:${busMainDcaContractMetering.join('|')}`,
      ],
      blockedReason: busMainDcaContractAvailable
        ? busMainDcaContractGapRemains
          ? 'Bus/group/main/DCA contract is present, but physical readback fixture evidence is still missing.'
          : 'Bus/group/main/DCA contract is present; physical gap state must still be reviewed before promotion.'
        : 'Bus/group/main/DCA physical readback contract is missing from proof summary.',
      nextStep: busMainDcaContractAvailable
        ? [
            busMainDcaContractPhysicalSurfaceId ? `physical_readback:${busMainDcaContractPhysicalSurfaceId}` : '',
            busMainDcaContractPhysicalActionId ? `physical_action:${busMainDcaContractPhysicalActionId}` : '',
            busMainDcaContractPromotionFamilies.length ? `families:${busMainDcaContractPromotionFamilies.join(',')}` : '',
            busMainDcaContractBusControls.length ? `bus_controls:${busMainDcaContractBusControls.join(',')}` : '',
            busMainDcaContractMainControls.length ? `main_controls:${busMainDcaContractMainControls.join(',')}` : '',
            busMainDcaContractDcaControls.length ? `dca_controls:${busMainDcaContractDcaControls.join(',')}` : '',
            busMainDcaContractMetering.length ? `metering:${busMainDcaContractMetering.join(',')}` : '',
            `physical_gap_remains:${busMainDcaContractGapRemains}`,
            `live_authority:${busMainDcaContractLiveAuthority}`,
            `target_consistency:${busMainDcaContractTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with bus/group/main/DCA readback contract schema validation.',
      tone: !busMainDcaContractAvailable || busMainDcaContractLiveAuthority || !busMainDcaContractTargetConsistency
        ? 'yellow'
        : busMainDcaContractGapRemains
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'bus_group_main_dca_physical_readback_artifact_intake',
      label: 'Bus/main/DCA readback artifact intake',
      domain: 'read_surface_artifact_intake',
      status: busMainDcaIntakeAvailable ? busMainDcaIntakeStatus || 'intake_available' : 'intake_missing',
      entrypoint: 'build_bus_group_main_dca_physical_readback_artifact_intake',
      module: busMainDcaIntakeActionId || 'no_intake_action',
      uiSurface: 'Read-surface physical artifact intake',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${busMainDcaIntakeAvailable}`,
        `schema:${busMainDcaIntakeSchemaId}`,
        `surface:${busMainDcaIntakeSurfaceId}`,
        `action:${busMainDcaIntakeActionId}`,
        `artifacts:${busMainDcaIntakeArtifactCount}`,
        `valid:${busMainDcaIntakeValidCount}`,
        `invalid:${busMainDcaIntakeInvalidCount}`,
        `valid_buses:${busMainDcaIntakeValidBusCount}`,
        `valid_mains:${busMainDcaIntakeValidMainCount}`,
        `valid_dcas:${busMainDcaIntakeValidDcaCount}`,
        `candidate:${busMainDcaIntakeCandidate}`,
        `operator_review:${busMainDcaIntakeOperatorReview}`,
        `physical_gap_remains:${busMainDcaIntakeGapRemains}`,
        `fixture_closure_allowed:${busMainDcaIntakeClosureAllowed}`,
        `live_authority:${busMainDcaIntakeLiveAuthority}`,
        `target_consistency:${busMainDcaIntakeTargetConsistency}`,
        busMainDcaIntakePromotionFamilies.length ? `promotion_families:${busMainDcaIntakePromotionFamilies.join('|')}` : '',
        busMainDcaIntakeSources.length ? `sources:${busMainDcaIntakeSources.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: busMainDcaIntakeAvailable
        ? busMainDcaIntakeBlockedBy.length
          ? busMainDcaIntakeBlockedBy.join(', ')
          : 'Bus/group/main/DCA artifact intake is present; manual review still gates fixture closure.'
        : 'Bus/group/main/DCA physical artifact intake is missing from proof summary.',
      nextStep: busMainDcaIntakeAvailable
        ? [
            busMainDcaIntakeSurfaceId ? `surface:${busMainDcaIntakeSurfaceId}` : '',
            busMainDcaIntakeActionId ? `action:${busMainDcaIntakeActionId}` : '',
            `artifacts:${busMainDcaIntakeArtifactCount}`,
            `valid:${busMainDcaIntakeValidCount}`,
            `invalid:${busMainDcaIntakeInvalidCount}`,
            `valid_buses:${busMainDcaIntakeValidBusCount}`,
            `valid_mains:${busMainDcaIntakeValidMainCount}`,
            `valid_dcas:${busMainDcaIntakeValidDcaCount}`,
            `candidate:${busMainDcaIntakeCandidate}`,
            `operator_review:${busMainDcaIntakeOperatorReview}`,
            `fixture_closure_allowed:${busMainDcaIntakeClosureAllowed}`,
            `live_authority:${busMainDcaIntakeLiveAuthority}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with bus/group/main/DCA artifact-intake validation.',
      tone: !busMainDcaIntakeAvailable || busMainDcaIntakeInvalidCount > 0 || busMainDcaIntakeClosureAllowed || busMainDcaIntakeLiveAuthority || !busMainDcaIntakeTargetConsistency
        ? 'yellow'
        : busMainDcaIntakeCandidate
          ? 'yellow'
          : 'yellow'
    });
    rows.push({
      moduleId: 'matrix_state_physical_readback_contract',
      label: 'Matrix state readback contract',
      domain: 'read_surface_contract',
      status: matrixStateContractAvailable ? matrixStateContractStatus || 'contract_available' : 'contract_missing',
      entrypoint: 'matrix_state_physical_readback_contract',
      module: matrixStateContractActionId || 'no_contract_action',
      uiSurface: 'Read-surface proof contract',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${matrixStateContractAvailable}`,
        `schema:${matrixStateContractSchemaId}`,
        `surface:${matrixStateContractSurfaceId}`,
        `physical_readback:${matrixStateContractPhysicalSurfaceId}`,
        `physical_action:${matrixStateContractPhysicalActionId}`,
        `physical_gap_remains:${matrixStateContractGapRemains}`,
        `live_authority:${matrixStateContractLiveAuthority}`,
        `target_consistency:${matrixStateContractTargetConsistency}`,
        `promotion_families:${matrixStateContractPromotionFamilies.join('|')}`,
        `controls:${matrixStateContractControls.join('|')}`,
        `eq_band_fields:${matrixStateContractEqBands.join('|')}`,
        `dynamics:${matrixStateContractDynamics.join('|')}`,
        `inserts:${matrixStateContractInserts.join('|')}`,
        `metering:${matrixStateContractMetering.join('|')}`,
      ],
      blockedReason: matrixStateContractAvailable
        ? matrixStateContractGapRemains
          ? 'Matrix-state contract is present, but physical readback fixture evidence is still missing.'
          : 'Matrix-state contract is present; physical gap state must still be reviewed before promotion.'
        : 'Matrix-state physical readback contract is missing from proof summary.',
      nextStep: matrixStateContractAvailable
        ? [
            matrixStateContractPhysicalSurfaceId ? `physical_readback:${matrixStateContractPhysicalSurfaceId}` : '',
            matrixStateContractPhysicalActionId ? `physical_action:${matrixStateContractPhysicalActionId}` : '',
            matrixStateContractPromotionFamilies.length ? `families:${matrixStateContractPromotionFamilies.join(',')}` : '',
            matrixStateContractControls.length ? `controls:${matrixStateContractControls.join(',')}` : '',
            matrixStateContractEqBands.length ? `eq_bands:${matrixStateContractEqBands.join(',')}` : '',
            matrixStateContractDynamics.length ? `dynamics:${matrixStateContractDynamics.join(',')}` : '',
            matrixStateContractInserts.length ? `inserts:${matrixStateContractInserts.join(',')}` : '',
            matrixStateContractMetering.length ? `metering:${matrixStateContractMetering.join(',')}` : '',
            `physical_gap_remains:${matrixStateContractGapRemains}`,
            `live_authority:${matrixStateContractLiveAuthority}`,
            `target_consistency:${matrixStateContractTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with matrix-state readback contract schema validation.',
      tone: !matrixStateContractAvailable || matrixStateContractLiveAuthority || !matrixStateContractTargetConsistency
        ? 'yellow'
        : matrixStateContractGapRemains
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'matrix_state_physical_readback_artifact_intake',
      label: 'Matrix state readback artifact intake',
      domain: 'read_surface_artifact_intake',
      status: matrixStateIntakeAvailable ? matrixStateIntakeStatus || 'intake_available' : 'intake_missing',
      entrypoint: 'build_matrix_state_physical_readback_artifact_intake',
      module: matrixStateIntakeActionId || 'no_intake_action',
      uiSurface: 'Read-surface physical artifact intake',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${matrixStateIntakeAvailable}`,
        `schema:${matrixStateIntakeSchemaId}`,
        `surface:${matrixStateIntakeSurfaceId}`,
        `action:${matrixStateIntakeActionId}`,
        `artifacts:${matrixStateIntakeArtifactCount}`,
        `valid:${matrixStateIntakeValidCount}`,
        `invalid:${matrixStateIntakeInvalidCount}`,
        `valid_matrices:${matrixStateIntakeValidMatrixCount}`,
        `candidate:${matrixStateIntakeCandidate}`,
        `operator_review:${matrixStateIntakeOperatorReview}`,
        `physical_gap_remains:${matrixStateIntakeGapRemains}`,
        `fixture_closure_allowed:${matrixStateIntakeClosureAllowed}`,
        `live_authority:${matrixStateIntakeLiveAuthority}`,
        `target_consistency:${matrixStateIntakeTargetConsistency}`,
        matrixStateIntakePromotionFamilies.length ? `promotion_families:${matrixStateIntakePromotionFamilies.join('|')}` : '',
        matrixStateIntakeSources.length ? `sources:${matrixStateIntakeSources.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: matrixStateIntakeAvailable
        ? matrixStateIntakeBlockedBy.length
          ? matrixStateIntakeBlockedBy.join(', ')
          : 'Matrix-state artifact intake is present; manual review still gates fixture closure.'
        : 'Matrix-state physical artifact intake is missing from proof summary.',
      nextStep: matrixStateIntakeAvailable
        ? [
            matrixStateIntakeSurfaceId ? `surface:${matrixStateIntakeSurfaceId}` : '',
            matrixStateIntakeActionId ? `action:${matrixStateIntakeActionId}` : '',
            `artifacts:${matrixStateIntakeArtifactCount}`,
            `valid:${matrixStateIntakeValidCount}`,
            `invalid:${matrixStateIntakeInvalidCount}`,
            `valid_matrices:${matrixStateIntakeValidMatrixCount}`,
            `candidate:${matrixStateIntakeCandidate}`,
            `operator_review:${matrixStateIntakeOperatorReview}`,
            `fixture_closure_allowed:${matrixStateIntakeClosureAllowed}`,
            `live_authority:${matrixStateIntakeLiveAuthority}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with matrix-state artifact-intake validation.',
      tone: !matrixStateIntakeAvailable || matrixStateIntakeInvalidCount > 0 || matrixStateIntakeClosureAllowed || matrixStateIntakeLiveAuthority || !matrixStateIntakeTargetConsistency
        ? 'yellow'
        : matrixStateIntakeCandidate
          ? 'yellow'
          : 'yellow'
    });
    rows.push({
      moduleId: 'fx_slots_inserts_physical_readback_contract',
      label: 'FX slots/inserts readback contract',
      domain: 'read_surface_contract',
      status: fxSlotsInsertsContractAvailable ? fxSlotsInsertsContractStatus || 'contract_available' : 'contract_missing',
      entrypoint: 'fx_slots_inserts_physical_readback_contract',
      module: fxSlotsInsertsContractActionId || 'no_contract_action',
      uiSurface: 'Read-surface proof contract',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${fxSlotsInsertsContractAvailable}`,
        `schema:${fxSlotsInsertsContractSchemaId}`,
        `surface:${fxSlotsInsertsContractSurfaceId}`,
        `physical_readback:${fxSlotsInsertsContractPhysicalSurfaceId}`,
        `physical_action:${fxSlotsInsertsContractPhysicalActionId}`,
        `physical_gap_remains:${fxSlotsInsertsContractGapRemains}`,
        `live_authority:${fxSlotsInsertsContractLiveAuthority}`,
        `target_consistency:${fxSlotsInsertsContractTargetConsistency}`,
        `promotion_families:${fxSlotsInsertsContractPromotionFamilies.join('|')}`,
        `slot_fields:${fxSlotsInsertsContractSlotFields.join('|')}`,
        `parameter_fields:${fxSlotsInsertsContractParameterFields.join('|')}`,
        `parameter_schema:${fxSlotsInsertsContractParameterSchemaFields.join('|')}`,
        `insert_fields:${fxSlotsInsertsContractInsertFields.join('|')}`,
        `insert_coverage:${fxSlotsInsertsContractInsertCoverage.join('|')}`,
        `false_safety:${fxSlotsInsertsContractFalseSafetyFlags.join('|')}`,
      ],
      blockedReason: fxSlotsInsertsContractAvailable
        ? fxSlotsInsertsContractGapRemains
          ? 'FX slots/inserts contract is present, but physical readback fixture evidence is still missing.'
          : 'FX slots/inserts contract is present; physical gap state must still be reviewed before promotion.'
        : 'FX slots/inserts physical readback contract is missing from proof summary.',
      nextStep: fxSlotsInsertsContractAvailable
        ? [
            fxSlotsInsertsContractPhysicalSurfaceId ? `physical_readback:${fxSlotsInsertsContractPhysicalSurfaceId}` : '',
            fxSlotsInsertsContractPhysicalActionId ? `physical_action:${fxSlotsInsertsContractPhysicalActionId}` : '',
            fxSlotsInsertsContractPromotionFamilies.length ? `families:${fxSlotsInsertsContractPromotionFamilies.join(',')}` : '',
            fxSlotsInsertsContractSlotFields.length ? `slot_fields:${fxSlotsInsertsContractSlotFields.join(',')}` : '',
            fxSlotsInsertsContractParameterFields.length ? `parameters:${fxSlotsInsertsContractParameterFields.join(',')}` : '',
            fxSlotsInsertsContractInsertFields.length ? `inserts:${fxSlotsInsertsContractInsertFields.join(',')}` : '',
            fxSlotsInsertsContractFalseSafetyFlags.length ? `false_safety:${fxSlotsInsertsContractFalseSafetyFlags.join(',')}` : '',
            `physical_gap_remains:${fxSlotsInsertsContractGapRemains}`,
            `live_authority:${fxSlotsInsertsContractLiveAuthority}`,
            `target_consistency:${fxSlotsInsertsContractTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with FX slots/inserts readback contract schema validation.',
      tone: !fxSlotsInsertsContractAvailable || fxSlotsInsertsContractLiveAuthority || !fxSlotsInsertsContractTargetConsistency
        ? 'yellow'
        : fxSlotsInsertsContractGapRemains
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'fx_slots_inserts_physical_readback_artifact_intake',
      label: 'FX slots/inserts readback artifact intake',
      domain: 'read_surface_artifact_intake',
      status: fxSlotsInsertsIntakeAvailable ? fxSlotsInsertsIntakeStatus || 'intake_available' : 'intake_missing',
      entrypoint: 'build_fx_slots_inserts_physical_readback_artifact_intake',
      module: fxSlotsInsertsIntakeActionId || 'no_intake_action',
      uiSurface: 'Read-surface physical artifact intake',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${fxSlotsInsertsIntakeAvailable}`,
        `schema:${fxSlotsInsertsIntakeSchemaId}`,
        `surface:${fxSlotsInsertsIntakeSurfaceId}`,
        `action:${fxSlotsInsertsIntakeActionId}`,
        `artifacts:${fxSlotsInsertsIntakeArtifactCount}`,
        `valid:${fxSlotsInsertsIntakeValidCount}`,
        `invalid:${fxSlotsInsertsIntakeInvalidCount}`,
        `valid_slots:${fxSlotsInsertsIntakeValidFxSlotCount}`,
        `valid_inserts:${fxSlotsInsertsIntakeValidInsertCount}`,
        `candidate:${fxSlotsInsertsIntakeCandidate}`,
        `operator_review:${fxSlotsInsertsIntakeOperatorReview}`,
        `physical_gap_remains:${fxSlotsInsertsIntakeGapRemains}`,
        `fixture_closure_allowed:${fxSlotsInsertsIntakeClosureAllowed}`,
        `live_authority:${fxSlotsInsertsIntakeLiveAuthority}`,
        `target_consistency:${fxSlotsInsertsIntakeTargetConsistency}`,
        fxSlotsInsertsIntakePromotionFamilies.length ? `promotion_families:${fxSlotsInsertsIntakePromotionFamilies.join('|')}` : '',
        fxSlotsInsertsIntakeSources.length ? `sources:${fxSlotsInsertsIntakeSources.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: fxSlotsInsertsIntakeAvailable
        ? fxSlotsInsertsIntakeBlockedBy.length
          ? fxSlotsInsertsIntakeBlockedBy.join(', ')
          : 'FX slots/inserts artifact intake is present; manual review still gates fixture closure.'
        : 'FX slots/inserts physical artifact intake is missing from proof summary.',
      nextStep: fxSlotsInsertsIntakeAvailable
        ? [
            fxSlotsInsertsIntakeSurfaceId ? `surface:${fxSlotsInsertsIntakeSurfaceId}` : '',
            fxSlotsInsertsIntakeActionId ? `action:${fxSlotsInsertsIntakeActionId}` : '',
            `artifacts:${fxSlotsInsertsIntakeArtifactCount}`,
            `valid:${fxSlotsInsertsIntakeValidCount}`,
            `invalid:${fxSlotsInsertsIntakeInvalidCount}`,
            `valid_slots:${fxSlotsInsertsIntakeValidFxSlotCount}`,
            `valid_inserts:${fxSlotsInsertsIntakeValidInsertCount}`,
            `candidate:${fxSlotsInsertsIntakeCandidate}`,
            `operator_review:${fxSlotsInsertsIntakeOperatorReview}`,
            `fixture_closure_allowed:${fxSlotsInsertsIntakeClosureAllowed}`,
            `live_authority:${fxSlotsInsertsIntakeLiveAuthority}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with FX slots/inserts artifact-intake validation.',
      tone: !fxSlotsInsertsIntakeAvailable || fxSlotsInsertsIntakeInvalidCount > 0 || fxSlotsInsertsIntakeClosureAllowed || fxSlotsInsertsIntakeLiveAuthority || !fxSlotsInsertsIntakeTargetConsistency
        ? 'yellow'
        : fxSlotsInsertsIntakeCandidate
          ? 'yellow'
          : 'yellow'
    });
    rows.push({
      moduleId: 'routing_sends_physical_readback_contract',
      label: 'Routing/sends readback contract',
      domain: 'read_surface_contract',
      status: routingSendsContractAvailable ? routingSendsContractStatus || 'contract_available' : 'contract_missing',
      entrypoint: 'routing_sends_physical_readback_contract',
      module: routingSendsContractActionId || 'no_contract_action',
      uiSurface: 'Read-surface proof contract',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${routingSendsContractAvailable}`,
        `schema:${routingSendsContractSchemaId}`,
        `surface:${routingSendsContractSurfaceId}`,
        `physical_readback:${routingSendsContractPhysicalSurfaceId}`,
        `physical_action:${routingSendsContractPhysicalActionId}`,
        `physical_gap_remains:${routingSendsContractGapRemains}`,
        `live_authority:${routingSendsContractLiveAuthority}`,
        `target_consistency:${routingSendsContractTargetConsistency}`,
        `promotion_families:${routingSendsContractPromotionFamilies.join('|')}`,
        `static_routes:${routingSendsContractStaticFields.join('|')}`,
        `bus_sends:${routingSendsContractBusSendFields.join('|')}`,
        `main_sends:${routingSendsContractMainSendFields.join('|')}`,
        `matrix_sends:${routingSendsContractMatrixSendFields.join('|')}`,
        `output_routes:${routingSendsContractOutputRouteFields.join('|')}`,
        `send_coverage:${routingSendsContractSendCoverage.join('|')}`,
        `output_coverage:${routingSendsContractOutputCoverage.join('|')}`,
        `false_safety:${routingSendsContractFalseSafetyFlags.join('|')}`,
      ],
      blockedReason: routingSendsContractAvailable
        ? routingSendsContractGapRemains
          ? 'Routing/sends contract is present, but physical readback fixture evidence is still missing.'
          : 'Routing/sends contract is present; physical gap state must still be reviewed before promotion.'
        : 'Routing/sends physical readback contract is missing from proof summary.',
      nextStep: routingSendsContractAvailable
        ? [
            routingSendsContractPhysicalSurfaceId ? `physical_readback:${routingSendsContractPhysicalSurfaceId}` : '',
            routingSendsContractPhysicalActionId ? `physical_action:${routingSendsContractPhysicalActionId}` : '',
            routingSendsContractPromotionFamilies.length ? `families:${routingSendsContractPromotionFamilies.join(',')}` : '',
            routingSendsContractStaticFields.length ? `static_routes:${routingSendsContractStaticFields.join(',')}` : '',
            routingSendsContractBusSendFields.length ? `bus_sends:${routingSendsContractBusSendFields.join(',')}` : '',
            routingSendsContractMainSendFields.length ? `main_sends:${routingSendsContractMainSendFields.join(',')}` : '',
            routingSendsContractMatrixSendFields.length ? `matrix_sends:${routingSendsContractMatrixSendFields.join(',')}` : '',
            routingSendsContractOutputRouteFields.length ? `output_routes:${routingSendsContractOutputRouteFields.join(',')}` : '',
            routingSendsContractFalseSafetyFlags.length ? `false_safety:${routingSendsContractFalseSafetyFlags.join(',')}` : '',
            `physical_gap_remains:${routingSendsContractGapRemains}`,
            `live_authority:${routingSendsContractLiveAuthority}`,
            `target_consistency:${routingSendsContractTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with routing/sends readback contract schema validation.',
      tone: !routingSendsContractAvailable || routingSendsContractLiveAuthority || !routingSendsContractTargetConsistency
        ? 'yellow'
        : routingSendsContractGapRemains
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'routing_sends_physical_readback_artifact_intake',
      label: 'Routing/sends readback artifact intake',
      domain: 'read_surface_artifact_intake',
      status: routingSendsIntakeAvailable ? routingSendsIntakeStatus || 'intake_available' : 'intake_missing',
      entrypoint: 'build_routing_sends_physical_readback_artifact_intake',
      module: routingSendsIntakeActionId || 'no_intake_action',
      uiSurface: 'Read-surface physical artifact intake',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${routingSendsIntakeAvailable}`,
        `schema:${routingSendsIntakeSchemaId}`,
        `surface:${routingSendsIntakeSurfaceId}`,
        `action:${routingSendsIntakeActionId}`,
        `artifacts:${routingSendsIntakeArtifactCount}`,
        `valid:${routingSendsIntakeValidCount}`,
        `invalid:${routingSendsIntakeInvalidCount}`,
        `valid_static_routes:${routingSendsIntakeValidStaticRouteCount}`,
        `valid_bus_sends:${routingSendsIntakeValidBusSendCount}`,
        `valid_main_sends:${routingSendsIntakeValidMainSendCount}`,
        `valid_matrix_sends:${routingSendsIntakeValidMatrixSendCount}`,
        `valid_output_routes:${routingSendsIntakeValidOutputRouteCount}`,
        `candidate:${routingSendsIntakeCandidate}`,
        `operator_review:${routingSendsIntakeOperatorReview}`,
        `physical_gap_remains:${routingSendsIntakeGapRemains}`,
        `fixture_closure_allowed:${routingSendsIntakeClosureAllowed}`,
        `live_authority:${routingSendsIntakeLiveAuthority}`,
        `target_consistency:${routingSendsIntakeTargetConsistency}`,
        routingSendsIntakePromotionFamilies.length ? `promotion_families:${routingSendsIntakePromotionFamilies.join('|')}` : '',
        routingSendsIntakeSources.length ? `sources:${routingSendsIntakeSources.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: routingSendsIntakeAvailable
        ? routingSendsIntakeBlockedBy.length
          ? routingSendsIntakeBlockedBy.join(', ')
          : 'Routing/sends artifact intake is present; manual review still gates fixture closure.'
        : 'Routing/sends physical artifact intake is missing from proof summary.',
      nextStep: routingSendsIntakeAvailable
        ? [
            routingSendsIntakeSurfaceId ? `surface:${routingSendsIntakeSurfaceId}` : '',
            routingSendsIntakeActionId ? `action:${routingSendsIntakeActionId}` : '',
            `artifacts:${routingSendsIntakeArtifactCount}`,
            `valid:${routingSendsIntakeValidCount}`,
            `invalid:${routingSendsIntakeInvalidCount}`,
            `valid_static_routes:${routingSendsIntakeValidStaticRouteCount}`,
            `valid_bus_sends:${routingSendsIntakeValidBusSendCount}`,
            `valid_main_sends:${routingSendsIntakeValidMainSendCount}`,
            `valid_matrix_sends:${routingSendsIntakeValidMatrixSendCount}`,
            `valid_output_routes:${routingSendsIntakeValidOutputRouteCount}`,
            `candidate:${routingSendsIntakeCandidate}`,
            `operator_review:${routingSendsIntakeOperatorReview}`,
            `fixture_closure_allowed:${routingSendsIntakeClosureAllowed}`,
            `live_authority:${routingSendsIntakeLiveAuthority}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with routing/sends artifact-intake validation.',
      tone: !routingSendsIntakeAvailable || routingSendsIntakeInvalidCount > 0 || routingSendsIntakeClosureAllowed || routingSendsIntakeLiveAuthority || !routingSendsIntakeTargetConsistency
        ? 'yellow'
        : routingSendsIntakeCandidate
          ? 'yellow'
          : 'yellow'
    });
    rows.push({
      moduleId: 'snapshots_scenes_physical_readback_contract',
      label: 'Snapshots/scenes readback contract',
      domain: 'read_surface_contract',
      status: snapshotsScenesContractAvailable ? snapshotsScenesContractStatus || 'contract_available' : 'contract_missing',
      entrypoint: 'snapshots_scenes_physical_readback_contract',
      module: snapshotsScenesContractActionId || 'no_contract_action',
      uiSurface: 'Read-surface proof contract',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${snapshotsScenesContractAvailable}`,
        `schema:${snapshotsScenesContractSchemaId}`,
        `surface:${snapshotsScenesContractSurfaceId}`,
        `physical_readback:${snapshotsScenesContractPhysicalSurfaceId}`,
        `physical_action:${snapshotsScenesContractPhysicalActionId}`,
        `physical_gap_remains:${snapshotsScenesContractGapRemains}`,
        `live_authority:${snapshotsScenesContractLiveAuthority}`,
        `target_consistency:${snapshotsScenesContractTargetConsistency}`,
        `scene_load_allowed:${snapshotsScenesContractSceneLoadAllowed}`,
        `scene_save_allowed:${snapshotsScenesContractSceneSaveAllowed}`,
        `promotion_families:${snapshotsScenesContractPromotionFamilies.join('|')}`,
        `active_scene:${snapshotsScenesContractActiveFields.join('|')}`,
        `snapshots:${snapshotsScenesContractSnapshotFields.join('|')}`,
        `coverage:${snapshotsScenesContractCoverage.join('|')}`,
        `false_safety:${snapshotsScenesContractFalseSafetyFlags.join('|')}`,
      ],
      blockedReason: snapshotsScenesContractAvailable
        ? snapshotsScenesContractGapRemains
          ? 'Snapshots/scenes contract is present, but physical readback fixture evidence is still missing.'
          : 'Snapshots/scenes contract is present; physical gap state must still be reviewed before promotion.'
        : 'Snapshots/scenes physical readback contract is missing from proof summary.',
      nextStep: snapshotsScenesContractAvailable
        ? [
            snapshotsScenesContractPhysicalSurfaceId ? `physical_readback:${snapshotsScenesContractPhysicalSurfaceId}` : '',
            snapshotsScenesContractPhysicalActionId ? `physical_action:${snapshotsScenesContractPhysicalActionId}` : '',
            snapshotsScenesContractPromotionFamilies.length ? `families:${snapshotsScenesContractPromotionFamilies.join(',')}` : '',
            snapshotsScenesContractActiveFields.length ? `active_scene:${snapshotsScenesContractActiveFields.join(',')}` : '',
            snapshotsScenesContractSnapshotFields.length ? `snapshots:${snapshotsScenesContractSnapshotFields.join(',')}` : '',
            snapshotsScenesContractCoverage.length ? `coverage:${snapshotsScenesContractCoverage.join(',')}` : '',
            snapshotsScenesContractFalseSafetyFlags.length ? `false_safety:${snapshotsScenesContractFalseSafetyFlags.join(',')}` : '',
            `scene_load_allowed:${snapshotsScenesContractSceneLoadAllowed}`,
            `scene_save_allowed:${snapshotsScenesContractSceneSaveAllowed}`,
            `physical_gap_remains:${snapshotsScenesContractGapRemains}`,
            `live_authority:${snapshotsScenesContractLiveAuthority}`,
            `target_consistency:${snapshotsScenesContractTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with snapshots/scenes readback contract schema validation.',
      tone: !snapshotsScenesContractAvailable || snapshotsScenesContractLiveAuthority || !snapshotsScenesContractTargetConsistency || snapshotsScenesContractSceneLoadAllowed || snapshotsScenesContractSceneSaveAllowed
        ? 'yellow'
        : snapshotsScenesContractGapRemains
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'snapshots_scenes_physical_readback_artifact_intake',
      label: 'Snapshots/scenes readback artifact intake',
      domain: 'read_surface_artifact_intake',
      status: snapshotsScenesIntakeAvailable ? snapshotsScenesIntakeStatus || 'intake_available' : 'intake_missing',
      entrypoint: 'build_snapshots_scenes_physical_readback_artifact_intake',
      module: snapshotsScenesIntakeActionId || 'no_intake_action',
      uiSurface: 'Read-surface physical artifact intake',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${snapshotsScenesIntakeAvailable}`,
        `schema:${snapshotsScenesIntakeSchemaId}`,
        `surface:${snapshotsScenesIntakeSurfaceId}`,
        `action:${snapshotsScenesIntakeActionId}`,
        `artifacts:${snapshotsScenesIntakeArtifactCount}`,
        `valid:${snapshotsScenesIntakeValidCount}`,
        `invalid:${snapshotsScenesIntakeInvalidCount}`,
        `valid_snapshots:${snapshotsScenesIntakeValidSnapshotCount}`,
        `candidate:${snapshotsScenesIntakeCandidate}`,
        `operator_review:${snapshotsScenesIntakeOperatorReview}`,
        `physical_gap_remains:${snapshotsScenesIntakeGapRemains}`,
        `fixture_closure_allowed:${snapshotsScenesIntakeClosureAllowed}`,
        `scene_load_allowed:${snapshotsScenesIntakeSceneLoadAllowed}`,
        `scene_save_allowed:${snapshotsScenesIntakeSceneSaveAllowed}`,
        `live_authority:${snapshotsScenesIntakeLiveAuthority}`,
        `target_consistency:${snapshotsScenesIntakeTargetConsistency}`,
        snapshotsScenesIntakePromotionFamilies.length ? `promotion_families:${snapshotsScenesIntakePromotionFamilies.join('|')}` : '',
        snapshotsScenesIntakeSources.length ? `sources:${snapshotsScenesIntakeSources.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: snapshotsScenesIntakeAvailable
        ? snapshotsScenesIntakeBlockedBy.length
          ? snapshotsScenesIntakeBlockedBy.join(', ')
          : 'Snapshots/scenes artifact intake is present; manual review still gates fixture closure.'
        : 'Snapshots/scenes physical artifact intake is missing from proof summary.',
      nextStep: snapshotsScenesIntakeAvailable
        ? [
            snapshotsScenesIntakeSurfaceId ? `surface:${snapshotsScenesIntakeSurfaceId}` : '',
            snapshotsScenesIntakeActionId ? `action:${snapshotsScenesIntakeActionId}` : '',
            `artifacts:${snapshotsScenesIntakeArtifactCount}`,
            `valid:${snapshotsScenesIntakeValidCount}`,
            `invalid:${snapshotsScenesIntakeInvalidCount}`,
            `valid_snapshots:${snapshotsScenesIntakeValidSnapshotCount}`,
            `candidate:${snapshotsScenesIntakeCandidate}`,
            `operator_review:${snapshotsScenesIntakeOperatorReview}`,
            `fixture_closure_allowed:${snapshotsScenesIntakeClosureAllowed}`,
            `scene_load_allowed:${snapshotsScenesIntakeSceneLoadAllowed}`,
            `scene_save_allowed:${snapshotsScenesIntakeSceneSaveAllowed}`,
            `live_authority:${snapshotsScenesIntakeLiveAuthority}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with snapshots/scenes artifact-intake validation.',
      tone: !snapshotsScenesIntakeAvailable || snapshotsScenesIntakeInvalidCount > 0 || snapshotsScenesIntakeClosureAllowed || snapshotsScenesIntakeSceneLoadAllowed || snapshotsScenesIntakeSceneSaveAllowed || snapshotsScenesIntakeLiveAuthority || !snapshotsScenesIntakeTargetConsistency
        ? 'yellow'
        : snapshotsScenesIntakeCandidate
          ? 'yellow'
          : 'yellow'
    });
    rows.push({
      moduleId: 'user_presets_physical_readback_contract',
      label: 'User presets readback contract',
      domain: 'read_surface_contract',
      status: userPresetsContractAvailable ? userPresetsContractStatus || 'contract_available' : 'contract_missing',
      entrypoint: 'user_presets_physical_readback_contract',
      module: userPresetsContractActionId || 'no_contract_action',
      uiSurface: 'Read-surface proof contract',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${userPresetsContractAvailable}`,
        `schema:${userPresetsContractSchemaId}`,
        `surface:${userPresetsContractSurfaceId}`,
        `physical_readback:${userPresetsContractPhysicalSurfaceId}`,
        `physical_action:${userPresetsContractPhysicalActionId}`,
        `physical_gap_remains:${userPresetsContractGapRemains}`,
        `live_authority:${userPresetsContractLiveAuthority}`,
        `target_consistency:${userPresetsContractTargetConsistency}`,
        `preset_recall_allowed:${userPresetsContractPresetRecallAllowed}`,
        `preset_store_allowed:${userPresetsContractPresetStoreAllowed}`,
        `promotion_families:${userPresetsContractPromotionFamilies.join('|')}`,
        `presets:${userPresetsContractPresetFields.join('|')}`,
        `coverage:${userPresetsContractCoverage.join('|')}`,
        `false_safety:${userPresetsContractFalseSafetyFlags.join('|')}`,
      ],
      blockedReason: userPresetsContractAvailable
        ? userPresetsContractGapRemains
          ? 'User presets contract is present, but physical readback fixture evidence is still missing.'
          : 'User presets contract is present; physical gap state must still be reviewed before promotion.'
        : 'User presets physical readback contract is missing from proof summary.',
      nextStep: userPresetsContractAvailable
        ? [
            userPresetsContractPhysicalSurfaceId ? `physical_readback:${userPresetsContractPhysicalSurfaceId}` : '',
            userPresetsContractPhysicalActionId ? `physical_action:${userPresetsContractPhysicalActionId}` : '',
            userPresetsContractPromotionFamilies.length ? `families:${userPresetsContractPromotionFamilies.join(',')}` : '',
            userPresetsContractPresetFields.length ? `presets:${userPresetsContractPresetFields.join(',')}` : '',
            userPresetsContractCoverage.length ? `coverage:${userPresetsContractCoverage.join(',')}` : '',
            userPresetsContractFalseSafetyFlags.length ? `false_safety:${userPresetsContractFalseSafetyFlags.join(',')}` : '',
            `preset_recall_allowed:${userPresetsContractPresetRecallAllowed}`,
            `preset_store_allowed:${userPresetsContractPresetStoreAllowed}`,
            `physical_gap_remains:${userPresetsContractGapRemains}`,
            `live_authority:${userPresetsContractLiveAuthority}`,
            `target_consistency:${userPresetsContractTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with user presets readback contract schema validation.',
      tone: !userPresetsContractAvailable || userPresetsContractLiveAuthority || !userPresetsContractTargetConsistency || userPresetsContractPresetRecallAllowed || userPresetsContractPresetStoreAllowed
        ? 'yellow'
        : userPresetsContractGapRemains
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'user_presets_physical_readback_artifact_intake',
      label: 'User presets readback artifact intake',
      domain: 'read_surface_artifact_intake',
      status: userPresetsIntakeAvailable ? userPresetsIntakeStatus || 'intake_available' : 'intake_missing',
      entrypoint: 'build_user_presets_physical_readback_artifact_intake',
      module: userPresetsIntakeActionId || 'no_intake_action',
      uiSurface: 'Read-surface physical artifact intake',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${userPresetsIntakeAvailable}`,
        `schema:${userPresetsIntakeSchemaId}`,
        `surface:${userPresetsIntakeSurfaceId}`,
        `action:${userPresetsIntakeActionId}`,
        `artifacts:${userPresetsIntakeArtifactCount}`,
        `valid:${userPresetsIntakeValidCount}`,
        `invalid:${userPresetsIntakeInvalidCount}`,
        `valid_presets:${userPresetsIntakeValidPresetCount}`,
        `candidate:${userPresetsIntakeCandidate}`,
        `operator_review:${userPresetsIntakeOperatorReview}`,
        `physical_gap_remains:${userPresetsIntakeGapRemains}`,
        `fixture_closure_allowed:${userPresetsIntakeClosureAllowed}`,
        `preset_recall_allowed:${userPresetsIntakePresetRecallAllowed}`,
        `preset_store_allowed:${userPresetsIntakePresetStoreAllowed}`,
        `live_authority:${userPresetsIntakeLiveAuthority}`,
        `target_consistency:${userPresetsIntakeTargetConsistency}`,
        userPresetsIntakePromotionFamilies.length ? `promotion_families:${userPresetsIntakePromotionFamilies.join('|')}` : '',
        userPresetsIntakeSources.length ? `sources:${userPresetsIntakeSources.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: userPresetsIntakeAvailable
        ? userPresetsIntakeBlockedBy.length
          ? userPresetsIntakeBlockedBy.join(', ')
          : 'User presets artifact intake is present; manual review still gates fixture closure.'
        : 'User presets physical artifact intake is missing from proof summary.',
      nextStep: userPresetsIntakeAvailable
        ? [
            userPresetsIntakeSurfaceId ? `surface:${userPresetsIntakeSurfaceId}` : '',
            userPresetsIntakeActionId ? `action:${userPresetsIntakeActionId}` : '',
            `artifacts:${userPresetsIntakeArtifactCount}`,
            `valid:${userPresetsIntakeValidCount}`,
            `invalid:${userPresetsIntakeInvalidCount}`,
            `valid_presets:${userPresetsIntakeValidPresetCount}`,
            `candidate:${userPresetsIntakeCandidate}`,
            `operator_review:${userPresetsIntakeOperatorReview}`,
            `fixture_closure_allowed:${userPresetsIntakeClosureAllowed}`,
            `preset_recall_allowed:${userPresetsIntakePresetRecallAllowed}`,
            `preset_store_allowed:${userPresetsIntakePresetStoreAllowed}`,
            `live_authority:${userPresetsIntakeLiveAuthority}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with user-presets artifact-intake validation.',
      tone: !userPresetsIntakeAvailable || userPresetsIntakeInvalidCount > 0 || userPresetsIntakeClosureAllowed || userPresetsIntakePresetRecallAllowed || userPresetsIntakePresetStoreAllowed || userPresetsIntakeLiveAuthority || !userPresetsIntakeTargetConsistency
        ? 'yellow'
        : userPresetsIntakeCandidate
          ? 'yellow'
          : 'yellow'
    });
    rows.push({
      moduleId: 'dlive_state_physical_readback_contract',
      label: 'dLive state readback contract',
      domain: 'read_surface_contract',
      status: dliveStateContractAvailable ? dliveStateContractStatus || 'contract_available' : 'contract_missing',
      entrypoint: 'dlive_state_physical_readback_contract',
      module: dliveStateContractActionId || 'no_contract_action',
      uiSurface: 'Read-surface proof contract',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${dliveStateContractAvailable}`,
        `schema:${dliveStateContractSchemaId}`,
        `surface:${dliveStateContractSurfaceId}`,
        `physical_readback:${dliveStateContractPhysicalSurfaceId}`,
        `physical_action:${dliveStateContractPhysicalActionId}`,
        `physical_gap_remains:${dliveStateContractGapRemains}`,
        `live_authority:${dliveStateContractLiveAuthority}`,
        `target_consistency:${dliveStateContractTargetConsistency}`,
        `midi_send_allowed:${dliveStateContractMidiSendAllowed}`,
        `tcp_send_allowed:${dliveStateContractTcpSendAllowed}`,
        `dlive_write_allowed:${dliveStateContractDliveWriteAllowed}`,
        `dlive_apply_allowed:${dliveStateContractDliveApplyAllowed}`,
        `promotion_families:${dliveStateContractPromotionFamilies.join('|')}`,
        `channels:${dliveStateContractChannelFields.join('|')}`,
        `midi_state:${dliveStateContractMidiFields.join('|')}`,
        `parity:${dliveStateContractParityFields.join('|')}`,
        `coverage:${dliveStateContractCoverage.join('|')}`,
        `false_safety:${dliveStateContractFalseSafetyFlags.join('|')}`,
      ],
      blockedReason: dliveStateContractAvailable
        ? dliveStateContractGapRemains
          ? 'dLive state contract is present, but physical readback and WING parity fixture evidence is still missing.'
          : 'dLive state contract is present; physical gap state must still be reviewed before promotion.'
        : 'dLive state physical readback contract is missing from proof summary.',
      nextStep: dliveStateContractAvailable
        ? [
            dliveStateContractPhysicalSurfaceId ? `physical_readback:${dliveStateContractPhysicalSurfaceId}` : '',
            dliveStateContractPhysicalActionId ? `physical_action:${dliveStateContractPhysicalActionId}` : '',
            dliveStateContractPromotionFamilies.length ? `families:${dliveStateContractPromotionFamilies.join(',')}` : '',
            dliveStateContractChannelFields.length ? `channels:${dliveStateContractChannelFields.join(',')}` : '',
            dliveStateContractMidiFields.length ? `midi_state:${dliveStateContractMidiFields.join(',')}` : '',
            dliveStateContractParityFields.length ? `parity:${dliveStateContractParityFields.join(',')}` : '',
            dliveStateContractCoverage.length ? `coverage:${dliveStateContractCoverage.join(',')}` : '',
            dliveStateContractFalseSafetyFlags.length ? `false_safety:${dliveStateContractFalseSafetyFlags.join(',')}` : '',
            `midi_send_allowed:${dliveStateContractMidiSendAllowed}`,
            `tcp_send_allowed:${dliveStateContractTcpSendAllowed}`,
            `dlive_write_allowed:${dliveStateContractDliveWriteAllowed}`,
            `dlive_apply_allowed:${dliveStateContractDliveApplyAllowed}`,
            `physical_gap_remains:${dliveStateContractGapRemains}`,
            `live_authority:${dliveStateContractLiveAuthority}`,
            `target_consistency:${dliveStateContractTargetConsistency}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with dLive state readback contract schema validation.',
      tone: !dliveStateContractAvailable || dliveStateContractLiveAuthority || !dliveStateContractTargetConsistency || dliveStateContractMidiSendAllowed || dliveStateContractTcpSendAllowed || dliveStateContractDliveWriteAllowed || dliveStateContractDliveApplyAllowed
        ? 'yellow'
        : dliveStateContractGapRemains
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'dlive_state_physical_readback_artifact_intake',
      label: 'dLive state readback artifact intake',
      domain: 'read_surface_artifact_intake',
      status: dliveStateIntakeAvailable ? dliveStateIntakeStatus || 'intake_available' : 'intake_missing',
      entrypoint: 'build_dlive_state_physical_readback_artifact_intake',
      module: dliveStateIntakeActionId || 'no_intake_action',
      uiSurface: 'Read-surface physical artifact intake',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${dliveStateIntakeAvailable}`,
        `schema:${dliveStateIntakeSchemaId}`,
        `surface:${dliveStateIntakeSurfaceId}`,
        `action:${dliveStateIntakeActionId}`,
        `artifacts:${dliveStateIntakeArtifactCount}`,
        `valid:${dliveStateIntakeValidCount}`,
        `invalid:${dliveStateIntakeInvalidCount}`,
        `valid_channels:${dliveStateIntakeValidChannelCount}`,
        `candidate:${dliveStateIntakeCandidate}`,
        `operator_review:${dliveStateIntakeOperatorReview}`,
        `physical_gap_remains:${dliveStateIntakeGapRemains}`,
        `fixture_closure_allowed:${dliveStateIntakeClosureAllowed}`,
        `midi_send_allowed:${dliveStateIntakeMidiSendAllowed}`,
        `tcp_send_allowed:${dliveStateIntakeTcpSendAllowed}`,
        `dlive_write_allowed:${dliveStateIntakeDliveWriteAllowed}`,
        `dlive_apply_allowed:${dliveStateIntakeDliveApplyAllowed}`,
        `live_authority:${dliveStateIntakeLiveAuthority}`,
        `target_consistency:${dliveStateIntakeTargetConsistency}`,
        dliveStateIntakePromotionFamilies.length ? `promotion_families:${dliveStateIntakePromotionFamilies.join('|')}` : '',
        dliveStateIntakeSources.length ? `sources:${dliveStateIntakeSources.join(',')}` : '',
      ].filter(Boolean),
      blockedReason: dliveStateIntakeAvailable
        ? dliveStateIntakeBlockedBy.length
          ? dliveStateIntakeBlockedBy.join(', ')
          : 'dLive state artifact intake is present; manual review still gates fixture closure.'
        : 'dLive state physical artifact intake is missing from proof summary.',
      nextStep: dliveStateIntakeAvailable
        ? [
            dliveStateIntakeSurfaceId ? `surface:${dliveStateIntakeSurfaceId}` : '',
            dliveStateIntakeActionId ? `action:${dliveStateIntakeActionId}` : '',
            `artifacts:${dliveStateIntakeArtifactCount}`,
            `valid:${dliveStateIntakeValidCount}`,
            `invalid:${dliveStateIntakeInvalidCount}`,
            `valid_channels:${dliveStateIntakeValidChannelCount}`,
            `candidate:${dliveStateIntakeCandidate}`,
            `operator_review:${dliveStateIntakeOperatorReview}`,
            `fixture_closure_allowed:${dliveStateIntakeClosureAllowed}`,
            `midi_send_allowed:${dliveStateIntakeMidiSendAllowed}`,
            `tcp_send_allowed:${dliveStateIntakeTcpSendAllowed}`,
            `dlive_write_allowed:${dliveStateIntakeDliveWriteAllowed}`,
            `dlive_apply_allowed:${dliveStateIntakeDliveApplyAllowed}`,
            `live_authority:${dliveStateIntakeLiveAuthority}`,
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with dLive-state artifact-intake validation.',
      tone: !dliveStateIntakeAvailable || dliveStateIntakeInvalidCount > 0 || dliveStateIntakeClosureAllowed || dliveStateIntakeMidiSendAllowed || dliveStateIntakeTcpSendAllowed || dliveStateIntakeDliveWriteAllowed || dliveStateIntakeDliveApplyAllowed || dliveStateIntakeLiveAuthority || !dliveStateIntakeTargetConsistency
        ? 'yellow'
        : dliveStateIntakeCandidate
          ? 'yellow'
          : 'yellow'
    });
    rows.push({
      moduleId: 'read_surface_physical_readback_artifact_review_queue',
      label: 'Readback artifact review queue',
      domain: 'read_surface_artifact_review_queue',
      status: artifactReviewQueueAvailable ? artifactReviewQueueStatus || 'queue_available' : 'queue_missing',
      entrypoint: 'build_read_surface_physical_readback_artifact_review_queue',
      module: artifactReviewQueueBlockedBy[0] || 'operator_artifact_review_queue',
      uiSurface: 'Read-surface physical artifact review queue',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${artifactReviewQueueAvailable}`,
        `status:${artifactReviewQueueStatus}`,
        `surfaces:${artifactReviewQueueSurfaceCount}`,
        `available_intakes:${artifactReviewQueueAvailableIntakeCount}`,
        `artifacts:${artifactReviewQueueArtifactCount}`,
        `valid:${artifactReviewQueueValidCount}`,
        `invalid:${artifactReviewQueueInvalidCount}`,
        `review_candidates:${artifactReviewQueueReviewCandidateCount}`,
        `missing:${artifactReviewQueueMissingCount}`,
        `unsafe_authority:${artifactReviewQueueUnsafeCount}`,
        `target_inconsistent:${artifactReviewQueueTargetInconsistentCount}`,
        `review_available:${artifactReviewQueueReviewAvailable}`,
        `operator_review:${artifactReviewQueueOperatorReview}`,
        `physical_gap_remains:${artifactReviewQueueGapRemains}`,
        `fixture_closure_allowed:${artifactReviewQueueClosureAllowed}`,
        `live_authority:${artifactReviewQueueLiveAuthority}`,
        `full_autonomous_ready:${artifactReviewQueueFullAuto}`,
        artifactReviewQueueCandidateSurfaces.length ? `candidate_surfaces:${artifactReviewQueueCandidateSurfaces.join('|')}` : '',
        artifactReviewQueueInvalidSurfaces.length ? `invalid_surfaces:${artifactReviewQueueInvalidSurfaces.join('|')}` : '',
        artifactReviewQueueMissingSurfaces.length ? `missing_surfaces:${artifactReviewQueueMissingSurfaces.join('|')}` : '',
        artifactReviewQueueUnsafeSurfaces.length ? `unsafe_surfaces:${artifactReviewQueueUnsafeSurfaces.join('|')}` : '',
        artifactReviewQueueTargetInconsistentSurfaces.length ? `target_inconsistent_surfaces:${artifactReviewQueueTargetInconsistentSurfaces.join('|')}` : '',
        artifactReviewQueueMethodIds.length ? `methods:${artifactReviewQueueMethodIds.join('|')}` : '',
      ].filter(Boolean),
      blockedReason: artifactReviewQueueAvailable
        ? artifactReviewQueueBlockedBy.length
          ? artifactReviewQueueBlockedBy.join(', ')
          : 'Artifact review queue is present; manual review still gates fixture closure.'
        : 'Read-surface artifact review queue is missing from proof summary.',
      nextStep: artifactReviewQueueAvailable
        ? [
            `queue:${artifactReviewQueueStatus}`,
            `surfaces:${artifactReviewQueueSurfaceCount}`,
            `available_intakes:${artifactReviewQueueAvailableIntakeCount}`,
            `artifacts:${artifactReviewQueueArtifactCount}`,
            `valid:${artifactReviewQueueValidCount}`,
            `invalid:${artifactReviewQueueInvalidCount}`,
            `review_candidates:${artifactReviewQueueReviewCandidateCount}`,
            `missing:${artifactReviewQueueMissingCount}`,
            `unsafe_authority:${artifactReviewQueueUnsafeCount}`,
            `target_inconsistent:${artifactReviewQueueTargetInconsistentCount}`,
            `review_available:${artifactReviewQueueReviewAvailable}`,
            `operator_review:${artifactReviewQueueOperatorReview}`,
            `fixture_closure_allowed:${artifactReviewQueueClosureAllowed}`,
            `live_authority:${artifactReviewQueueLiveAuthority}`,
            `full_autonomous_ready:${artifactReviewQueueFullAuto}`,
            artifactReviewQueueMissingSurfaces.length ? `missing_surfaces:${artifactReviewQueueMissingSurfaces.join(',')}` : '',
            artifactReviewQueueCandidateSurfaces.length ? `candidate_surfaces:${artifactReviewQueueCandidateSurfaces.join(',')}` : '',
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with read-surface artifact review queue validation.',
      tone: !artifactReviewQueueAvailable || artifactReviewQueueInvalidCount > 0 || artifactReviewQueueUnsafeCount > 0 || artifactReviewQueueTargetInconsistentCount > 0 || artifactReviewQueueClosureAllowed || artifactReviewQueueLiveAuthority || artifactReviewQueueFullAuto
        ? 'yellow'
        : artifactReviewQueueReviewCandidateCount > 0 || artifactReviewQueueMissingCount > 0
          ? 'yellow'
          : 'green'
    });
    rows.push({
      moduleId: 'read_surface_physical_readback_fixture_closure_plan',
      label: 'Readback fixture closure plan',
      domain: 'read_surface_fixture_closure_plan',
      status: fixtureClosurePlanAvailable ? fixtureClosurePlanStatus || 'closure_plan_available' : 'closure_plan_missing',
      entrypoint: 'build_read_surface_physical_readback_fixture_closure_plan',
      module: fixtureClosurePlanBlockedBy[0] || 'manual_fixture_closure_plan',
      uiSurface: 'Read-surface physical fixture closure plan',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${fixtureClosurePlanAvailable}`,
        `status:${fixtureClosurePlanStatus}`,
        `review_candidates:${fixtureClosurePlanReviewCandidateCount}`,
        `acceptance_records:${fixtureClosurePlanAcceptanceRecordCount}`,
        `accepted:${fixtureClosurePlanAcceptedRecordCount}`,
        `rejected:${fixtureClosurePlanRejectedRecordCount}`,
        `invalid:${fixtureClosurePlanInvalidRecordCount}`,
        `unreviewed:${fixtureClosurePlanUnreviewedCount}`,
        `closure_candidates:${fixtureClosurePlanCandidateCount}`,
        `manual_candidate_available:${fixtureClosurePlanCandidateAvailable}`,
        `manual_closure_required:${fixtureClosurePlanManualRequired}`,
        `operator_review:${fixtureClosurePlanOperatorReview}`,
        `physical_gap_remains:${fixtureClosurePlanGapRemains}`,
        `fixture_update_allowed:${fixtureClosurePlanUpdateAllowed}`,
        `fixture_closure_allowed:${fixtureClosurePlanClosureAllowed}`,
        `live_authority:${fixtureClosurePlanLiveAuthority}`,
        `full_autonomous_ready:${fixtureClosurePlanFullAuto}`,
        fixtureClosurePlanCandidateSurfaces.length ? `candidate_surfaces:${fixtureClosurePlanCandidateSurfaces.join('|')}` : '',
        fixtureClosurePlanRejectedSurfaces.length ? `rejected_surfaces:${fixtureClosurePlanRejectedSurfaces.join('|')}` : '',
        fixtureClosurePlanInvalidRecordIds.length ? `invalid_records:${fixtureClosurePlanInvalidRecordIds.join('|')}` : '',
        fixtureClosurePlanUnreviewedSurfaces.length ? `unreviewed_surfaces:${fixtureClosurePlanUnreviewedSurfaces.join('|')}` : '',
      ].filter(Boolean),
      blockedReason: fixtureClosurePlanAvailable
        ? fixtureClosurePlanBlockedBy.length
          ? fixtureClosurePlanBlockedBy.join(', ')
          : 'Fixture closure plan is present; manual fixture baseline update still gates gap closure.'
        : 'Read-surface fixture closure plan is missing from proof summary.',
      nextStep: fixtureClosurePlanAvailable
        ? [
            `plan:${fixtureClosurePlanStatus}`,
            `accepted:${fixtureClosurePlanAcceptedRecordCount}`,
            `rejected:${fixtureClosurePlanRejectedRecordCount}`,
            `invalid:${fixtureClosurePlanInvalidRecordCount}`,
            `unreviewed:${fixtureClosurePlanUnreviewedCount}`,
            `closure_candidates:${fixtureClosurePlanCandidateCount}`,
            `manual_candidate_available:${fixtureClosurePlanCandidateAvailable}`,
            `manual_closure_required:${fixtureClosurePlanManualRequired}`,
            `operator_review:${fixtureClosurePlanOperatorReview}`,
            `physical_gap_remains:${fixtureClosurePlanGapRemains}`,
            `fixture_update_allowed:${fixtureClosurePlanUpdateAllowed}`,
            `fixture_closure_allowed:${fixtureClosurePlanClosureAllowed}`,
            `live_authority:${fixtureClosurePlanLiveAuthority}`,
            `full_autonomous_ready:${fixtureClosurePlanFullAuto}`,
            fixtureClosurePlanCandidateSurfaces.length ? `candidate_surfaces:${fixtureClosurePlanCandidateSurfaces.join(',')}` : '',
            fixtureClosurePlanUnreviewedSurfaces.length ? `unreviewed_surfaces:${fixtureClosurePlanUnreviewedSurfaces.join(',')}` : '',
          ].filter(Boolean).join(' / ')
        : 'Run no-WING release checks with read-surface fixture closure-plan validation.',
      tone: !fixtureClosurePlanAvailable || fixtureClosurePlanInvalidRecordCount > 0 || fixtureClosurePlanUnreviewedCount > 0 || fixtureClosurePlanUpdateAllowed || fixtureClosurePlanClosureAllowed || fixtureClosurePlanLiveAuthority || fixtureClosurePlanFullAuto
        ? 'yellow'
        : fixtureClosurePlanCandidateCount > 0
          ? 'yellow'
          : 'yellow'
    });
    return rows;
  })() : [];

  const readSurfaceRemediation = asRecord(data.read_surface_remediation_plan);
  const readSurfaceRemediationRows: ModuleStatusRecord[] = Object.keys(readSurfaceRemediation).length ? (() => {
    const summary = asRecord(readSurfaceRemediation.summary);
    const status = str(readSurfaceRemediation.status, 'unknown');
    const actionCount = Math.trunc(num(summary.action_count, 0));
    const incompleteCount = Math.trunc(num(summary.incomplete_surface_count, actionCount));
    const physicalGapIds = stringList(summary.physical_fixture_missing_surface_ids);
    const nextPhysicalReadback = str(summary.next_physical_readback_surface_id, physicalGapIds[0] ?? '');
    const nextActionId = str(summary.next_physical_readback_action_id, '');
    const globalBlockers = stringList(summary.global_blockers);
    const nextGlobalStep = str(summary.next_global_step, 'Review read-surface remediation before autonomous promotion.');
    const planAvailable = status === 'read_remediation_required' || status === 'read_coverage_complete';
    return [{
      moduleId: 'read_surface_remediation_plan',
      label: 'Read surface remediation plan',
      domain: 'read_surface_remediation',
      status,
      entrypoint: 'build_read_surface_remediation_plan',
      module: nextActionId || 'no_next_physical_action',
      uiSurface: 'Read-surface remediation plan',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `available:${planAvailable}`,
        `actions:${actionCount}`,
        `incomplete:${incompleteCount}`,
        `physical_gaps:${physicalGapIds.length}`,
        `next_physical_readback:${nextPhysicalReadback}`,
        `next_action:${nextActionId}`,
      ],
      blockedReason: globalBlockers.length
        ? globalBlockers.join(', ')
        : 'Read-surface remediation is complete or has no current blockers.',
      nextStep: [
        nextPhysicalReadback ? `next_physical_readback:${nextPhysicalReadback}` : '',
        nextActionId ? `next_action:${nextActionId}` : '',
        nextGlobalStep,
      ].filter(Boolean).join(' / '),
      tone: status === 'read_coverage_complete' ? 'green' : 'yellow'
    }];
  })() : [];

  const overlay = asRecord(data.autonomy_evidence_overlay);
  const overlayRows = asArray(overlay.rows).map(row => {
    const valueType = str(row.value_type, 'unknown');
    const targetId = str(row.target_id, 'unknown_target');
    const status = str(row.status, 'unknown');
    const blockers = stringList(row.blockers);
    const nextActions = stringList(row.next_required_actions);
    const physicalIds = stringList(row.physical_candidate_proof_ids);
    const dryRunIds = stringList(row.dry_run_ready_proof_ids);
    const blockedIds = stringList(row.blocked_proof_ids);
    const remediationIds = stringList(row.remediation_action_ids);
    const proofCounts = [
      `physical:${Math.trunc(num(row.physical_candidate_count, physicalIds.length))}`,
      `dry_run:${Math.trunc(num(row.dry_run_ready_count, dryRunIds.length))}`,
      `blocked:${Math.trunc(num(row.blocked_artifact_count, blockedIds.length))}`,
      `remediation:${Math.trunc(num(row.remediation_action_count, remediationIds.length))}`,
    ];
    const proofText = [
      physicalIds.length ? `Physical: ${physicalIds.join(', ')}` : '',
      dryRunIds.length ? `Dry-run: ${dryRunIds.join(', ')}` : '',
      blockedIds.length ? `Blocked: ${blockedIds.join(', ')}` : '',
      remediationIds.length ? `Remediate: ${remediationIds.join(', ')}` : '',
    ].filter(Boolean).join(' / ');
    return {
      moduleId: `autonomy_evidence:${targetId}`,
      label: `Autonomy proof evidence: ${titleFromToken(valueType)}`,
      domain: 'autonomy_evidence_overlay',
      status,
      entrypoint: valueType,
      module: targetId,
      uiSurface: 'Autonomy proof evidence overlay',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [`target:${targetId}`, ...proofCounts],
      blockedReason: blockers.length
        ? blockers.join(', ')
        : 'Overlay is read-only; manual safety review is still required before promotion.',
      nextStep: [nextActions.join(' / '), proofText || 'Collect proof artifacts before promotion.']
        .filter(Boolean)
        .join(' / '),
      tone: toneForAutonomyEvidenceOverlayStatus(status)
    };
  });

  const gapCounts = asRecord(gapSummary.gap_bucket_counts);
  const blockedByBucket = asRecord(gapSummary.blocked_target_ids_by_bucket);
  const evidenceByFamily = asRecord(gapSummary.evidence_requirement_ids_by_action_family);
  const nextByFamily = asRecord(gapSummary.next_required_evidence_by_action_family);
  const evidenceText = Object.entries(evidenceByFamily)
    .map(([family, values]) => {
      const ids = stringList(values);
      return ids.length ? `${family}: ${ids.join(', ')}` : '';
    })
    .filter(Boolean)
    .join(' / ');
  const nextEvidenceText = Object.entries(nextByFamily)
    .map(([family, values]) => {
      const ids = stringList(values);
      return ids.length ? `${family}: ${ids.join(', ')}` : '';
    })
    .filter(Boolean)
    .join(' / ');
  const gapRows: ModuleStatusRecord[] = [];
  Object.entries(gapCounts).forEach(([bucket, rawCount]) => {
    const count = Math.trunc(num(rawCount, 0));
    if (count <= 0) return;
    const blockedTargets = stringList(blockedByBucket[bucket]);
    const nextStep = [
      bucket === 'evidence_requirement' && evidenceText ? `Evidence: ${evidenceText}` : '',
      nextEvidenceText ? `Next: ${nextEvidenceText}` : 'Review promotion blockers before widening automation.'
    ].filter(Boolean).join(' / ');
    gapRows.push({
      moduleId: `promotion_gap:${bucket}`,
      label: `Autonomy gap: ${titleFromToken(bucket)}`,
      domain: 'promotion_gap',
      status: 'blocked',
      entrypoint: bucket,
      module: blockedTargets.length ? blockedTargets.join(', ') : 'all_promotion_targets',
      uiSurface: 'Autonomy promotion gap summary',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [`gap_count:${count}`, ...blockedTargets],
      blockedReason: blockedTargets.length
        ? `${count} gap(s): ${blockedTargets.join(', ')}`
        : `${count} promotion evidence gap(s).`,
      nextStep,
      tone: 'red'
    });
  });

  return [
    ...moduleRows,
    ...surfaceRows,
    ...coverageRows,
    ...projectGoalCapabilityRows,
    ...projectGoalCapabilitySummaryRows,
    ...progressSummaryRows,
    ...objectiveAlignmentRows,
    ...modeGateRows,
    ...readinessRows,
    ...proofEvidenceRows,
    ...readSurfaceRemediationRows,
    ...overlayRows,
    ...gapRows,
    ...readinessSummaryRows
  ];
}

export function adaptTelemetryAlertsSnapshot(payload: unknown): TelemetryAlertsSnapshot {
  const data = asRecord(payload);
  const summary = asRecord(data.summary);
  const safety = asRecord(data.safety);
  const available = Boolean(data.available);
  const latestSessionId = str(summary.latest_session_id);
  const alerts = asArray(data.alerts).map((row, index) => {
    const sessionId = str(row.session_id, latestSessionId);
    const rule = cleanToken(row.rule, 'telemetry alert');
    const severity = str(row.severity, 'info').toLowerCase();
    const targetType = cleanToken(row.target_type, 'target');
    const targetId = str(row.target_id, 'unknown');
    return {
      id: `${sessionId}:${rule}:${targetType}:${targetId}:${str(row.timestamp_ms, String(index))}`,
      sessionId,
      time: telemetryTimeLabel(row.timestamp_ms, data.generated_at),
      rule,
      severity,
      target: `${targetType}:${targetId}`,
      message: str(row.message, 'Telemetry alert requires review.'),
      value: formatTelemetryNumber(row.value),
      threshold: formatTelemetryNumber(row.threshold),
      tone: toneForTelemetrySeverity(severity)
    };
  });
  const sessions = asArray(data.sessions).map(row => {
    const sessionSafety = asRecord(row.safety);
    return {
      sessionId: str(row.session_id, 'unknown_session'),
      generatedAt: str(row.generated_at, 'n/a'),
      eventCount: Math.trunc(num(row.event_count, 0)),
      alertCount: Math.trunc(num(row.alert_count, 0)),
      dashboardRowCount: Math.trunc(num(row.dashboard_row_count, 0)),
      readOnly: sessionSafety.read_only !== false,
      liveWrite: Boolean(sessionSafety.live_write)
    };
  });

  return {
    available,
    generatedAt: str(data.generated_at, available ? 'runtime' : 'mock'),
    sourceLabel: str(data.source, available ? 'telemetry_alert_artifacts' : 'mock'),
    storageDir: str(data.storage_dir),
    alertCount: Math.trunc(num(summary.alert_count, alerts.length)),
    criticalCount: Math.trunc(num(summary.critical_count, alerts.filter(row => row.severity === 'critical').length)),
    warningCount: Math.trunc(num(summary.warning_count, alerts.filter(row => row.severity === 'warning').length)),
    latestSessionId,
    readOnly: safety.read_only !== false,
    liveWrite: Boolean(safety.live_write),
    generatesArtifacts: Boolean(safety.generates_artifacts),
    sessions,
    alerts
  };
}

export function adaptBusesFromChannels(channels: Channel[], fallback: BusRecord[]): BusRecord[] {
  if (!channels.length) return fallback;

  const groups = new Map<string, Channel[]>();
  channels.forEach(channel => {
    const key = channel.group?.trim() || 'UNASSIGNED';
    const rows = groups.get(key) ?? [];
    rows.push(channel);
    groups.set(key, rows);
  });

  return Array.from(groups.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([group, rows]) => {
      const activeChannels = rows.filter(channel => channel.status === 'ACTIVE').length;
      const mutedChannels = rows.filter(channel => channel.status === 'MUTED').length;
      return {
        name: busName(group),
        role: busRole(group),
        level: avg(rows.map(channel => channel.level), -60),
        lufs: avg(rows.map(channel => channel.lufs), -60),
        channels: rows.length,
        activeChannels,
        mutedChannels,
        dynamic: avg(rows.map(channel => channel.dynamic), 0),
        confidence: Math.round(avg(rows.map(channel => channel.confidence), 0)),
        status: busStatus(rows),
        color: busTone(group),
        members: rows.map(channel => `${String(channel.id).padStart(2, '0')} ${channel.name}`)
      };
    });
}

export function adaptBusesSnapshot(payload: unknown, fallback: BusRecord[]): BusRecord[] {
  const data = asRecord(payload);
  const hasBusesField = Array.isArray(data.buses);
  const rows = asArray(data.buses);
  if (!rows.length) return hasBusesField ? [] : fallback;

  return rows.map((row, index) => {
    const base = fallback[index] ?? fallback[fallback.length - 1];
    const name = str(row.name, base?.name ?? `Bus ${index + 1}`);
    return {
      name,
      role: str(row.role, base?.role ?? 'Group'),
      level: num(row.level ?? row.level_db, base?.level ?? -60),
      lufs: num(row.lufs, base?.lufs ?? -60),
      channels: Math.trunc(num(row.channels ?? row.channel_count, base?.channels ?? 0)),
      activeChannels: Math.trunc(num(row.activeChannels ?? row.active_channels, base?.activeChannels ?? 0)),
      mutedChannels: Math.trunc(num(row.mutedChannels ?? row.muted_channels, base?.mutedChannels ?? 0)),
      dynamic: num(row.dynamic ?? row.dynamic_range_db, base?.dynamic ?? 0),
      confidence: Math.round(num(row.confidence, base?.confidence ?? 0)),
      status: str(row.status, base?.status ?? 'Inactive'),
      color: toneValue(row.color, busTone(name)),
      members: Array.isArray(row.members)
        ? row.members.map(item => String(item))
        : base?.members ?? []
    };
  });
}

export function adaptTasks(payload: unknown, fallback: Task[]): Task[] {
  const data = asRecord(payload);
  const pending = asArray(data.pending_actions);
  const history = asArray(data.history).slice(-10).reverse();
  const rows = [...pending, ...history];
  if (!rows.length) return fallback;

  return rows.map((row, index) => {
    const base = fallback[index] ?? fallback[fallback.length - 1];
    const raw = asRecord(row.raw);
    const requestedChange = asRecord(row.requested_change);
    const canApply = Boolean(row.can_apply ?? requestedChange.value_type);
    const id = str(row.id, base?.id ?? `P-${index + 1}`);
    return {
      id,
      proposalId: id,
      priority: priorityFromSeverity(row.severity ?? raw.severity),
      title: str(row.title, base?.title ?? 'Operator proposal'),
      agent: str(row.source, base?.agent ?? 'Product Layer'),
      status: taskStatusFromBackend(row.status),
      time: timeLabel(row.updated_at ?? row.created_at, base?.time ?? 'now'),
      target: str(row.target, String(base?.target ?? requestedChange.channel ?? 'Session')),
      canApply,
      backendStatus: str(row.status, 'pending'),
      reason: str(row.reason, str(raw.reason, '')),
      applyBlockers: Array.isArray(row.apply_blockers)
        ? row.apply_blockers.map(item => String(item)).filter(Boolean)
        : undefined,
      requestedValueType: str(requestedChange.value_type),
      safetyEvidence: adaptSafetyEvidence(requestedChange.safety_evidence)
    };
  });
}

export function adaptProposalOnlyContractTasks(payload: unknown): Task[] {
  const data = asRecord(payload);
  const rows = asArray(data.proposal_only_contracts);
  if (!rows.length) return [];

  return rows.map((row, index) => {
    const contract = asRecord(row.proposal_contract);
    const valueType = str(contract.value_type, str(row.value_type, 'review'));
    const blockers = Array.isArray(row.blockers)
      ? row.blockers.map(item => String(item)).filter(Boolean)
      : [`unsupported_value_type:${valueType}`];
    const methodLabel = str(row.method_label, str(row.method_id, 'Correction method'));
    const domain = str(row.domain, 'review');
    const id = str(row.id, `proposal-only-${index + 1}`);
    return {
      id,
      proposalId: id,
      priority: row.risk === 'blocked' ? 'High' : 'Medium',
      title: str(row.title, `${methodLabel}: proposal contract only`),
      agent: methodLabel,
      status: 'Review',
      time: timeLabel(data.generated_at, 'review'),
      target: cleanToken(row.target, cleanToken(contract.target_kind, domain)),
      canApply: false,
      backendStatus: 'proposal_only',
      reason: str(row.reason, 'Correction exists but live apply is not promoted.'),
      reviewOnly: true,
      applyBlockers: blockers,
      correctionDomain: domain,
      requestedValueType: valueType,
      proposalContract: {
        valueType,
        targetKind: str(contract.target_kind, str(row.target, domain)),
        requiredFields: Array.isArray(contract.required_fields)
          ? contract.required_fields.map(item => String(item))
          : [],
        oldValueSource: str(contract.old_value_source, 'readback_required_before_promotion'),
        safetyEvidenceRequired: Array.isArray(contract.safety_evidence_required)
          ? contract.safety_evidence_required.map(item => String(item))
          : [],
        requiredApproval: str(contract.required_approval, 'operator_supervised'),
        readbackRequired: Boolean(contract.readback_required),
        rollbackRequired: Boolean(contract.rollback_required),
        liveApplyAllowed: Boolean(contract.live_apply_allowed),
        allowedLiveWriteKinds: Array.isArray(contract.allowed_live_write_kinds)
          ? contract.allowed_live_write_kinds.map(item => String(item))
          : [],
        promotionNext: str(contract.promotion_next, 'hardware_readback_and_rollback_required')
      }
    };
  });
}

export function adaptAgents(payload: unknown, fallback: Agent[]): Agent[] {
  const data = asRecord(payload);
  const rows = asArray(data.agents);
  if (!rows.length) return fallback;

  return rows.map((row, index) => {
    const base = fallback[index] ?? fallback[fallback.length - 1];
    const status = agentStatusFromBackend(row.status);
    const confidence = Math.round(num(row.confidence, base?.confidence ?? 0));
    return {
      id: str(row.id, base?.id ?? `agent-${index + 1}`),
      name: str(row.name, base?.name ?? 'Agent'),
      role: str(row.role, base?.role ?? 'Runtime role'),
      status,
      task: str(row.task, base?.task ?? 'Runtime state'),
      confidence,
      activity: base?.activity ?? [24, 48, 36, 62, 54, 76, 58, 84, 64, 72],
      lastAction: str(row.blocked_reason, str(row.category, base?.lastAction ?? 'Backend update')),
      tone: status === 'BLOCKED' ? 'red' : status === 'UNAVAILABLE' ? 'yellow' : toneForBackendStatus(str(row.tone, str(row.status, 'active')))
    };
  });
}

export function adaptOverviewMetrics(
  dashboard: unknown,
  agentRoster: unknown,
  decisionQueue: unknown,
  fallback: Metric[]
): Metric[] {
  const dash = asRecord(dashboard);
  const agentSummary = asRecord(asRecord(agentRoster).summary);
  const decisionSummary = asRecord(asRecord(decisionQueue).summary ?? dash.decision_summary);
  const channelSummary = asRecord(dash.channel_summary);
  const safety = asRecord(dash.safety);
  if (!Object.keys(dash).length && !Object.keys(agentSummary).length && !Object.keys(decisionSummary).length) {
    return fallback;
  }

  const metrics: Metric[] = [
    { label: 'Active Agents', value: String(num(agentSummary.active_count, 0)), sub: `${num(agentSummary.agent_count, 0)} total agents`, tone: 'green' },
    { label: 'Decisions Today', value: String(num(decisionSummary.history_count, 0)), sub: `${num(decisionSummary.pending_count, 0)} pending`, tone: 'blue' },
    { label: 'AI Confidence', value: `${Math.max(0, Math.min(100, Math.round(num(agentSummary.can_live_execute, 0) ? 92 : 82)))}%`, sub: str(safety.live_write_policy, 'backend policy'), tone: 'green' },
    { label: 'Suggestions Pending', value: String(num(decisionSummary.pending_count, 0)), sub: 'backend queue', tone: 'yellow' },
    { label: 'Actions Applied', value: String(num(decisionSummary.applied_count, 0)), sub: 'proposal history', tone: 'green' },
    { label: 'Warnings', value: String(num(agentSummary.blocked_count, 0) + num(decisionSummary.blocked_count, 0)), sub: 'blocked / review', tone: 'yellow' },
    { label: 'Channels', value: String(num(channelSummary.total_channels, 0)), sub: `${num(channelSummary.active_channels, 0)} active`, tone: 'blue' }
  ];
  return metrics.slice(0, fallback.length);
}

export function adaptDashboardRuntime(
  dashboard: unknown,
  agentRoster: unknown,
  decisionQueue: unknown
): DashboardRuntime {
  const dash = asRecord(dashboard);
  const master = asRecord(dash.master_bus);
  const analyzer = asRecord(dash.analyzer ?? dash.analysis ?? dash.spectrum);
  const channelSummary = asRecord(dash.channel_summary);
  const decisionSummary = asRecord(asRecord(decisionQueue).summary ?? dash.decision_summary);
  const roster = asRecord(agentRoster);
  const agentSummary = asRecord(roster.summary);
  const agentRows = asArray(roster.agents);
  const hasRuntime = Boolean(
    Object.keys(dash).length ||
    Object.keys(agentSummary).length ||
    Object.keys(decisionSummary).length
  );

  if (!hasRuntime) {
    return {
      integratedLufs: -14.2,
      truePeakDb: -1.1,
      shortTermLufs: -13.8,
      momentaryLufs: -12.1,
      lraLu: 6.7,
      aiConfidencePercent: 87,
      activeChannels: 36,
      totalChannels: 48,
      pendingDecisions: 23,
      appliedDecisions: 12,
      blockedDecisions: 3,
      sourceLabel: 'Mock'
    };
  }

  const confidenceValues = agentRows
    .map(row => metricNum(row.confidence, Number.NaN))
    .filter(value => Number.isFinite(value));
  const averageConfidence = confidenceValues.length
    ? confidenceValues.reduce((sum, value) => sum + value, 0) / confidenceValues.length
    : Boolean(agentSummary.can_live_execute) ? 92 : 82;
  const spectrumBins = firstNumericList(
    analyzer.spectrum_bins,
    analyzer.spectrumBins,
    analyzer.bins,
    dash.spectrum_bins,
    master.spectrum_bins
  );
  const analysisTraces = adaptDashboardAnalysisTraces(analyzer, master);

  return {
    integratedLufs: metricNum(master.integrated_lufs, -14.2),
    truePeakDb: metricNum(master.true_peak_db, -1.1),
    shortTermLufs: metricNum(master.short_term_lufs, metricNum(master.integrated_lufs, -13.8)),
    momentaryLufs: metricNum(master.momentary_lufs, metricNum(master.short_term_lufs, -12.1)),
    lraLu: metricNum(master.lra_lu, 6.7),
    aiConfidencePercent: Math.max(0, Math.min(100, Math.round(averageConfidence))),
    activeChannels: Math.trunc(metricNum(channelSummary.active_channels, 0)),
    totalChannels: Math.trunc(metricNum(channelSummary.total_channels, 0)),
    pendingDecisions: Math.trunc(metricNum(decisionSummary.pending_count, 0)),
    appliedDecisions: Math.trunc(metricNum(decisionSummary.applied_count, 0)),
    blockedDecisions: Math.trunc(metricNum(decisionSummary.blocked_count, 0)),
    sourceLabel: 'Backend',
    spectrumBins,
    ...(Object.keys(analysisTraces).length ? { analysisTraces } : {})
  };
}

function booleanLabel(value: unknown, trueLabel: string, falseLabel: string): string {
  return Boolean(value) ? trueLabel : falseLabel;
}

export function adaptDirectorRuntime(
  dashboard: unknown,
  agentRoster: unknown,
  pipelinePreview?: unknown
): DirectorRuntime {
  const dash = asRecord(dashboard);
  const safety = asRecord(dash.safety);
  const readyForLive = asRecord(safety.ready_for_live);
  const readyChecks = asRecord(readyForLive.checks);
  const manual = asRecord(safety.manual_write_supervision);
  const audioCapture = asRecord(dash.audio_capture);
  const runtime = asRecord(dash.runtime);
  const processing = asRecord(runtime.processing);
  const preview = asRecord(pipelinePreview);
  const previewSummary = asRecord(preview.summary);
  const previewExecutor = asRecord(preview.executor_policy);
  const rosterSummary = asRecord(asRecord(agentRoster).summary);
  const hasRuntime = Boolean(
    Object.keys(dash).length ||
    Object.keys(safety).length ||
    Object.keys(rosterSummary).length ||
    Object.keys(preview).length
  );

  if (!hasRuntime) {
    return {
      currentGoal: 'Maintain vocal intelligibility and balanced live mix while keeping all writes supervised.',
      readyLabel: 'Mock only',
      readyTone: 'red',
      audioLabel: 'Mock meters',
      audioTone: 'yellow',
      proposalQueueLabel: 'Mock queue',
      proposalQueueTone: 'blue',
      liveWritePolicy: 'mock_frontend_only',
      manualGateLabel: 'UI only',
      manualGateTone: 'yellow',
      resetRequired: false,
      hardwareProofItems: [
        { label: 'Fader proof', value: 'Not live proven', tone: 'yellow' },
        { label: 'Readback', value: 'Not live proven', tone: 'yellow' },
        { label: 'Rollback', value: 'Not live proven', tone: 'yellow' },
        { label: 'Final readiness', value: 'Mock only', tone: 'red' }
      ],
      blockedReasons: [],
      statusItems: [
        { label: 'Mixer connected', value: 'No', tone: 'red' },
        { label: 'Gain staging', value: 'Mock', tone: 'blue' },
        { label: 'Auto fader', value: 'Mock', tone: 'blue' },
        { label: 'No clipping', value: 'Unverified', tone: 'yellow' }
      ],
      sourceLabel: 'Mock'
    };
  }

  const ready = Boolean(readyForLive.ready);
  const audioRunning = Boolean(audioCapture.running);
  const proposalQueueAvailable = Boolean(
    runtime.proposal_queue_available ?? rosterSummary.proposal_queue_available
  );
  const canLiveExecute = Boolean(rosterSummary.can_live_execute);
  const emergencyStopped = Boolean(manual.emergency_stopped);
  const liveWritePolicy = str(safety.live_write_policy, 'backend_read_only');
  const mainPipeline = str(preview.pipeline_id, str(processing.main_pipeline_id, 'main_live_assist'));
  const pipelineStatus = str(preview.pipeline_status, str(processing.main_pipeline_status, 'unknown'));
  const previewStatus = str(preview.status, 'not requested');
  const previewProposalCount = Math.trunc(metricNum(previewSummary.proposal_count, 0));
  const previewBlockerCount = Math.trunc(metricNum(previewSummary.blocker_count, 0));
  const previewQueueMutation = Boolean(previewSummary.queue_mutation ?? previewExecutor.queue_mutation);
  const liveApplyPolicy = str(previewExecutor.live_apply_policy, str(processing.live_apply_policy, liveWritePolicy));
  const readyBlockedReasons = Array.isArray(readyForLive.blocked_reasons)
    ? readyForLive.blocked_reasons.map(item => String(item)).filter(Boolean)
    : [];
  const policyBlockedReasons = Array.isArray(safety.blocked_reasons)
    ? safety.blocked_reasons.map(item => String(item)).filter(Boolean)
    : [];
  const blockedReasons = Array.from(new Set([...policyBlockedReasons, ...readyBlockedReasons]));
  const resetRequired = Boolean(manual.reset_required ?? emergencyStopped);
  const automationUnfrozen = readyChecks.automation_unfrozen !== false;

  return {
    currentGoal: emergencyStopped
      ? 'Emergency stop is active. Live apply remains blocked until the operator resets the latch and reviews frozen automation.'
      : canLiveExecute
      ? 'Supervised gate is available for approved fader/gain proposals. The UI still avoids direct console writes.'
      : 'Collect runtime evidence, keep proposals review-only, and wait for the supervised gate before any console apply.',
    readyLabel: ready ? 'Ready for live' : 'Not ready',
    readyTone: emergencyStopped ? 'red' : ready ? 'green' : 'yellow',
    audioLabel: audioRunning ? 'Audio capture running' : 'Audio capture offline',
    audioTone: audioRunning ? 'green' : 'yellow',
    proposalQueueLabel: proposalQueueAvailable ? 'Proposal queue online' : 'Proposal queue unavailable',
    proposalQueueTone: proposalQueueAvailable ? 'green' : 'yellow',
    liveWritePolicy,
    manualGateLabel: emergencyStopped
      ? 'Reset required'
      : canLiveExecute ? 'Supervised gate ready' : 'Apply blocked',
    manualGateTone: emergencyStopped ? 'red' : canLiveExecute ? 'green' : 'yellow',
    resetRequired,
    hardwareProofItems: [
      {
        label: 'Fader proof',
        value: emergencyStopped ? 'Blocked after proof' : 'Supervised-only',
        tone: emergencyStopped ? 'red' : canLiveExecute ? 'green' : 'yellow'
      },
      {
        label: 'Readback',
        value: 'Required for apply',
        tone: 'yellow'
      },
      {
        label: 'Rollback',
        value: 'Required before ready',
        tone: ready ? 'green' : 'yellow'
      },
      {
        label: 'Final readiness',
        value: ready ? 'Ready' : 'Partial / blocked',
        tone: ready ? 'green' : 'red'
      }
    ],
    blockedReasons,
    statusItems: [
      {
        label: 'Mixer connected',
        value: booleanLabel(readyChecks.mixer_connected, 'Yes', 'No'),
        tone: readyChecks.mixer_connected ? 'green' : 'yellow'
      },
      {
        label: 'Gain staging',
        value: booleanLabel(readyChecks.gain_staging_available, 'Available', 'Missing'),
        tone: readyChecks.gain_staging_available ? 'green' : 'yellow'
      },
      {
        label: 'Auto fader',
        value: booleanLabel(readyChecks.auto_fader_available, 'Available', 'Offline'),
        tone: readyChecks.auto_fader_available ? 'green' : 'yellow'
      },
      {
        label: 'No clipping',
        value: booleanLabel(readyChecks.no_clipping, 'Clear', 'Risk'),
        tone: readyChecks.no_clipping !== false ? 'green' : 'red'
      },
      {
        label: 'Emergency reset',
        value: resetRequired ? 'Required' : 'Clear',
        tone: resetRequired ? 'red' : 'green'
      },
      {
        label: 'Automation freeze',
        value: automationUnfrozen ? 'Clear' : 'Frozen',
        tone: automationUnfrozen ? 'green' : 'yellow'
      },
      {
        label: 'Pipeline',
        value: `${mainPipeline} · ${pipelineStatus}`,
        tone: pipelineStatus === 'canonical' ? 'green' : 'yellow'
      },
      {
        label: 'Pipeline preview',
        value: `${previewStatus} · ${previewProposalCount} proposal(s)`,
        tone: previewQueueMutation ? 'red' : previewStatus === 'ok' || previewStatus === 'clear' ? 'green' : 'yellow'
      },
      {
        label: 'Unavailable methods',
        value: `${previewBlockerCount} blocked`,
        tone: previewBlockerCount ? 'yellow' : 'green'
      },
      {
        label: 'Preview queue writes',
        value: previewQueueMutation ? 'Enabled' : 'Disabled',
        tone: previewQueueMutation ? 'red' : 'green'
      },
      {
        label: 'Apply policy',
        value: liveApplyPolicy,
        tone: liveApplyPolicy === 'supervised_fader_gain_only' ? 'green' : 'yellow'
      }
    ],
    sourceLabel: 'Backend'
  };
}

export function adaptPerformanceRecords(
  agentRoster: unknown,
  decisionQueue: unknown,
  fallback: PerformanceRecord[]
): PerformanceRecord[] {
  const roster = asRecord(agentRoster);
  const hasAgentsField = Array.isArray(roster.agents);
  const rows = asArray(roster.agents);
  if (!rows.length) return hasAgentsField ? [] : fallback;

  const summary = asRecord(asRecord(decisionQueue).summary);
  const pendingCount = Math.trunc(metricNum(summary.pending_count, 0));
  const historyCount = Math.trunc(metricNum(summary.history_count, 0));

  return rows.map((row, index) => {
    const confidence = Math.max(0, Math.min(100, Math.round(metricNum(row.confidence, 0))));
    const status = str(row.status, 'UNKNOWN').toUpperCase();
    const category = str(row.category, 'runtime');
    const canWrite = Boolean(row.can_write);
    const blockedReason = str(row.blocked_reason);
    const baseCpu = status === 'ACTIVE' ? 10 + (index % 5) * 3 : status === 'BLOCKED' ? 2 : 5;
    const memoryMb = 360 + index * 64 + (category === 'analysis' ? 96 : 0);
    const latencyMs = status === 'ACTIVE' ? 10 + index * 3 : status === 'BLOCKED' ? 0 : 25 + index * 4;
    const decisions = Math.max(0, historyCount + pendingCount + index * 2);
    const statusLabel = blockedReason
      ? `Blocked: ${blockedReason}`
      : canWrite ? 'Gate ready' : status.charAt(0) + status.slice(1).toLowerCase();

    return {
      agent: str(row.name, `Runtime Agent ${index + 1}`),
      cpu: `${baseCpu}%`,
      memory: memoryMb >= 1024 ? `${(memoryMb / 1024).toFixed(1)} GB` : `${memoryMb} MB`,
      decisions,
      success: `${confidence}%`,
      latency: latencyMs ? `${latencyMs} ms` : 'held',
      status: statusLabel
    };
  });
}

export function adaptRuntimeCommunications(
  logs: LogRecord[],
  agents: Agent[],
  tasks: Task[],
  fallback: Communication[]
): Communication[] {
  if (!logs.length && !tasks.length && !agents.length) return fallback;

  const logRows: Communication[] = logs.slice(0, 16).map(log => ({
    time: log.time.slice(0, 5),
    from: log.source,
    to: 'Automixer Director',
    topic: log.message,
    status: communicationStatusFromLogLevel(log.level),
    tone: toneFromLogLevel(log.level)
  }));

  const taskRows: Communication[] = tasks.slice(0, 8).map(task => ({
    time: task.time,
    from: task.agent,
    to: 'Automixer Director',
    topic: task.title,
    status: task.status === 'Applied' ? 'Approved' : task.status === 'Rejected' ? 'Blocked' : task.status,
    tone: task.status === 'Applied'
      ? 'green'
      : task.status === 'Rejected'
        ? 'red'
        : task.priority === 'High'
          ? 'yellow'
          : 'blue'
  }));

  const agentRows: Communication[] = agents.slice(0, 6).map(agent => ({
    time: 'now',
    from: agent.name,
    to: 'Automixer Director',
    topic: agent.task,
    status: agent.status === 'ACTIVE' ? 'Approved' : agent.status === 'BLOCKED' ? 'Blocked' : 'Review',
    tone: agent.tone
  }));

  return [...logRows, ...taskRows, ...agentRows].slice(0, 30);
}

export function adaptCommunicationsSnapshot(payload: unknown, fallback: Communication[]): Communication[] {
  const data = asRecord(payload);
  const hasCommunicationsField = Array.isArray(data.communications);
  const rows = asArray(data.communications);
  if (!rows.length) return hasCommunicationsField ? [] : fallback;

  return rows.map((row, index) => {
    const base = fallback[index] ?? fallback[fallback.length - 1];
    return {
      time: timeLabel(row.time ?? row.generated_at, base?.time ?? 'now'),
      from: str(row.from, base?.from ?? 'Agent'),
      to: str(row.to, base?.to ?? 'Automixer Director'),
      topic: str(row.topic, base?.topic ?? 'Runtime communication'),
      status: str(row.status, base?.status ?? 'Info'),
      tone: toneValue(row.tone, base?.tone ?? 'blue')
    };
  });
}

export function adaptRuntimeMemories(
  agents: Agent[],
  tasks: Task[],
  channels: Channel[],
  logs: LogRecord[],
  fallback: MemoryRecord[]
): MemoryRecord[] {
  if (!agents.length && !tasks.length && !channels.length && !logs.length) return fallback;

  const agentRows: MemoryRecord[] = agents.slice(0, 8).map(agent => ({
    name: `${agent.name} runtime state`,
    type: 'Agent State',
    size: `${Math.max(1.2, agent.confidence / 22).toFixed(1)} KB`,
    confidence: agent.confidence,
    updated: agent.lastAction || 'now'
  }));

  const taskRows: MemoryRecord[] = tasks.slice(0, 8).map(task => ({
    name: `${task.target ?? task.title} decision memory`,
    type: task.status === 'Applied' ? 'Decision History' : 'Decision Draft',
    size: `${task.priority === 'High' ? '3.2' : task.priority === 'Medium' ? '2.4' : '1.6'} KB`,
    confidence: task.priority === 'High' ? 94 : task.priority === 'Medium' ? 88 : 82,
    updated: task.time
  }));

  const channelRows: MemoryRecord[] = channels.slice(0, 8).map(channel => ({
    name: `${channel.name} channel profile`,
    type: `${channel.group} Profile`,
    size: `${Math.max(1.1, channel.dynamic / 2).toFixed(1)} KB`,
    confidence: Math.round(channel.confidence),
    updated: 'inventory'
  }));

  const logRows: MemoryRecord[] = logs.slice(0, 5).map(log => ({
    name: log.message,
    type: `${log.source} Event`,
    size: '0.8 KB',
    confidence: log.level === 'ERROR' ? 58 : log.level === 'WARN' ? 76 : 92,
    updated: log.time.slice(0, 5)
  }));

  return [...agentRows, ...taskRows, ...channelRows, ...logRows].slice(0, 32);
}

export function adaptMemorySnapshot(payload: unknown, fallback: MemoryRecord[]): MemoryRecord[] {
  const data = asRecord(payload);
  const hasMemoriesField = Array.isArray(data.memories);
  const rows = asArray(data.memories);
  if (!rows.length) return hasMemoriesField ? [] : fallback;

  return rows.map((row, index) => {
    const base = fallback[index] ?? fallback[fallback.length - 1];
    return {
      name: str(row.name, base?.name ?? `Memory ${index + 1}`),
      type: str(row.type, base?.type ?? 'Runtime'),
      size: str(row.size, base?.size ?? '1.0 KB'),
      confidence: Math.round(num(row.confidence, base?.confidence ?? 70)),
      updated: str(row.updated, base?.updated ?? 'runtime')
    };
  });
}

export function adaptTrainingSnapshot(
  payload: unknown,
  fallback: TrainingSessionSnapshotRow[]
): TrainingSessionSnapshotRow[] {
  const data = asRecord(payload);
  const hasSessionsField = Array.isArray(data.sessions);
  const rows = asArray(data.sessions);
  if (!rows.length) return hasSessionsField ? [] : fallback;

  return rows.map((row, index) => {
    const base = fallback[index] ?? fallback[fallback.length - 1];
    return {
      name: str(row.name, base?.name ?? `Training Session ${index + 1}`),
      status: str(row.status, base?.status ?? 'Queued'),
      examples: Math.trunc(num(row.examples, base?.examples ?? 0)),
      score: str(row.score, base?.score ?? 'N/A'),
      updated: str(row.updated, base?.updated ?? 'runtime')
    };
  });
}

export function adaptRulesSnapshot(
  payload: unknown,
  fallback: RuleLibrarySnapshotRow[]
): RuleLibrarySnapshotRow[] {
  const data = asRecord(payload);
  const hasRulesField = Array.isArray(data.rules);
  const rows = asArray(data.rules);
  if (!rows.length) return hasRulesField ? [] : fallback;

  return rows.map((row, index) => {
    const base = fallback[index] ?? fallback[fallback.length - 1];
    return {
      name: str(row.name, base?.name ?? `Rule ${index + 1}`),
      category: str(row.category, base?.category ?? 'Policy'),
      confidence: str(row.confidence, base?.confidence ?? '80%'),
      state: str(row.state, base?.state ?? 'Review')
    };
  });
}

export function adaptDevices(payload: unknown, fallback: DeviceRecord[]): DeviceRecord[] {
  const data = asRecord(payload);
  const connectedDevices = asArray(data.connected_devices);
  const nodes = asArray(data.nodes);
  const hasDeviceField = Array.isArray(data.connected_devices) || Array.isArray(data.nodes);
  const rows = connectedDevices.length ? connectedDevices : nodes;
  if (!rows.length) return hasDeviceField ? [] : fallback;

  return rows.map((row, index) => {
    const base = fallback[index] ?? fallback[fallback.length - 1];
    const name = str(row.name ?? row.label, base?.name ?? `Device ${index + 1}`);
    const status = str(row.status, base?.status ?? 'Unknown');
    return {
      name,
      ip: str(row.ip, base?.ip ?? 'n/a'),
      type: str(row.kind ?? row.type, base?.type ?? 'Runtime Node'),
      sampleRate: str(row.sample_rate, base?.sampleRate ?? 'n/a'),
      io: str(row.detail ?? row.io ?? row.channels, base?.io ?? 'Product Layer'),
      status: status.charAt(0).toUpperCase() + status.slice(1)
    };
  });
}

export function adaptTopologyNodes(payload: unknown): TopologyNode[] {
  return asArray(asRecord(payload).nodes).map((row, index) => ({
    id: str(row.id, `node-${index + 1}`),
    label: str(row.label, `Node ${index + 1}`),
    kind: str(row.kind, 'runtime'),
    status: str(row.status, 'unknown'),
    detail: str(row.detail, '')
  }));
}

export function adaptTopologyLinks(payload: unknown): TopologyLink[] {
  return asArray(asRecord(payload).links).map((row, index) => ({
    from: str(row.from, `node-${index}`),
    to: str(row.to, `node-${index + 1}`),
    kind: str(row.kind, 'link'),
    status: str(row.status, 'unknown')
  }));
}

export function adaptRoutingSegments(payload: unknown): RoutingSegment[] {
  return asArray(asRecord(payload).routing_scheme).map((row, index) => {
    const start = Math.trunc(num(row.start, index + 1));
    const end = Math.trunc(num(row.end, start));
    return {
      id: `${str(row.role, 'segment')}-${start}-${end}`,
      start,
      end,
      role: str(row.role, 'routing'),
      tapPoint: str(row.tap_point, ''),
      labelShort: str(row.label_short, `Ch ${start}-${end}`),
      labelFull: str(row.label_full, ''),
      wingRoutingHint: str(row.wing_routing_hint, ''),
      stereo: Boolean(row.stereo),
      required: Boolean(row.required),
      usedBy: Array.isArray(row.used_by) ? row.used_by.map(item => String(item)) : []
    };
  });
}

function toneForTrace(row: UnknownRecord): Tone {
  const status = str(row.status).toLowerCase();
  const blocked = str(row.blocked_reason);
  const proof = str(row.proof_status).toLowerCase();
  if (blocked || status.includes('blocked') || status.includes('error')) return 'red';
  if (proof === 'verified' || status === 'applied') return 'green';
  if (status.includes('created') || status.includes('pending')) return 'blue';
  return 'yellow';
}

export function adaptDecisionTraces(payload: unknown): DecisionTraceRecord[] {
  return asArray(asRecord(payload).traces).slice(-24).reverse().map((row, index) => ({
    decisionId: str(row.decision_id, `trace-${index + 1}`),
    proposalId: str(row.proposal_id) || undefined,
    source: str(row.source, 'automation_trace_store'),
    mode: str(row.mode) || undefined,
    target: str(row.target) || undefined,
    actionType: str(row.action_type, 'proposal'),
    status: str(row.status, 'observed'),
    delta: row.delta === null || row.delta === undefined ? undefined : String(row.delta),
    proofStatus: str(row.proof_status, 'unverified'),
    blockedReason: str(row.blocked_reason) || undefined,
    createdAt: timeLabel(row.created_at, 'now'),
    tone: toneForTrace(row)
  }));
}

export function adaptSafetyMonitor(payload: unknown, backend: BackendUiState): SafetyMonitorState {
  const data = asRecord(payload);
  const liveWritePolicy = asRecord(data.live_write_policy);
  const manual = asRecord(data.manual_write_supervision);
  const proof = asRecord(data.proof);
  const lastReadback = asRecord(proof.last_readback);
  const summary = asRecord(data.summary);
  const mode = asRecord(data.operator_mode);
  const blockedSurfaces = Array.isArray(data.blocked_surfaces)
    ? data.blocked_surfaces.map(item => String(item))
    : ['eq', 'compressor', 'routing', 'snapshots', 'autonomous_apply'];
  return {
    operatorMode: str(mode.mode, backend.liveWritePolicy === 'mock_frontend_only' ? 'mock' : 'unknown'),
    liveWritePolicy: str(liveWritePolicy.policy, backend.liveWritePolicy),
    applyPath: str(liveWritePolicy.apply_path, backend.applyPath),
    emergencyStopped: Boolean(manual.emergency_stopped),
    resetRequired: Boolean(manual.reset_required),
    ready: Boolean(summary.ready),
    mixerConnected: Boolean(summary.mixer_connected),
    approvedLiveWriteKinds: Array.isArray(liveWritePolicy.approved_live_write_kinds)
      ? liveWritePolicy.approved_live_write_kinds.map(item => String(item))
      : backend.writeScope,
    blockedSurfaces,
    lastProofStatus: str(proof.last_proof_status, 'none'),
    lastReadbackStatus: str(lastReadback.status, 'none'),
    traceCount: Math.trunc(num(summary.trace_count, 0)),
    blockedTraceCount: Math.trunc(num(summary.blocked_trace_count, 0)),
    sourceLabel: str(data.source, backend.sourceLabel)
  };
}

export function adaptReplayRuns(payload: unknown): ReplayRunRecord[] {
  return asArray(asRecord(payload).runs).map((row, index) => ({
    runId: str(row.run_id, `replay-${index + 1}`),
    status: str(row.status, 'unknown'),
    reason: str(row.reason, ''),
    generatedAt: timeLabel(row.generated_at, 'runtime'),
    reportPath: str(row.report_path, ''),
    tracePath: str(row.decision_traces_path, ''),
    metricsPath: str(row.metrics_path, ''),
    traceCount: Math.trunc(num(row.trace_count, 0)),
    offlineOnly: row.offline_only !== false,
    liveWrite: Boolean(row.live_write)
  }));
}

export function adaptChannelIntelligence(payload: unknown): ChannelIntelligenceRecord[] {
  return asArray(asRecord(payload).channels).map((row, index) => ({
    channel: Math.trunc(num(row.channel, index + 1)),
    name: str(row.name, `Ch ${index + 1}`),
    role: str(row.role, 'Channel'),
    group: str(row.group, 'CHANNELS'),
    risk: str(row.risk, 'low'),
    gainRideState: str(row.gain_ride_state, 'shadow_proposal_only'),
    recentTraceCount: Math.trunc(num(row.recent_trace_count, 0)),
    liveApplyAllowed: Boolean(row.live_apply_allowed)
  }));
}

export function appendBackendLog(
  logs: LogRecord[],
  eventType: string,
  message: string,
  level: LogRecord['level'] = 'INFO'
): LogRecord[] {
  const time = new Date().toTimeString().slice(0, 8);
  const next = [{ time, level, source: eventType, message }, ...logs];
  return next.slice(0, 40);
}
