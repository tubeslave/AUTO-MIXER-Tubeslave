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

const modeToBackend: Record<AutomixerMode, BackendOperatorMode> = {
  Analyze: 'analyze',
  Assist: 'assist',
  Supervised: 'supervised',
  Auto: 'auto'
};

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
  const normalized = String(value || '').trim().toLowerCase() as BackendOperatorMode;
  return modeToUi[normalized] ?? 'Assist';
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
  const modeGateRows = asArray(modeGate.rows).map(row => {
    const mode = str(row.mode, 'unknown_mode');
    const status = str(row.status, 'unknown');
    const blockers = stringList(row.blockers);
    const nextActions = stringList(row.next_required_actions);
    const writeKinds = stringList(row.current_live_write_kinds);
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
      gapText ? `Full Auto gaps: ${gapText}` : '',
    ].filter(Boolean).join(' / ') || 'Mode gate is read-only.';
    return {
      moduleId: `mode_gate:${mode}`,
      label: str(row.project_mode_name, `Mode gate: ${titleFromToken(mode)}`),
      domain: 'autonomy_mode_gate',
      status,
      entrypoint: mode,
      module: writeKinds.length ? `policy:${writeKinds.join(',')}` : 'no_current_write_authority',
      uiSurface: 'Autonomy mode transition gate',
      canWriteLive: false,
      liveWriteKinds: [],
      pipelineProfiles: [
        `runtime_entry:${Boolean(row.runtime_entry_allowed)}`,
        `proposal_creation:${Boolean(row.proposal_creation_allowed)}`,
        `live_apply:${Boolean(row.live_apply_allowed)}`,
        ...writeKinds.map(kind => `current_write:${kind}`),
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
    const freshnessText = available
      ? `age:${Math.round(ageSec)}s stale_after:${Math.round(staleAfterSec)}s stale:${stale}`
      : '';
    const proofQualityText = proofQualityAvailable
      ? `read_surface_gaps:${physicalGapSurfaceCount}; fixture_paths:${declaredFixturePathCount}; test_names:${declaredTestNameCount}; summary_recomputed:${summaryRecomputed}; declarations:${fixtureDeclarationValidation}; contract_gap:${contractGapConsistency}; gap_surfaces:${physicalGapSurfaceIds.join(',') || 'none'}; next_readback:${nextPhysicalReadbackSurfaceId || 'none'}; source_next_readback:${sourceNextPhysicalReadbackSurfaceId || 'none'}; next_readback_gate:${nextPhysicalReadbackTargetConsistent}; capture_plan:${readSurfaceCapturePlanStatus || 'missing'}; capture_steps:${readSurfaceCapturePlanStepCount}; capture_execution:${readSurfaceCapturePlanExecutionAllowed}; capture_target_gate:${readSurfaceCapturePlanTargetConsistency}; raw_contract:${rawMixerContractStatus || 'missing'}; raw_contract_gap:${rawMixerContractGapRemains}; raw_contract_live_authority:${rawMixerContractLiveAuthority}; raw_contract_target_gate:${rawMixerContractTargetConsistency}; raw_intake:${rawMixerIntakeStatus || 'missing'}; raw_intake_artifacts:${rawMixerIntakeArtifactCount}; raw_intake_valid:${rawMixerIntakeValidCount}; raw_intake_invalid:${rawMixerIntakeInvalidCount}; raw_intake_candidate:${rawMixerIntakeCandidate}; raw_intake_closure:${rawMixerIntakeClosureAllowed}; raw_intake_target_gate:${rawMixerIntakeTargetConsistency}; channel_contract:${channelBasicContractStatus || 'missing'}; channel_contract_gap:${channelBasicContractGapRemains}; channel_contract_live_authority:${channelBasicContractLiveAuthority}; channel_contract_target_gate:${channelBasicContractTargetConsistency}; channel_intake:${channelBasicIntakeStatus || 'missing'}; channel_intake_artifacts:${channelBasicIntakeArtifactCount}; channel_intake_valid:${channelBasicIntakeValidCount}; channel_intake_invalid:${channelBasicIntakeInvalidCount}; channel_intake_valid_channels:${channelBasicIntakeValidChannelCount}; channel_intake_candidate:${channelBasicIntakeCandidate}; channel_intake_gap:${channelBasicIntakeGapRemains}; channel_intake_closure:${channelBasicIntakeClosureAllowed}; channel_intake_live_authority:${channelBasicIntakeLiveAuthority}; channel_intake_target_gate:${channelBasicIntakeTargetConsistency}; processing_contract:${channelProcessingContractStatus || 'missing'}; processing_contract_gap:${channelProcessingContractGapRemains}; processing_contract_live_authority:${channelProcessingContractLiveAuthority}; processing_contract_target_gate:${channelProcessingContractTargetConsistency}; processing_intake:${channelProcessingIntakeStatus || 'missing'}; processing_intake_artifacts:${channelProcessingIntakeArtifactCount}; processing_intake_valid:${channelProcessingIntakeValidCount}; processing_intake_invalid:${channelProcessingIntakeInvalidCount}; processing_intake_valid_channels:${channelProcessingIntakeValidChannelCount}; processing_intake_candidate:${channelProcessingIntakeCandidate}; processing_intake_gap:${channelProcessingIntakeGapRemains}; processing_intake_closure:${channelProcessingIntakeClosureAllowed}; processing_intake_live_authority:${channelProcessingIntakeLiveAuthority}; processing_intake_target_gate:${channelProcessingIntakeTargetConsistency}; bus_contract:${busMainDcaContractStatus || 'missing'}; bus_contract_gap:${busMainDcaContractGapRemains}; bus_contract_live_authority:${busMainDcaContractLiveAuthority}; bus_contract_target_gate:${busMainDcaContractTargetConsistency}; matrix_contract:${matrixStateContractStatus || 'missing'}; matrix_contract_gap:${matrixStateContractGapRemains}; matrix_contract_live_authority:${matrixStateContractLiveAuthority}; matrix_contract_target_gate:${matrixStateContractTargetConsistency}; fx_contract:${fxSlotsInsertsContractStatus || 'missing'}; fx_contract_gap:${fxSlotsInsertsContractGapRemains}; fx_contract_live_authority:${fxSlotsInsertsContractLiveAuthority}; fx_contract_target_gate:${fxSlotsInsertsContractTargetConsistency}; routing_contract:${routingSendsContractStatus || 'missing'}; routing_contract_gap:${routingSendsContractGapRemains}; routing_contract_live_authority:${routingSendsContractLiveAuthority}; routing_contract_target_gate:${routingSendsContractTargetConsistency}; scenes_contract:${snapshotsScenesContractStatus || 'missing'}; scenes_contract_gap:${snapshotsScenesContractGapRemains}; scenes_contract_live_authority:${snapshotsScenesContractLiveAuthority}; scenes_contract_target_gate:${snapshotsScenesContractTargetConsistency}; presets_contract:${userPresetsContractStatus || 'missing'}; presets_contract_gap:${userPresetsContractGapRemains}; presets_contract_live_authority:${userPresetsContractLiveAuthority}; presets_contract_target_gate:${userPresetsContractTargetConsistency}; dlive_contract:${dliveStateContractStatus || 'missing'}; dlive_contract_gap:${dliveStateContractGapRemains}; dlive_contract_live_authority:${dliveStateContractLiveAuthority}; dlive_contract_target_gate:${dliveStateContractTargetConsistency}`
      : 'read_surface_quality:missing';
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
        `matrix_contract_available:${matrixStateContractAvailable}`,
        `matrix_contract_status:${matrixStateContractStatus}`,
        `matrix_contract_schema:${matrixStateContractSchemaId}`,
        `matrix_contract_gap:${matrixStateContractGapRemains}`,
        `matrix_contract_live_authority:${matrixStateContractLiveAuthority}`,
        `matrix_contract_target_gate:${matrixStateContractTargetConsistency}`,
        `fx_contract_available:${fxSlotsInsertsContractAvailable}`,
        `fx_contract_status:${fxSlotsInsertsContractStatus}`,
        `fx_contract_schema:${fxSlotsInsertsContractSchemaId}`,
        `fx_contract_gap:${fxSlotsInsertsContractGapRemains}`,
        `fx_contract_live_authority:${fxSlotsInsertsContractLiveAuthority}`,
        `fx_contract_target_gate:${fxSlotsInsertsContractTargetConsistency}`,
        `routing_contract_available:${routingSendsContractAvailable}`,
        `routing_contract_status:${routingSendsContractStatus}`,
        `routing_contract_schema:${routingSendsContractSchemaId}`,
        `routing_contract_gap:${routingSendsContractGapRemains}`,
        `routing_contract_live_authority:${routingSendsContractLiveAuthority}`,
        `routing_contract_target_gate:${routingSendsContractTargetConsistency}`,
        `scenes_contract_available:${snapshotsScenesContractAvailable}`,
        `scenes_contract_status:${snapshotsScenesContractStatus}`,
        `scenes_contract_schema:${snapshotsScenesContractSchemaId}`,
        `scenes_contract_gap:${snapshotsScenesContractGapRemains}`,
        `scenes_contract_live_authority:${snapshotsScenesContractLiveAuthority}`,
        `scenes_contract_target_gate:${snapshotsScenesContractTargetConsistency}`,
        `presets_contract_available:${userPresetsContractAvailable}`,
        `presets_contract_status:${userPresetsContractStatus}`,
        `presets_contract_schema:${userPresetsContractSchemaId}`,
        `presets_contract_gap:${userPresetsContractGapRemains}`,
        `presets_contract_live_authority:${userPresetsContractLiveAuthority}`,
        `presets_contract_target_gate:${userPresetsContractTargetConsistency}`,
        `dlive_contract_available:${dliveStateContractAvailable}`,
        `dlive_contract_status:${dliveStateContractStatus}`,
        `dlive_contract_schema:${dliveStateContractSchemaId}`,
        `dlive_contract_gap:${dliveStateContractGapRemains}`,
        `dlive_contract_live_authority:${dliveStateContractLiveAuthority}`,
        `dlive_contract_target_gate:${dliveStateContractTargetConsistency}`,
      ],
      blockedReason: available
        ? stale
          ? 'Proof report artifact is stale; rerun proof checks before promotion review.'
          : 'Read-only artifact source; safety is normalized before cockpit display.'
        : 'No supervised proof report artifact found.',
      nextStep: available
        ? `Latest report status: ${str(proofSummary.latest_report_status, 'unknown')}; overlay: ${overlayStatus}; artifacts: ${artifactCount}; invalid: ${invalidCount}; ${freshnessText}; ${proofQualityText}.`
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
    ...modeGateRows,
    ...readinessRows,
    ...proofEvidenceRows,
    ...readSurfaceRemediationRows,
    ...overlayRows,
    ...gapRows
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
