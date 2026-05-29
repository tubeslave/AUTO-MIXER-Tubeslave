import type {
  Agent,
  AutomixerMode,
  BackendUiState,
  BusRecord,
  Channel,
  Communication,
  DashboardRuntime,
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

function metricNum(value: unknown, fallback: number): number {
  if (value === null || value === undefined || value === '') return fallback;
  return num(value, fallback);
}

function str(value: unknown, fallback = ''): string {
  if (value === null || value === undefined) return fallback;
  const next = String(value).trim();
  return next || fallback;
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
  if (normalized === 'canonical' && canWriteLive) return 'green';
  if (normalized === 'canonical') return 'blue';
  if (normalized === 'proposal_only' || normalized === 'shadow_only' || normalized === 'offline_only') return 'yellow';
  if (normalized === 'lab') return 'purple';
  if (normalized === 'placeholder' || normalized === 'quarantined') return 'red';
  if (normalized === 'deprecated') return 'muted';
  return canWriteLive ? 'yellow' : 'blue';
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
    const confidence = num(recognition.confidence, base?.confidence ?? 72);
    return {
      id,
      name: str(row.name, base?.name ?? `Ch ${id}`),
      type: str(row.role, base?.type ?? 'Channel'),
      source: str(row.source, base?.source ?? `In ${String(id).padStart(2, '0')}`),
      level: num(row.level_db ?? meter.rms_db, base?.level ?? -60),
      lufs: num(row.lufs ?? meter.lufs, base?.lufs ?? -60),
      dynamic: num(row.dynamic_range_db ?? meter.dynamic_range_db, base?.dynamic ?? 0),
      status: status === 'active' ? 'ACTIVE' : status === 'muted' ? 'MUTED' : 'INACTIVE',
      recognition: `${str(recognition.label, str(row.role, base?.type ?? 'Channel'))}${confidence ? ` (${Math.round(confidence)}%)` : ''}`,
      group: str(row.group, base?.group ?? 'CHANNELS').toUpperCase(),
      fader: num(control.fader_db, base?.fader ?? -50),
      confidence
    };
  });
}

export function adaptModuleStatusRecords(payload: unknown): ModuleStatusRecord[] {
  const data = asRecord(payload);
  const rows = asArray(data.module_status_map);
  if (!rows.length) return [];

  return rows.map(row => {
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
  const rows = asArray(asRecord(payload).buses);
  if (!rows.length) return fallback;

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
    sourceLabel: 'Backend'
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
      readyLabel: 'Mock ready',
      readyTone: 'blue',
      audioLabel: 'Mock meters',
      audioTone: 'blue',
      proposalQueueLabel: 'Mock queue',
      proposalQueueTone: 'blue',
      liveWritePolicy: 'mock_frontend_only',
      manualGateLabel: 'UI only',
      manualGateTone: 'blue',
      resetRequired: false,
      hardwareProofItems: [
        { label: 'Fader proof', value: 'Mock only', tone: 'blue' },
        { label: 'Readback', value: 'Mock only', tone: 'blue' },
        { label: 'Rollback', value: 'Mock only', tone: 'blue' },
        { label: 'Final readiness', value: 'Prototype', tone: 'blue' }
      ],
      blockedReasons: [],
      statusItems: [
        { label: 'Mixer connected', value: 'Mock', tone: 'blue' },
        { label: 'Gain staging', value: 'Mock', tone: 'blue' },
        { label: 'Auto fader', value: 'Mock', tone: 'blue' },
        { label: 'No clipping', value: 'Assumed', tone: 'green' }
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
  const rows = asArray(roster.agents);
  if (!rows.length) return fallback;

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
  const rows = asArray(asRecord(payload).communications);
  if (!rows.length) return fallback;

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
  const rows = asArray(asRecord(payload).memories);
  if (!rows.length) return fallback;

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
  const rows = asArray(asRecord(payload).sessions);
  if (!rows.length) return fallback;

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
  const rows = asArray(asRecord(payload).rules);
  if (!rows.length) return fallback;

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
  const rows = connectedDevices.length ? connectedDevices : nodes;
  if (!rows.length) return fallback;

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
