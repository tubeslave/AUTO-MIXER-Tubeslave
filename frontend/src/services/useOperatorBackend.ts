import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import websocketService from './websocket';
import {
  adaptAgents,
  adaptBackendState,
  adaptBusesFromChannels,
  adaptBusesSnapshot,
  adaptChannelIntelligence,
  adaptChannels,
  adaptCommunicationsSnapshot,
  adaptDashboardRuntime,
  adaptDecisionTraces,
  adaptDevices,
  adaptDirectorRuntime,
  adaptOverviewMetrics,
  adaptPerformanceRecords,
  adaptProposalOnlyContractTasks,
  adaptReplayRuns,
  adaptMemorySnapshot,
  adaptModuleStatusRecords,
  adaptRulesSnapshot,
  adaptRuntimeCommunications,
  adaptRuntimeMemories,
  adaptRoutingSegments,
  adaptSafetyMonitor,
  adaptTelemetryAlertsSnapshot,
  adaptTrainingSnapshot,
  adaptTasks,
  adaptTopologyLinks,
  adaptTopologyNodes,
  appendBackendLog,
  mockBackendState,
  toBackendMode,
  toUiMode
} from './operatorAdapters';
import {
  classifyApplyToneValue,
  classifyReadbackToneValue,
  classifyRollbackToneValue
} from './operatorStatusTone.mjs';
import {
  agents as mockAgents,
  buses as mockBuses,
  channels as mockChannels,
  communications as mockCommunications,
  devices as mockDevices,
  logs as mockLogs,
  memories as mockMemories,
  overviewMetrics as mockOverviewMetrics,
  performance as mockPerformance,
  tasks as mockTasks
} from '../mock/automixerData';
import type {
  Agent,
  AutomixerMode,
  BackendConnectionState,
  BackendDataState,
  BackendUiState,
  BusRecord,
  Channel,
  Communication,
  DashboardRuntime,
  DecisionTraceRecord,
  DeviceRecord,
  DirectorRuntime,
  EmergencyStopUiState,
  ChannelIntelligenceRecord,
  LogRecord,
  MemoryRecord,
  Metric,
  ModuleStatusRecord,
  PerformanceRecord,
  ProposalApplyResult,
  ReplayRunRecord,
  RuleLibrarySnapshotRow,
  RoutingSegment,
  SafetyMonitorState,
  SoundcheckRuntime,
  Task,
  TelemetryAlertsSnapshot,
  Tone,
  TrainingSessionSnapshotRow,
  TopologyLink,
  TopologyNode
} from '../types';

const idleSoundcheckRuntime: SoundcheckRuntime = {
  running: false,
  state: 'idle',
  message: 'Soundcheck is idle.',
  error: '',
  observeOnly: true,
  proposalOnly: true,
  mixerConnected: false,
  audioRunning: false,
  selectedChannels: [],
  channelsWithSignal: 0,
  totalChannels: 0,
  recommendationCount: 0,
  transactionPlanCount: 0,
  transactionOperationCount: 0,
  liveAllowedFamilies: [],
  familyLifecycle: {},
  familyLiveWriteKinds: {},
  familyWriteKinds: {},
  familyContracts: {},
  derivationComplete: false,
  readiness: {
    schemaVersion: 'soundcheck_readiness/v1',
    runId: 'idle',
    planRevision: 1,
    planFingerprint: '',
    mixReadiness: 'soundcheck_incomplete',
    safetyState: 'clear',
    executionStatus: 'awaiting_operator_review',
    logicalTotal: 0,
    logicalApplied: 0,
    logicalAlreadySatisfied: 0,
    logicalReplaced: 0,
    logicalOperatorReview: 0,
    logicalForbidden: 0,
    logicalPending: 0,
    unresolvedRequiredIds: [],
    unresolvedReasons: {},
    globalBlockers: [],
    evidenceComplete: false,
    stages: {}
  },
  replan: {
    status: 'not_required',
    draftAvailable: false,
    draftId: '',
    runId: '',
    fromPlanRevision: 0,
    toPlanRevision: 0,
    splitOperationCount: 0,
    safeAlternativeOperationCount: 0,
    deferredTargetCount: 0,
    operatorRequests: [],
    safeAlternatives: []
  },
  interceptedWriteCount: 0,
  lastUpdated: ''
};

const idleEmergencyStop: EmergencyStopUiState = {
  status: 'ready',
  message: 'Emergency stop ready. Arm, then confirm immediately.',
  updatedAt: '',
  reconciliationRequired: false
};

const EMERGENCY_STOP_ACK_TIMEOUT_MS = 5000;

function soundcheckRuntimeFromPayload(
  current: SoundcheckRuntime,
  payload: unknown
): SoundcheckRuntime {
  const data = recordValue(payload);
  const selected = Array.isArray(data.selected_channels)
    ? data.selected_channels.map(Number).filter(Number.isFinite)
    : current.selectedChannels;
  const summary = recordValue(data.summary);
  const readinessPayload = recordValue(data.soundcheck_readiness);
  const readinessCounts = recordValue(readinessPayload.logical_counts);
  const readinessStages = Object.fromEntries(
    Object.entries(recordValue(readinessPayload.stages)).map(([stageId, rawStage]) => {
      const stage = recordValue(rawStage);
      return [stageId, {
        required: numberValue(stage.required),
        resolved: numberValue(stage.resolved),
        applied: numberValue(stage.applied),
        alreadySatisfied: numberValue(stage.already_satisfied),
        replaced: numberValue(stage.replaced),
        operatorReview: numberValue(stage.operator_review),
        forbidden: numberValue(stage.forbidden),
        pending: numberValue(stage.pending),
        ready: Boolean(stage.ready)
      }];
    })
  );
  const hasReadiness = Object.prototype.hasOwnProperty.call(data, 'soundcheck_readiness');
  const lifecycleWithoutReadiness = !hasReadiness && [
    'running',
    'is_running',
    'state',
    'current_step'
  ].some(key => Object.prototype.hasOwnProperty.call(data, key));
  const mixReadinessValue = stringValue(readinessPayload.mix_readiness ?? readinessPayload.status);
  const mixReadiness = (
    ['soundcheck_incomplete', 'safe_core_ready', 'show_mix_ready'].includes(mixReadinessValue)
      ? mixReadinessValue
      : 'soundcheck_incomplete'
  ) as SoundcheckRuntime['readiness']['mixReadiness'];
  const readiness = hasReadiness ? {
    schemaVersion: stringValue(readinessPayload.schema_version, 'soundcheck_readiness/v1'),
    runId: stringValue(readinessPayload.run_id, 'unknown'),
    planRevision: numberValue(readinessPayload.plan_revision, 1),
    planFingerprint: stringValue(readinessPayload.plan_fingerprint),
    mixReadiness,
    safetyState: stringValue(readinessPayload.safety_state, 'manual_review_required'),
    executionStatus: stringValue(readinessPayload.execution_status, 'awaiting_operator_review'),
    logicalTotal: numberValue(readinessCounts.total),
    logicalApplied: numberValue(readinessCounts.applied),
    logicalAlreadySatisfied: numberValue(readinessCounts.already_satisfied),
    logicalReplaced: numberValue(readinessCounts.replaced),
    logicalOperatorReview: numberValue(readinessCounts.operator_review),
    logicalForbidden: numberValue(readinessCounts.forbidden),
    logicalPending: numberValue(readinessCounts.pending),
    unresolvedRequiredIds: Array.isArray(readinessPayload.unresolved_required_operation_ids)
      ? readinessPayload.unresolved_required_operation_ids.map(String)
      : [],
    unresolvedReasons: Object.fromEntries(
      Object.entries(recordValue(readinessPayload.unresolved_reasons)).map(
        ([operationId, reason]) => [operationId, String(reason)]
      )
    ),
    globalBlockers: Array.isArray(readinessPayload.global_blockers)
      ? readinessPayload.global_blockers.map(String)
      : [],
    evidenceComplete: Boolean(readinessPayload.readiness_evidence_complete),
    stages: readinessStages
  } : lifecycleWithoutReadiness ? {
    ...current.readiness,
    mixReadiness: 'soundcheck_incomplete' as const,
    safetyState: 'manual_review_required',
    executionStatus: 'readiness_contract_missing_from_lifecycle_event',
    evidenceComplete: false
  } : current.readiness;
  const hasReplan = Object.prototype.hasOwnProperty.call(data, 'soundcheck_replan');
  const replanPayload = recordValue(data.soundcheck_replan);
  const replan = hasReplan ? {
    status: stringValue(replanPayload.status, 'not_required'),
    draftAvailable: Boolean(replanPayload.draft_available),
    draftId: stringValue(replanPayload.draft_id),
    runId: stringValue(replanPayload.run_id),
    fromPlanRevision: numberValue(replanPayload.from_plan_revision),
    toPlanRevision: numberValue(replanPayload.to_plan_revision),
    splitOperationCount: numberValue(replanPayload.split_operation_count),
    safeAlternativeOperationCount: numberValue(
      replanPayload.safe_alternative_operation_count
    ),
    deferredTargetCount: numberValue(replanPayload.deferred_target_count),
    operatorRequests: Array.isArray(replanPayload.operator_requests)
      ? replanPayload.operator_requests.map(raw => {
        const request = recordValue(raw);
        return {
          operationId: stringValue(request.operation_id),
          reason: stringValue(request.reason),
          safeAutomaticReplacementAvailable: Boolean(
            request.safe_automatic_replacement_available
          ),
          selectedSafeStrategy: stringValue(request.selected_safe_strategy)
        };
      })
      : [],
    safeAlternatives: Array.isArray(replanPayload.safe_alternatives)
      ? replanPayload.safe_alternatives.map(raw => {
        const alternative = recordValue(raw);
        return {
          strategy: stringValue(alternative.strategy),
          translationDb: Number.isFinite(Number(alternative.translation_db))
            ? Number(alternative.translation_db)
            : undefined,
          relativeBalancePreserved: Boolean(
            alternative.relative_balance_preserved
          ),
          finalTargetVectorPreserved: Boolean(
            alternative.final_target_vector_preserved
          ),
          currentStepTransitional: Boolean(
            alternative.current_step_transitional
          ),
          constructiveLaneUsed: Boolean(alternative.constructive_lane_used)
        };
      })
      : []
  } : current.replan;
  const hasFamilyRegistry = Object.prototype.hasOwnProperty.call(
    data,
    'supervised_console_family_registry'
  );
  const familyRegistry = recordValue(data.supervised_console_family_registry);
  const familyContracts = Object.fromEntries(
    Object.entries(familyRegistry).map(([familyId, rawContract]) => {
      const contract = recordValue(rawContract);
      const writeKinds = Array.isArray(contract.write_kinds)
        ? contract.write_kinds.map(String)
        : [];
      const liveAllowedWriteKinds = Array.isArray(contract.live_allowed_write_kinds)
        ? contract.live_allowed_write_kinds.map(String)
        : [];
      return [familyId, {
        label: stringValue(contract.label, familyId.replace(/_/g, ' ')),
        lifecycle: stringValue(contract.lifecycle, 'unknown'),
        riskClass: stringValue(contract.risk_class, 'unknown'),
        blockers: Array.isArray(contract.blockers) ? contract.blockers.map(String) : [],
        separateApproval: Boolean(contract.separate_approval),
        hardwareProof: stringValue(contract.hardware_proof, 'missing'),
        proofRunAllowed: Boolean(contract.proof_run_allowed),
        writeKinds,
        liveAllowedWriteKinds
      }];
    })
  );
  const familyLifecycle = Object.fromEntries(
    Object.entries(familyRegistry).map(([familyId, rawContract]) => {
      const contract = recordValue(rawContract);
      return [familyId, stringValue(contract.lifecycle, 'unknown')];
    })
  );
  const familyLiveWriteKinds = Object.fromEntries(
    Object.entries(familyRegistry).map(([familyId, rawContract]) => {
      const values = recordValue(rawContract).live_allowed_write_kinds;
      return [familyId, Array.isArray(values) ? values.map(String) : []];
    })
  );
  const familyWriteKinds = Object.fromEntries(
    Object.entries(familyRegistry).map(([familyId, rawContract]) => {
      const values = recordValue(rawContract).write_kinds;
      return [familyId, Array.isArray(values) ? values.map(String) : []];
    })
  );
  const hasLiveAllowedFamilies = Object.prototype.hasOwnProperty.call(
    data,
    'supervised_console_live_allowed_families'
  );
  const liveAllowedFamilies = Array.isArray(data.supervised_console_live_allowed_families)
    ? data.supervised_console_live_allowed_families
      .map(value => String(value).trim())
      .filter(Boolean)
    : hasLiveAllowedFamilies
      ? []
      : hasFamilyRegistry
        ? Object.entries(familyContracts)
          .filter(([, contract]) => contract.lifecycle === 'live_allowed' && contract.liveAllowedWriteKinds.length > 0)
          .map(([familyId]) => familyId)
        : current.liveAllowedFamilies;
  return {
    ...current,
    running: Boolean(data.running ?? data.is_running ?? current.running),
    state: stringValue(data.current_step ?? data.state, current.state),
    message: stringValue(data.message, current.message),
    error: stringValue(data.error),
    observeOnly: data.observe_only !== false,
    proposalOnly: data.proposal_only !== false,
    mixerConnected: Boolean(data.mixer_connected ?? current.mixerConnected),
    audioRunning: Boolean(data.audio_running ?? current.audioRunning),
    selectedChannels: selected,
    channelsWithSignal: numberValue(data.channels_with_signal, current.channelsWithSignal),
    totalChannels: numberValue(data.total_channels, selected.length || current.totalChannels),
    recommendationCount: numberValue(
      data.soundcheck_recommendation_count ?? data.recommendation_count,
      current.recommendationCount
    ),
    transactionPlanCount: numberValue(
      data.supervised_transaction_plan_count,
      current.transactionPlanCount
    ),
    transactionOperationCount: numberValue(
      data.supervised_transaction_operation_count,
      current.transactionOperationCount
    ),
    liveAllowedFamilies,
    familyLifecycle: hasFamilyRegistry ? familyLifecycle : current.familyLifecycle,
    familyLiveWriteKinds: hasFamilyRegistry
      ? familyLiveWriteKinds
      : current.familyLiveWriteKinds,
    familyWriteKinds: hasFamilyRegistry ? familyWriteKinds : current.familyWriteKinds,
    familyContracts: hasFamilyRegistry ? familyContracts : current.familyContracts,
    derivationComplete: Boolean(
      data.recommendation_derivation_complete ?? current.derivationComplete
    ),
    readiness,
    replan,
    interceptedWriteCount: numberValue(
      summary.total_operations ?? data.intercepted_write_count,
      current.interceptedWriteCount
    ),
    lastUpdated: new Date().toLocaleTimeString()
  };
}

type ProductLayerPayloads = {
  dashboard?: unknown;
  channels?: unknown;
  decisions?: unknown;
  decisionTraces?: unknown;
  safetyMonitor?: unknown;
  replayRuns?: unknown;
  channelIntelligence?: unknown;
  agents?: unknown;
  topology?: unknown;
  mode?: unknown;
  groups?: unknown;
  memory?: unknown;
  training?: unknown;
  rules?: unknown;
  communications?: unknown;
  processing?: unknown;
  pipelinePreview?: unknown;
  telemetryAlerts?: unknown;
};

export interface OperatorViewData {
  channels: Channel[];
  buses: BusRecord[];
  agents: Agent[];
  tasks: Task[];
  decisionTraces: DecisionTraceRecord[];
  safetyMonitor: SafetyMonitorState;
  replayRuns: ReplayRunRecord[];
  channelIntelligence: ChannelIntelligenceRecord[];
  overviewMetrics: Metric[];
  devices: DeviceRecord[];
  logs: LogRecord[];
  moduleStatuses: ModuleStatusRecord[];
  telemetryAlerts: TelemetryAlertsSnapshot;
  communications: Communication[];
  memories: MemoryRecord[];
  trainingSessions: TrainingSessionSnapshotRow[];
  rules: RuleLibrarySnapshotRow[];
  topologyNodes: TopologyNode[];
  topologyLinks: TopologyLink[];
  routingSegments: RoutingSegment[];
  dashboardRuntime: DashboardRuntime;
  directorRuntime: DirectorRuntime;
  performanceRecords: PerformanceRecord[];
  readModelSources: {
    groups: string;
    memory: string;
    performance: string;
    training: string;
    rules: string;
    telemetryAlerts: string;
    communications: string;
  };
}

const AUTOCONNECT_STORAGE_KEY = 'automixer_backend_autoconnect';
const EXPECTED_RUNTIME_PAYLOADS: (keyof ProductLayerPayloads)[] = [
  'dashboard',
  'channels',
  'decisions',
  'decisionTraces',
  'safetyMonitor',
  'replayRuns',
  'channelIntelligence',
  'agents',
  'topology',
  'mode',
  'groups',
  'memory',
  'training',
  'rules',
  'communications',
  'processing',
  'pipelinePreview',
  'telemetryAlerts'
];

function shouldAutoConnect() {
  try {
    const preference = globalThis.localStorage?.getItem(AUTOCONNECT_STORAGE_KEY);
    if (preference === 'off') return false;
    return true;
  } catch {
    return true;
  }
}

function connectionMessage(state: BackendConnectionState) {
  if (state === 'connected') return 'Product Layer connected.';
  if (state === 'connecting') return 'Connecting to Product Layer.';
  if (state === 'error') return 'Backend connection failed. UI stays on mock fallback.';
  return mockBackendState.message;
}

function socketIsOpen() {
  return (websocketService as { isReady?: () => boolean }).isReady?.() === true;
}

function payloadKeys(payloads: ProductLayerPayloads) {
  return EXPECTED_RUNTIME_PAYLOADS.filter(key => Boolean(payloads[key]));
}

function payloadSource(payload: unknown, fallback: string) {
  if (!payload || typeof payload !== 'object') return fallback;
  const data = payload as Record<string, unknown>;
  const type = data.type ? String(data.type) : '';
  const source = data.source ? String(data.source) : '';
  return [type, source].filter(Boolean).join(' / ') || fallback;
}

function dataStateLabel(state: BackendDataState) {
  if (state === 'connected') return 'Product Layer live';
  if (state === 'partial') return 'Product Layer partial';
  if (state === 'stale') return 'Stale snapshot';
  if (state === 'disconnected') return 'WING disconnected';
  if (state === 'error') return 'Backend unavailable';
  if (state === 'connecting') return 'Connecting';
  return 'Mock fallback';
}

function dataStateTone(state: BackendDataState) {
  if (state === 'connected') return 'green' as const;
  if (state === 'partial' || state === 'stale' || state === 'connecting') return 'yellow' as const;
  if (state === 'error' || state === 'disconnected') return 'red' as const;
  return 'blue' as const;
}

function resolveDataState(connectionState: BackendConnectionState, payloads: ProductLayerPayloads): BackendDataState {
  const keys = payloadKeys(payloads);
  const hasRuntimeData = keys.length > 0;
  const hasAllRuntimeData = keys.length >= EXPECTED_RUNTIME_PAYLOADS.length;

  if (connectionState === 'connecting') return 'connecting';
  if (connectionState === 'error') return 'error';
  if (connectionState === 'offline') return hasRuntimeData ? 'stale' : 'disconnected';
  if (connectionState === 'mock') return 'mock_fallback';
  if (!hasRuntimeData) return 'partial';
  if (!hasAllRuntimeData) return 'partial';
  return 'connected';
}

function proposalIdFromPayload(payload: unknown): string | undefined {
  if (!payload || typeof payload !== 'object') return undefined;
  const data = payload as Record<string, unknown>;
  const proposal = data.proposal && typeof data.proposal === 'object'
    ? data.proposal as Record<string, unknown>
    : {};
  return String(data.proposal_id ?? proposal.id ?? data.id ?? '').trim() || undefined;
}

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? value as Record<string, unknown> : {};
}

function stringValue(value: unknown, fallback = ''): string {
  const next = String(value ?? '').trim();
  return next || fallback;
}

function numberValue(value: unknown, fallback = 0): number {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function replayRowsFromPayload(payload: unknown): Record<string, unknown>[] {
  const data = recordValue(payload);
  const rows = Array.isArray(data.runs) ? data.runs : [];
  return rows.filter((row): row is Record<string, unknown> => Boolean(row && typeof row === 'object'));
}

function mergeReplayRunsPayload(current: unknown, incoming: unknown): unknown {
  const currentRows = replayRowsFromPayload(current);
  const incomingRows = replayRowsFromPayload(incoming);
  if (!currentRows.length || !incomingRows.length) return incoming;
  const incomingIds = new Set(incomingRows.map(row => stringValue(row.run_id)));
  const localRows = currentRows.filter(row => row.local_echo === true && !incomingIds.has(stringValue(row.run_id)));
  if (!localRows.length) return incoming;
  return {
    ...recordValue(incoming),
    runs: [...localRows, ...incomingRows],
    summary: {
      ...recordValue(recordValue(incoming).summary),
      run_count: localRows.length + incomingRows.length,
      offline_only: true,
      live_write: false
    }
  };
}

function replayRunsPayloadFromStarted(current: unknown, payload: unknown): unknown {
  const data = recordValue(payload);
  const artifacts = recordValue(data.artifacts);
  const result = recordValue(data.result);
  const summary = recordValue(result.summary);
  const runId = stringValue(data.run_id || artifacts.run_id, `frontend-replay-${Date.now()}`);
  const row = {
    run_id: runId,
    status: stringValue(data.status || result.status, 'requested'),
    reason: stringValue(data.reason || result.reason || result.write_policy, 'offline_replay_requested'),
    generated_at: stringValue(data.generated_at || result.generated_at, new Date().toISOString()),
    report_path: stringValue(artifacts.report_path),
    decision_traces_path: stringValue(artifacts.decision_traces_path),
    metrics_path: stringValue(artifacts.metrics_path),
    trace_count: numberValue(data.trace_count ?? result.trace_count ?? summary.trace_count),
    offline_only: data.offline_only !== false,
    live_write: Boolean(data.live_write),
    local_echo: true
  };
  const currentRows = replayRowsFromPayload(current).filter(existing => stringValue(existing.run_id) !== runId);
  return {
    type: 'replay_runs',
    status: 'ok',
    source: 'replay_run_started_local_echo',
    generated_at: row.generated_at,
    summary: {
      run_count: currentRows.length + 1,
      offline_only: row.offline_only,
      live_write: row.live_write
    },
    runs: [row, ...currentRows]
  };
}

export function classifyReadbackTone(status: unknown): Tone {
  return classifyReadbackToneValue(status) as Tone;
}

export function classifyRollbackTone(status: unknown): Tone {
  return classifyRollbackToneValue(status) as Tone;
}

function activeSoundcheckChannels(payload: unknown): number[] {
  const rows = recordValue(payload).channels;
  if (!Array.isArray(rows)) return [];
  return [...new Set(rows.flatMap(row => {
    const channel = recordValue(row);
    const id = Number(channel.id ?? channel.number ?? channel.channel);
    const status = stringValue(channel.status).toUpperCase();
    // Canonical WING capture map: USB 1..16 are raw console sources;
    // 17..48 are post-fader/stem/Main measurement taps and must never become
    // physical console channel correction targets.
    return Number.isInteger(id) && id >= 1 && id <= 16 && status === 'ACTIVE'
      ? [id]
      : [];
  }))].sort((a, b) => a - b);
}

function applyResultFromPayload(payload: unknown, fallbackStatus = 'updated'): ProposalApplyResult {
  const data = payload && typeof payload === 'object' ? payload as Record<string, unknown> : {};
  const rawResult = data.apply_result && typeof data.apply_result === 'object'
    ? data.apply_result as Record<string, unknown>
    : data;
  const status = String(data.status ?? rawResult.status ?? fallbackStatus);
  const reason = String(rawResult.reason ?? data.reason ?? '');
  const readbackStatus = String(rawResult.readback_status ?? 'not verified');
  const rollbackAvailable = Boolean(rawResult.rollback_available);
  const rollbackResults = Array.isArray(rawResult.rollback_results) ? rawResult.rollback_results : [];
  const rollbackFailed = rollbackResults.some(item => {
    const row = item && typeof item === 'object' ? item as Record<string, unknown> : {};
    return row.status !== 'applied' || row.rollback_verified === false;
  });
  const explicitRollbackStatus = String(
    rawResult.rollback_status ?? rawResult.rollback_reason ?? ''
  ).trim();
  const rollbackStatus = explicitRollbackStatus || (rollbackFailed
    ? 'rollback failed/unavailable'
    : rollbackResults.length
      ? 'rollback executed'
      : rollbackAvailable
        ? 'rollback available'
        : 'rollback unavailable');
  const tone = classifyApplyToneValue(status, reason, rollbackFailed) as Tone;
  const blocked = tone === 'red';
  return {
    proposalId: proposalIdFromPayload(payload),
    status,
    reason,
    applyRequest: blocked ? 'apply blocked' : status === 'applied' ? 'apply requested' : status,
    readbackStatus,
    rollbackStatus,
    rollbackAvailable,
    readbackTone: classifyReadbackTone(readbackStatus),
    rollbackTone: classifyRollbackTone(rollbackStatus),
    tone,
    updatedAt: new Date().toLocaleTimeString()
  };
}


function mergeMeterFrameIntoPayloads(current: ProductLayerPayloads, frame: Record<string, unknown>): ProductLayerPayloads {
  const frameChannels = Array.isArray(frame.channels) ? frame.channels as Array<Record<string, unknown>> : [];
  const meterByChannel = new Map<number, Record<string, unknown>>();
  frameChannels.forEach(row => {
    if (!row || typeof row !== 'object') return;
    const id = Number((row as Record<string, unknown>).channel);
    if (Number.isFinite(id)) meterByChannel.set(id, row as Record<string, unknown>);
  });

  const next: ProductLayerPayloads = { ...current };

  const inventory = current.channels && typeof current.channels === 'object'
    ? current.channels as Record<string, unknown>
    : null;
  if (inventory && Array.isArray(inventory.channels)) {
    next.channels = {
      ...inventory,
      channels: (inventory.channels as unknown[]).map(row => {
        if (!row || typeof row !== 'object') return row;
        const record = row as Record<string, unknown>;
        const meter = meterByChannel.get(Number(record.id ?? record.number));
        const previousMeter = record.meter && typeof record.meter === 'object'
          ? record.meter as Record<string, unknown>
          : {};
        if (!meter) {
          return {
            ...record,
            level_db: null,
            meter: {
              ...previousMeter,
              available: false,
              rms_db: null,
              peak_db: null,
              lufs: null
            }
          };
        }
        return {
          ...record,
          level_db: meter.rms_db ?? record.level_db,
          meter: {
            ...previousMeter,
            available: true,
            rms_db: meter.rms_db,
            peak_db: meter.peak_db,
            lufs: meter.lufs ?? previousMeter.lufs
          }
        };
      })
    };
  }

  const dashboard = current.dashboard && typeof current.dashboard === 'object'
    ? current.dashboard as Record<string, unknown>
    : null;
  if (dashboard) {
    next.dashboard = {
      ...dashboard,
      ...(frame.master_bus && typeof frame.master_bus === 'object' ? { master_bus: frame.master_bus } : {}),
      ...(frame.analyzer && typeof frame.analyzer === 'object' ? { analyzer: frame.analyzer } : {})
    };
  }
  return next;
}

function mergeAudioCaptureStatusIntoPayloads(
  current: ProductLayerPayloads,
  payload: Record<string, unknown>
): ProductLayerPayloads {
  const dashboard = recordValue(current.dashboard);
  return {
    ...current,
    dashboard: {
      ...dashboard,
      type: stringValue(dashboard.type, 'dashboard_snapshot'),
      generated_at: new Date().toISOString(),
      audio_capture: {
        ...payload,
        _frontend_authoritative_status: true
      }
    }
  };
}

function mergeDashboardSnapshotIntoPayloads(
  current: ProductLayerPayloads,
  payload: unknown
): ProductLayerPayloads {
  const incoming = recordValue(payload);
  const currentDashboard = recordValue(current.dashboard);
  const currentAudioCapture = recordValue(currentDashboard.audio_capture);
  return {
    ...current,
    dashboard: {
      ...incoming,
      ...(currentAudioCapture._frontend_authoritative_status === true
        ? { audio_capture: currentAudioCapture }
        : {})
    }
  };
}

function mergeConnectionStatusIntoPayloads(
  current: ProductLayerPayloads,
  payload: Record<string, unknown>
): ProductLayerPayloads {
  const dashboard = recordValue(current.dashboard);
  const connection = recordValue(dashboard.connection);
  return {
    ...current,
    dashboard: {
      ...dashboard,
      type: stringValue(dashboard.type, 'dashboard_snapshot'),
      generated_at: new Date().toISOString(),
      connection: {
        ...connection,
        connected: Boolean(payload.connected),
        connection_mode: payload.mode ?? connection.connection_mode,
        ip: payload.ip ?? connection.ip,
        state: payload.state ?? connection.state
      }
    }
  };
}

function mixerUpdateValue(values: unknown): unknown {
  if (!Array.isArray(values) || values.length === 0) return undefined;
  const semantic = values[values.length - 1];
  return semantic ?? values[0];
}

function mergeMixerUpdateIntoPayloads(
  current: ProductLayerPayloads,
  payload: Record<string, unknown>
): ProductLayerPayloads {
  const address = stringValue(payload.address);
  const channelMatch = address.match(/^\/ch\/(\d+)\/(fdr|mute|pan|\$name|name)$/);
  const value = mixerUpdateValue(payload.values);
  const dashboard = recordValue(current.dashboard);
  const dashboardConnection = recordValue(dashboard.connection);
  const dashboardState = recordValue(dashboardConnection.state);
  const nextDashboard = {
    ...dashboard,
    generated_at: new Date().toISOString(),
    connection: {
      ...dashboardConnection,
      state: address ? { ...dashboardState, [address]: value } : dashboardState
    }
  };

  if (!channelMatch) {
    return { ...current, dashboard: nextDashboard };
  }

  const channelId = Number(channelMatch[1]);
  const field = channelMatch[2];
  const inventory = recordValue(current.channels);
  const rows = Array.isArray(inventory.channels) ? inventory.channels : [];
  const nextRows = rows.map(row => {
    const record = recordValue(row);
    if (Number(record.id ?? record.number) !== channelId) return row;
    const control = recordValue(record.control_state);
    if (field === '$name' || field === 'name') {
      const name = stringValue(value);
      return name ? { ...record, name } : record;
    }
    const controlKey = field === 'fdr' ? 'fader_db' : field;
    return {
      ...record,
      control_state: {
        ...control,
        [controlKey]: value
      }
    };
  });

  return {
    ...current,
    dashboard: nextDashboard,
    channels: rows.length ? { ...inventory, channels: nextRows } : current.channels
  };
}

export function useOperatorBackend() {
  const [connectionState, setConnectionState] = useState<BackendConnectionState>(() => (
    socketIsOpen() ? 'connected' : shouldAutoConnect() ? 'connecting' : 'mock'
  ));
  const [payloads, setPayloads] = useState<ProductLayerPayloads>({});
  const [modeFromBackend, setModeFromBackend] = useState<AutomixerMode | null>(null);
  const [logs, setLogs] = useState<LogRecord[]>(mockLogs);
  const [proposalApplyResult, setProposalApplyResult] = useState<ProposalApplyResult | null>(null);
  const [soundcheck, setSoundcheck] = useState<SoundcheckRuntime>(idleSoundcheckRuntime);
  const [emergencyStop, setEmergencyStop] = useState<EmergencyStopUiState>(idleEmergencyStop);
  const emergencyStopAckTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearEmergencyStopAckTimer = useCallback(() => {
    if (emergencyStopAckTimer.current !== null) {
      globalThis.clearTimeout(emergencyStopAckTimer.current);
      emergencyStopAckTimer.current = null;
    }
  }, []);

  const requestSnapshot = useCallback(() => {
    websocketService.getOperatorMode();
    websocketService.getManualWriteSupervision();
    websocketService.getDashboardSnapshot();
    websocketService.getChannelInventory(48);
    websocketService.getDecisionQueue(50);
    websocketService.getDecisionTraces(50);
    websocketService.getSafetyMonitor();
    websocketService.getReplayRuns(20);
    websocketService.getChannelIntelligence(48);
    websocketService.getOperatorAgentRoster();
    websocketService.getConnectionTopology();
    websocketService.getGroupsBusesSnapshot();
    websocketService.getMemoryKnowledgeSnapshot();
    websocketService.getTrainingSessionsSnapshot();
    websocketService.getRulesLibrarySnapshot();
    websocketService.getAgentCommunicationsSnapshot();
    websocketService.getProcessingPipelineSnapshot(false);
    websocketService.getMainPipelinePreview(48, false);
    websocketService.getTelemetryAlertsSnapshot(50, 8);
    websocketService.getAudioCaptureStatus();
    websocketService.getAutoSoundcheckStatus();
    websocketService.subscribeMeters();
  }, []);

  const refreshBackend = useCallback(() => {
    if (socketIsOpen()) {
      requestSnapshot();
      return;
    }

    setConnectionState('connecting');
    websocketService.connect()
      .then(() => {
        setConnectionState('connected');
        setLogs(current => appendBackendLog(current, 'backend', 'Product Layer websocket connected.', 'SUCCESS'));
        requestSnapshot();
      })
      .catch(() => {
        setConnectionState('error');
        setLogs(current => appendBackendLog(current, 'backend', 'Product Layer unavailable; mock fallback remains active.', 'WARN'));
      });
  }, [requestSnapshot]);

  const runBackendAnalysis = useCallback(() => {
    if (!socketIsOpen()) {
      refreshBackend();
      setLogs(current => appendBackendLog(current, 'backend', 'Connect backend before running analysis.', 'WARN'));
      return;
    }
    websocketService.runOperatorAnalysis(48, true);
  }, [refreshBackend]);

  const startReplayRun = useCallback((runId?: string) => {
    if (!socketIsOpen()) {
      refreshBackend();
      setLogs(current => appendBackendLog(current, 'replay_run', 'Connect backend before starting offline replay.', 'WARN'));
      return;
    }
    websocketService.startReplayRun(runId ?? `frontend-replay-${Date.now()}`);
    setLogs(current => appendBackendLog(current, 'replay_run', 'Offline replay requested. Live writes remain disabled.', 'INFO'));
  }, [refreshBackend]);

  const toggleAudioPreview = useCallback(() => {
    if (!socketIsOpen()) {
      refreshBackend();
      setLogs(current => appendBackendLog(current, 'audio_capture', 'Connect backend before changing read-only audio meters.', 'WARN'));
      return;
    }
    const dashboard = recordValue(payloads.dashboard);
    const audioCapture = recordValue(dashboard.audio_capture);
    if (audioCapture.running) {
      websocketService.stopAudioCapturePreview();
      setLogs(current => appendBackendLog(current, 'audio_capture', 'Stopping read-only WING audio meters.', 'INFO'));
      return;
    }
    websocketService.startAudioCapturePreview(null, 48, 48000);
    setLogs(current => appendBackendLog(current, 'audio_capture', 'Starting read-only WING 48-channel meters.', 'INFO'));
  }, [payloads.dashboard, refreshBackend]);

  const connectWingReadOnly = useCallback(() => {
    const connect = () => {
      websocketService.connectWing('192.168.1.102', 2223, 2223, true);
      setLogs(current => appendBackendLog(current, 'connection', 'Requested WING 192.168.1.102 in observation-only mode.', 'INFO'));
    };
    if (socketIsOpen()) {
      connect();
      return;
    }
    setConnectionState('connecting');
    websocketService.connect()
      .then(() => {
        setConnectionState('connected');
        connect();
      })
      .catch(() => {
        setConnectionState('error');
        setLogs(current => appendBackendLog(current, 'connection', 'Backend unavailable; WING read-only connect was not sent.', 'WARN'));
      });
  }, []);

  const setBackendMode = useCallback((mode: AutomixerMode) => {
    websocketService.setOperatorMode(toBackendMode(mode), 'frontend_operator_mode_dropdown');
  }, []);

  const acceptProposal = useCallback((proposalId: string) => {
    websocketService.acceptOperatorProposal(proposalId);
  }, []);

  const dismissProposal = useCallback((proposalId: string) => {
    websocketService.dismissOperatorProposal(proposalId, 'frontend_operator_dismiss');
  }, []);

  const applyProposal = useCallback((proposalId: string) => {
    setProposalApplyResult({
      proposalId,
      status: 'requested',
      reason: 'frontend_supervised_operator_approved_apply',
      applyRequest: 'apply requested',
      readbackStatus: 'waiting for backend readback',
      rollbackStatus: 'waiting for rollback proof',
      rollbackAvailable: false,
      readbackTone: 'yellow',
      rollbackTone: 'yellow',
      tone: 'yellow',
      updatedAt: new Date().toLocaleTimeString()
    });
    websocketService.applyOperatorProposal(proposalId, true, 'frontend_supervised_operator_approved_apply');
  }, []);

  const startSoundcheck = useCallback(() => {
    if (!socketIsOpen()) {
      refreshBackend();
      setLogs(current => appendBackendLog(current, 'auto_soundcheck', 'Connect backend before starting soundcheck.', 'WARN'));
      return;
    }
    const selectedChannels = activeSoundcheckChannels(payloads.channels);
    if (!selectedChannels.length) {
      setLogs(current => appendBackendLog(
        current,
        'auto_soundcheck',
        'No active backend channels are available; soundcheck was not started.',
        'WARN'
      ));
      return;
    }
    websocketService.startAutoSoundcheck(null, selectedChannels, {}, {}, {}, true);
    setSoundcheck(current => ({
      ...current,
      running: true,
      state: 'starting',
      message: 'Proposal-only soundcheck requested.',
      error: '',
      observeOnly: true,
      proposalOnly: true,
      derivationComplete: false,
      readiness: {
        ...idleSoundcheckRuntime.readiness,
        runId: 'starting',
        executionStatus: 'pending'
      },
      replan: { ...idleSoundcheckRuntime.replan },
      recommendationCount: 0,
      transactionPlanCount: 0,
      transactionOperationCount: 0,
      liveAllowedFamilies: [],
      familyLifecycle: {},
      familyLiveWriteKinds: {},
      familyWriteKinds: {},
      familyContracts: {},
      selectedChannels,
      totalChannels: selectedChannels.length,
      channelsWithSignal: 0,
      interceptedWriteCount: 0,
      lastUpdated: new Date().toLocaleTimeString()
    }));
    setLogs(current => appendBackendLog(current, 'auto_soundcheck', 'Proposal-only soundcheck requested. Mixer writes remain blocked.', 'INFO'));
  }, [payloads.channels, refreshBackend]);

  const stopSoundcheck = useCallback(() => {
    if (!socketIsOpen()) return;
    websocketService.stopAutoSoundcheck();
    setLogs(current => appendBackendLog(current, 'auto_soundcheck', 'Soundcheck stop requested.', 'INFO'));
  }, []);

  const acceptSoundcheckReplan = useCallback(() => {
    if (
      !socketIsOpen()
      || !soundcheck.replan.draftAvailable
      || !soundcheck.replan.runId
      || !soundcheck.replan.draftId
    ) return;
    const queued = websocketService.acceptSoundcheckReplan(
      soundcheck.replan.runId,
      soundcheck.replan.draftId,
      `frontend-soundcheck-replan-${Date.now()}`
    );
    setLogs(current => appendBackendLog(
      current,
      'soundcheck_replan',
      queued
        ? 'Safe next-revision request queued; waiting for backend acknowledgement.'
        : 'Safe next-revision request was not sent.',
      queued ? 'INFO' : 'WARN'
    ));
  }, [soundcheck.replan]);

  const reanalyzeSoundcheckStage = useCallback(() => {
    const runId = soundcheck.readiness.runId;
    if (!socketIsOpen() || !runId || ['idle', 'starting', 'unknown'].includes(runId)) return;
    const queued = websocketService.reanalyzeSoundcheckStage(
      runId,
      `frontend-soundcheck-reanalysis-${Date.now()}`
    );
    setLogs(current => appendBackendLog(
      current,
      'soundcheck_reanalysis',
      queued
        ? 'Fresh read-only stage analysis queued; a new immutable revision will require review.'
        : 'Stage reanalysis request was not sent.',
      queued ? 'INFO' : 'WARN'
    ));
  }, [soundcheck.readiness.runId]);

  const finalizeSoundcheckReadiness = useCallback(() => {
    const runId = soundcheck.readiness.runId;
    if (!socketIsOpen() || !runId || ['idle', 'starting', 'unknown'].includes(runId)) return;
    const queued = websocketService.finalizeSoundcheckReadiness(runId);
    setLogs(current => appendBackendLog(
      current,
      'soundcheck_finalize',
      queued
        ? 'Fresh read-only target/Main reconciliation request queued.'
        : 'Read-only reconciliation request was not sent.',
      queued ? 'INFO' : 'WARN'
    ));
  }, [soundcheck.readiness.runId]);

  const triggerEmergencyStop = useCallback(() => {
    clearEmergencyStopAckTimer();
    if (!socketIsOpen()) {
      setEmergencyStop({
        status: 'error',
        message: 'STOP NOT SENT: backend is offline. Reconnect and retry.',
        updatedAt: new Date().toLocaleTimeString(),
        reconciliationRequired: true
      });
      setLogs(current => appendBackendLog(current, 'emergency_stop', 'Emergency stop was not sent because the backend is offline.', 'ERROR'));
      return false;
    }

    const sent = websocketService.emergencyStop();
    if (!sent) {
      setEmergencyStop({
        status: 'error',
        message: 'STOP NOT SENT: WebSocket send failed. Retry now.',
        updatedAt: new Date().toLocaleTimeString(),
        reconciliationRequired: true
      });
      setLogs(current => appendBackendLog(current, 'emergency_stop', 'Emergency stop WebSocket send failed.', 'ERROR'));
      return false;
    }

    setEmergencyStop({
      status: 'pending',
      message: 'Emergency stop sent. Waiting for backend acknowledgement.',
      updatedAt: new Date().toLocaleTimeString(),
      reconciliationRequired: true
    });
    setLogs(current => appendBackendLog(current, 'emergency_stop', 'System emergency stop requested.', 'WARN'));
    emergencyStopAckTimer.current = globalThis.setTimeout(() => {
      emergencyStopAckTimer.current = null;
      setEmergencyStop(current => current.status === 'pending' ? {
        status: 'error',
        message: 'STOP ACK MISSING: backend did not confirm within 5 seconds.',
        updatedAt: new Date().toLocaleTimeString(),
        reconciliationRequired: true
      } : current);
      setLogs(current => appendBackendLog(current, 'emergency_stop', 'Emergency stop acknowledgement timed out.', 'ERROR'));
      websocketService.getManualWriteSupervision();
    }, EMERGENCY_STOP_ACK_TIMEOUT_MS);
    return true;
  }, [clearEmergencyStopAckTimer]);

  useEffect(() => () => clearEmergencyStopAckTimer(), [clearEmergencyStopAckTimer]);

  useEffect(() => {
    const handleDashboard = (payload: unknown) => {
      setPayloads(current => mergeDashboardSnapshotIntoPayloads(current, payload));
      setLogs(current => appendBackendLog(current, 'dashboard_snapshot', 'Dashboard snapshot received.', 'SUCCESS'));
    };
    const handleChannels = (payload: unknown) => {
      setPayloads(current => ({ ...current, channels: payload }));
      const candidates = activeSoundcheckChannels(payload);
      setSoundcheck(current => current.running ? current : {
        ...current,
        selectedChannels: candidates,
        totalChannels: candidates.length
      });
      setLogs(current => appendBackendLog(current, 'channel_inventory', 'Channel inventory received.', 'SUCCESS'));
    };
    const handleDecisions = (payload: unknown) => {
      setPayloads(current => ({ ...current, decisions: payload }));
      setLogs(current => appendBackendLog(current, 'decision_queue', 'Decision queue received.', 'SUCCESS'));
    };
    const handleDecisionTraces = (payload: unknown) => {
      setPayloads(current => ({ ...current, decisionTraces: payload }));
      setLogs(current => appendBackendLog(current, 'decision_traces', 'Decision traces received.', 'SUCCESS'));
    };
    const handleSafetyMonitor = (payload: unknown) => {
      setPayloads(current => ({ ...current, safetyMonitor: payload }));
      setLogs(current => appendBackendLog(current, 'safety_monitor', 'Safety monitor received.', 'SUCCESS'));
    };
    const handleReplayRuns = (payload: unknown) => {
      setPayloads(current => ({ ...current, replayRuns: mergeReplayRunsPayload(current.replayRuns, payload) }));
      setLogs(current => appendBackendLog(current, 'replay_runs', 'Replay runs received.', 'SUCCESS'));
    };
    const handleReplayRunStarted = (payload: unknown) => {
      setLogs(current => appendBackendLog(current, 'replay_run', 'Offline replay run completed.', 'SUCCESS'));
      setPayloads(current => ({ ...current, replayRuns: replayRunsPayloadFromStarted(current.replayRuns, payload) }));
      refreshBackend();
    };
    const handleChannelIntelligence = (payload: unknown) => {
      setPayloads(current => ({ ...current, channelIntelligence: payload }));
      setLogs(current => appendBackendLog(current, 'channel_intelligence', 'Channel intelligence received.', 'SUCCESS'));
    };
    const handleAgents = (payload: unknown) => {
      setPayloads(current => ({ ...current, agents: payload }));
      setLogs(current => appendBackendLog(current, 'operator_agent_roster', 'Agent roster received.', 'SUCCESS'));
    };
    const handleTopology = (payload: unknown) => {
      setPayloads(current => ({ ...current, topology: payload }));
      setLogs(current => appendBackendLog(current, 'connection_topology', 'Connection topology received.', 'SUCCESS'));
    };
    const handleGroups = (payload: unknown) => {
      setPayloads(current => ({ ...current, groups: payload }));
      setLogs(current => appendBackendLog(current, 'groups_buses_snapshot', 'Groups/buses snapshot received.', 'SUCCESS'));
    };
    const handleMemory = (payload: unknown) => {
      setPayloads(current => ({ ...current, memory: payload }));
      setLogs(current => appendBackendLog(current, 'memory_knowledge_snapshot', 'Memory/knowledge snapshot received.', 'SUCCESS'));
    };
    const handleTraining = (payload: unknown) => {
      setPayloads(current => ({ ...current, training: payload }));
      setLogs(current => appendBackendLog(current, 'training_sessions_snapshot', 'Training sessions snapshot received.', 'SUCCESS'));
    };
    const handleRules = (payload: unknown) => {
      setPayloads(current => ({ ...current, rules: payload }));
      setLogs(current => appendBackendLog(current, 'rules_library_snapshot', 'Rules library snapshot received.', 'SUCCESS'));
    };
    const handleCommunications = (payload: unknown) => {
      setPayloads(current => ({ ...current, communications: payload }));
      setLogs(current => appendBackendLog(current, 'agent_communications_snapshot', 'Agent communications snapshot received.', 'SUCCESS'));
    };
    const handleProcessing = (payload: unknown) => {
      setPayloads(current => ({ ...current, processing: payload }));
      setLogs(current => appendBackendLog(current, 'processing_pipeline_snapshot', 'Processing pipeline snapshot received.', 'SUCCESS'));
    };
    const handleMainPipelinePreview = (payload: unknown) => {
      setPayloads(current => ({ ...current, pipelinePreview: payload }));
      setLogs(current => appendBackendLog(current, 'main_pipeline_preview', 'Main pipeline preview received. No proposals were queued.', 'SUCCESS'));
    };
    const handleTelemetryAlerts = (payload: unknown) => {
      setPayloads(current => ({ ...current, telemetryAlerts: payload }));
      setLogs(current => appendBackendLog(current, 'telemetry_alerts_snapshot', 'Telemetry alerts snapshot received. Read-only display only.', 'SUCCESS'));
    };
    const handleMode = (payload: unknown) => {
      setPayloads(current => ({ ...current, mode: payload }));
      setModeFromBackend(toUiMode((payload as { mode?: unknown })?.mode));
      setLogs(current => appendBackendLog(current, 'operator_mode_status', 'Operator mode status received.', 'INFO'));
    };
    const handleManualWriteSupervision = (payload: unknown) => {
      const data = payload && typeof payload === 'object' ? payload as Record<string, unknown> : {};
      const emergencyStopped = data.emergency_stopped === true || data.reset_required === true;
      if (emergencyStopped) clearEmergencyStopAckTimer();
      setEmergencyStop(current => {
        if (emergencyStopped) {
          return {
            status: 'acknowledged',
            message: 'STOP ACTIVE: backend supervision confirms the emergency latch is engaged.',
            updatedAt: new Date().toLocaleTimeString(),
            reconciliationRequired: false
          };
        }
        if (current.status === 'pending') return current;
        if (current.status === 'acknowledged' || current.reconciliationRequired) {
          return {
            status: 'ready',
            message: 'Backend reports the emergency stop is reset. Stop control is ready.',
            updatedAt: new Date().toLocaleTimeString(),
            reconciliationRequired: false
          };
        }
        return current;
      });
      setLogs(current => appendBackendLog(
        current,
        'manual_write_supervision',
        emergencyStopped ? 'Manual write emergency reset required.' : 'Manual write supervision clear.',
        emergencyStopped ? 'WARN' : 'INFO'
      ));
    };
    const handleEmergencyResetResult = (payload: unknown) => {
      const data = payload && typeof payload === 'object' ? payload as Record<string, unknown> : {};
      const status = String(data.status ?? 'updated');
      const level: LogRecord['level'] = status === 'blocked' ? 'WARN' : 'SUCCESS';
      setLogs(current => appendBackendLog(current, 'manual_write_emergency_reset', `Emergency reset ${status}.`, level));
      if (data.success === true && ['reset', 'not_required'].includes(status)) {
        clearEmergencyStopAckTimer();
        setEmergencyStop({
          status: 'ready',
          message: 'Emergency stop reset acknowledged. Stop control is ready.',
          updatedAt: new Date().toLocaleTimeString(),
          reconciliationRequired: false
        });
      }
      refreshBackend();
    };
    const handleEmergencyStopApplied = (payload: unknown) => {
      const data = recordValue(payload);
      const status = stringValue(data.status).toLowerCase();
      const supervision = recordValue(data.manual_write_supervision);
      const acknowledged = data.success === true
        && status === 'stopped'
        && (supervision.emergency_stopped === true || supervision.reset_required === true);
      clearEmergencyStopAckTimer();
      setEmergencyStop({
        status: acknowledged ? 'acknowledged' : 'error',
        message: acknowledged
          ? 'STOP ACTIVE: backend confirmed the emergency latch is engaged.'
          : `STOP ERROR: backend response was not an acknowledgement${status ? ` (${status})` : ''}.`,
        updatedAt: new Date().toLocaleTimeString(),
        reconciliationRequired: !acknowledged
      });
      setLogs(current => appendBackendLog(
        current,
        'emergency_stop',
        acknowledged ? 'Backend acknowledged the system emergency stop.' : 'Backend rejected or failed the system emergency stop.',
        acknowledged ? 'SUCCESS' : 'ERROR'
      ));
      websocketService.getManualWriteSupervision();
      websocketService.getSafetyMonitor();
    };
    const handleAnalysis = (payload: unknown) => {
      setLogs(current => appendBackendLog(current, 'operator_analysis_report', 'Operator analysis completed.', 'SUCCESS'));
      setPayloads(current => ({ ...current, decisions: current.decisions ?? payload }));
      refreshBackend();
    };
    const handleProposalMutation = (payload: unknown) => {
      const status = String((payload as { status?: unknown })?.status ?? 'updated');
      const level: LogRecord['level'] = status.includes('blocked') ? 'WARN' : 'SUCCESS';
      setLogs(current => appendBackendLog(current, 'operator_proposal', `Proposal ${status}.`, level));
      refreshBackend();
    };
    const handleProposalApplyResult = (payload: unknown) => {
      setProposalApplyResult(applyResultFromPayload(payload, 'applied'));
      handleProposalMutation(payload);
    };
    const handleProposalApplyBlocked = (payload: unknown) => {
      setProposalApplyResult(applyResultFromPayload(payload, 'apply_blocked'));
      handleProposalMutation(payload);
    };
    const handleMeterFrame = (payload: unknown) => {
      const frame = payload && typeof payload === 'object' ? payload as Record<string, unknown> : null;
      if (!frame) return;
      setPayloads(current => mergeMeterFrameIntoPayloads(current, frame));
    };
    const handleAudioCaptureStatus = (payload: unknown) => {
      const data = recordValue(payload);
      setPayloads(current => mergeAudioCaptureStatusIntoPayloads(current, data));
      const running = Boolean(data.running);
      const level: LogRecord['level'] = data.error ? 'WARN' : running ? 'SUCCESS' : 'INFO';
      setLogs(current => appendBackendLog(
        current,
        'audio_capture_status',
        data.error ? stringValue(data.error) : running ? `Read-only audio meters online: ${numberValue(data.num_channels, 0)} channels.` : 'Read-only audio meters stopped.',
        level
      ));
      websocketService.getDashboardSnapshot();
      websocketService.getChannelInventory(48);
      websocketService.getConnectionTopology();
      websocketService.getAutoSoundcheckStatus();
    };
    const handleAutoSoundcheckStatus = (payload: unknown) => {
      setSoundcheck(current => soundcheckRuntimeFromPayload(current, payload));
      const data = recordValue(payload);
      const error = stringValue(data.error);
      setLogs(current => appendBackendLog(
        current,
        'auto_soundcheck',
        error || stringValue(data.message, `Soundcheck ${stringValue(data.state, 'status updated')}.`),
        error ? 'WARN' : data.running || data.is_running ? 'INFO' : 'SUCCESS'
      ));
    };
    const handleAutoSoundcheckObservation = (payload: unknown) => {
      setSoundcheck(current => soundcheckRuntimeFromPayload(current, payload));
    };
    const handleAutoSoundcheckChannel = () => {
      websocketService.getAutoSoundcheckStatus();
    };
    const handleSoundcheckProposalImport = (payload: unknown) => {
      const data = recordValue(payload);
      if (stringValue(data.source) !== 'soundcheck_recommendation_bundle') return;
      setSoundcheck(current => {
        const parsed = soundcheckRuntimeFromPayload(current, payload);
        return {
          ...parsed,
          recommendationCount: numberValue(
            data.imported_count ?? data.created_count,
            current.recommendationCount
          ),
          derivationComplete: Boolean(data.recommendation_derivation_complete),
          readiness: Object.prototype.hasOwnProperty.call(data, 'soundcheck_readiness')
            ? parsed.readiness
            : {
              ...idleSoundcheckRuntime.readiness,
              runId: 'missing-readiness-contract',
              safetyState: 'manual_review_required'
            },
          lastUpdated: new Date().toLocaleTimeString()
        };
      });
    };
    const handleSoundcheckControlResult = (payload: unknown) => {
      const data = recordValue(payload);
      setSoundcheck(current => soundcheckRuntimeFromPayload(current, payload));
      const success = data.success === true;
      setLogs(current => appendBackendLog(
        current,
        stringValue(data.type, 'soundcheck_control'),
        success
          ? 'Soundcheck read-only finalize/replan request completed.'
          : `Soundcheck control blocked: ${stringValue(data.reason, 'unknown reason')}.`,
        success ? 'SUCCESS' : 'WARN'
      ));
      websocketService.getAutoSoundcheckStatus();
    };
    const handleConnectionStatus = (payload: unknown) => {
      const data = recordValue(payload);
      setPayloads(current => mergeConnectionStatusIntoPayloads(current, data));
      setLogs(current => appendBackendLog(
        current,
        'connection_status',
        data.connected ? `WING connected in ${data.observation_only === false ? 'write-capable' : 'observation-only'} mode.` : stringValue(data.error, 'Mixer disconnected.'),
        data.connected ? 'SUCCESS' : 'WARN'
      ));
      requestSnapshot();
    };
    const handleMixerUpdate = (payload: unknown) => {
      const data = recordValue(payload);
      setPayloads(current => mergeMixerUpdateIntoPayloads(current, data));
    };
    const handleDisconnected = () => {
      setConnectionState(previous => previous === 'mock' ? 'mock' : 'offline');
      setProposalApplyResult(null);
      setSoundcheck(current => ({
        ...current,
        running: false,
        state: 'disconnected',
        message: 'Backend disconnected; soundcheck readiness is no longer valid.',
        readiness: {
          ...current.readiness,
          mixReadiness: 'soundcheck_incomplete',
          safetyState: 'manual_review_required',
          evidenceComplete: false,
          globalBlockers: [
            ...new Set([...current.readiness.globalBlockers, 'backend_disconnected'])
          ]
        }
      }));
      clearEmergencyStopAckTimer();
      setEmergencyStop(current => ['pending', 'acknowledged'].includes(current.status) ? {
        status: 'error',
        message: 'STOP STATE UNKNOWN: backend disconnected; reconnect to verify the emergency latch.',
        updatedAt: new Date().toLocaleTimeString(),
        reconciliationRequired: true
      } : current);
      setLogs(current => appendBackendLog(current, 'backend', 'Product Layer disconnected; using mock fallback.', 'WARN'));
    };
    const handleConnected = () => {
      setPayloads({});
      setModeFromBackend(null);
      setProposalApplyResult(null);
      setConnectionState('connected');
      setLogs(current => appendBackendLog(
        current,
        'backend',
        'Product Layer websocket connected.',
        'SUCCESS'
      ));
      requestSnapshot();
    };

    websocketService.on('dashboard_snapshot', handleDashboard);
    websocketService.on('channel_inventory', handleChannels);
    websocketService.on('decision_queue', handleDecisions);
    websocketService.on('decision_traces', handleDecisionTraces);
    websocketService.on('safety_monitor', handleSafetyMonitor);
    websocketService.on('replay_runs', handleReplayRuns);
    websocketService.on('replay_run_started', handleReplayRunStarted);
    websocketService.on('channel_intelligence', handleChannelIntelligence);
    websocketService.on('operator_agent_roster', handleAgents);
    websocketService.on('connection_topology', handleTopology);
    websocketService.on('groups_buses_snapshot', handleGroups);
    websocketService.on('memory_knowledge_snapshot', handleMemory);
    websocketService.on('training_sessions_snapshot', handleTraining);
    websocketService.on('rules_library_snapshot', handleRules);
    websocketService.on('agent_communications_snapshot', handleCommunications);
    websocketService.on('processing_pipeline_snapshot', handleProcessing);
    websocketService.on('main_pipeline_preview', handleMainPipelinePreview);
    websocketService.on('telemetry_alerts_snapshot', handleTelemetryAlerts);
    websocketService.on('operator_mode_status', handleMode);
    websocketService.on('manual_write_supervision_status', handleManualWriteSupervision);
    websocketService.on('manual_write_emergency_reset_result', handleEmergencyResetResult);
    websocketService.on('emergency_stop_applied', handleEmergencyStopApplied);
    websocketService.on('operator_analysis_report', handleAnalysis);
    websocketService.on('operator_proposal_created', handleProposalMutation);
    websocketService.on('operator_proposal_accepted', handleProposalMutation);
    websocketService.on('operator_proposal_dismissed', handleProposalMutation);
    websocketService.on('operator_proposal_apply_result', handleProposalApplyResult);
    websocketService.on('operator_proposal_apply_blocked', handleProposalApplyBlocked);
    websocketService.on('operator_proposals_imported', handleProposalMutation);
    websocketService.on('meter_frame', handleMeterFrame);
    websocketService.on('audio_capture_status', handleAudioCaptureStatus);
    websocketService.on('auto_soundcheck_status', handleAutoSoundcheckStatus);
    websocketService.on('auto_soundcheck_observation', handleAutoSoundcheckObservation);
    websocketService.on('auto_soundcheck_channel_update', handleAutoSoundcheckChannel);
    websocketService.on('operator_proposals_imported', handleSoundcheckProposalImport);
    websocketService.on('soundcheck_replan_result', handleSoundcheckControlResult);
    websocketService.on('soundcheck_stage_reanalysis_result', handleSoundcheckControlResult);
    websocketService.on('soundcheck_finalize_result', handleSoundcheckControlResult);
    websocketService.on('connection_status', handleConnectionStatus);
    websocketService.on('mixer_update', handleMixerUpdate);
    websocketService.on('disconnected', handleDisconnected);
    websocketService.on('connected', handleConnected);

    if (socketIsOpen()) {
      handleConnected();
    } else if (shouldAutoConnect()) {
      setConnectionState('connecting');
      websocketService.connect()
        .then(() => undefined)
        .catch(() => {
          setConnectionState('error');
          setLogs(current => appendBackendLog(current, 'backend', 'Product Layer unavailable; mock fallback remains active.', 'WARN'));
        });
    }

    return () => {
      websocketService.off('dashboard_snapshot', handleDashboard);
      websocketService.off('channel_inventory', handleChannels);
      websocketService.off('decision_queue', handleDecisions);
      websocketService.off('decision_traces', handleDecisionTraces);
      websocketService.off('safety_monitor', handleSafetyMonitor);
      websocketService.off('replay_runs', handleReplayRuns);
      websocketService.off('replay_run_started', handleReplayRunStarted);
      websocketService.off('channel_intelligence', handleChannelIntelligence);
      websocketService.off('operator_agent_roster', handleAgents);
      websocketService.off('connection_topology', handleTopology);
      websocketService.off('groups_buses_snapshot', handleGroups);
      websocketService.off('memory_knowledge_snapshot', handleMemory);
      websocketService.off('training_sessions_snapshot', handleTraining);
      websocketService.off('rules_library_snapshot', handleRules);
      websocketService.off('agent_communications_snapshot', handleCommunications);
      websocketService.off('processing_pipeline_snapshot', handleProcessing);
      websocketService.off('main_pipeline_preview', handleMainPipelinePreview);
      websocketService.off('telemetry_alerts_snapshot', handleTelemetryAlerts);
      websocketService.off('operator_mode_status', handleMode);
      websocketService.off('manual_write_supervision_status', handleManualWriteSupervision);
      websocketService.off('manual_write_emergency_reset_result', handleEmergencyResetResult);
      websocketService.off('emergency_stop_applied', handleEmergencyStopApplied);
      websocketService.off('operator_analysis_report', handleAnalysis);
      websocketService.off('operator_proposal_created', handleProposalMutation);
      websocketService.off('operator_proposal_accepted', handleProposalMutation);
      websocketService.off('operator_proposal_dismissed', handleProposalMutation);
      websocketService.off('operator_proposal_apply_result', handleProposalApplyResult);
      websocketService.off('operator_proposal_apply_blocked', handleProposalApplyBlocked);
      websocketService.off('operator_proposals_imported', handleProposalMutation);
      websocketService.off('meter_frame', handleMeterFrame);
      websocketService.off('audio_capture_status', handleAudioCaptureStatus);
      websocketService.off('auto_soundcheck_status', handleAutoSoundcheckStatus);
      websocketService.off('auto_soundcheck_observation', handleAutoSoundcheckObservation);
      websocketService.off('auto_soundcheck_channel_update', handleAutoSoundcheckChannel);
      websocketService.off('operator_proposals_imported', handleSoundcheckProposalImport);
      websocketService.off('soundcheck_replan_result', handleSoundcheckControlResult);
      websocketService.off('soundcheck_stage_reanalysis_result', handleSoundcheckControlResult);
      websocketService.off('soundcheck_finalize_result', handleSoundcheckControlResult);
      websocketService.off('connection_status', handleConnectionStatus);
      websocketService.off('mixer_update', handleMixerUpdate);
      websocketService.off('disconnected', handleDisconnected);
      websocketService.off('connected', handleConnected);
      if (socketIsOpen()) {
        websocketService.unsubscribeMeters();
      }
    };
  }, [clearEmergencyStopAckTimer, refreshBackend, requestSnapshot]);

  const backend: BackendUiState = useMemo(() => {
    const rawSourcePayload = payloads.dashboard ?? payloads.mode ?? {};
    const packagedOfflineLock = window.automixerRuntimeSafety?.packagedOfflineLock === true;
    const rawSourceRecord = recordValue(rawSourcePayload);
    const sourcePayload = packagedOfflineLock
      ? {
          ...rawSourceRecord,
          operator_mode: {
            ...recordValue(rawSourceRecord.operator_mode ?? rawSourceRecord),
            packaged_offline_lock: true,
            mixer_transport_locked: true,
            mode: 'analyze'
          }
        }
      : rawSourcePayload;
    const base = adaptBackendState(sourcePayload, connectionState, connectionMessage(connectionState));
    const receivedPayloads = payloadKeys(payloads);
    const dataState = resolveDataState(connectionState, payloads);
    const hasRuntimeData = receivedPayloads.length > 0;
    return {
      ...base,
      dataState,
      dataStateLabel: dataStateLabel(dataState),
      dataStateTone: dataStateTone(dataState),
      hasRuntimeData,
      isPartialData: dataState === 'partial',
      staleSnapshot: dataState === 'stale',
      sourceLabel: hasRuntimeData ? base.sourceLabel : mockBackendState.sourceLabel,
      statusLabel: dataState === 'partial' ? 'Partial data' : dataState === 'stale' ? 'Stale data' : base.statusLabel,
      statusTone: dataState === 'partial' || dataState === 'stale' ? 'yellow' : base.statusTone,
      proposalApplyResult
    };
  }, [connectionState, payloads.agents, payloads.channelIntelligence, payloads.channels, payloads.communications, payloads.dashboard, payloads.decisionTraces, payloads.decisions, payloads.groups, payloads.memory, payloads.mode, payloads.pipelinePreview, payloads.processing, payloads.replayRuns, payloads.rules, payloads.safetyMonitor, payloads.telemetryAlerts, payloads.topology, payloads.training, proposalApplyResult]);

  const viewData: OperatorViewData = useMemo(() => {
    const channels = adaptChannels(payloads.channels, mockChannels);
    const derivedBuses = adaptBusesFromChannels(channels, mockBuses);
    const buses = adaptBusesSnapshot(payloads.groups, derivedBuses);
    const agents = adaptAgents(payloads.agents, mockAgents);
    const queueTasks = adaptTasks(payloads.decisions, mockTasks);
    const reviewOnlyTasks = payloads.pipelinePreview ? adaptProposalOnlyContractTasks(payloads.pipelinePreview) : [];
    const tasks = [...queueTasks, ...reviewOnlyTasks];
    const decisionTraces = adaptDecisionTraces(payloads.decisionTraces);
    const replayRuns = adaptReplayRuns(payloads.replayRuns);
    const channelIntelligence = adaptChannelIntelligence(payloads.channelIntelligence);
    const overviewMetrics = adaptOverviewMetrics(payloads.dashboard, payloads.agents, payloads.decisions, mockOverviewMetrics);
    const devices = adaptDevices(payloads.topology, mockDevices);
    const topologyNodes = adaptTopologyNodes(payloads.topology);
    const topologyLinks = adaptTopologyLinks(payloads.topology);
    const routingSegments = adaptRoutingSegments(payloads.topology);
    const dashboardRuntime = adaptDashboardRuntime(payloads.dashboard, payloads.agents, payloads.decisions);
    const directorRuntime = adaptDirectorRuntime(payloads.dashboard, payloads.agents, payloads.pipelinePreview);
    const safetyMonitor = adaptSafetyMonitor(payloads.safetyMonitor, backend);
    const performanceRecords = adaptPerformanceRecords(payloads.agents, payloads.decisions, mockPerformance);
    const moduleStatuses = adaptModuleStatusRecords(payloads.processing);
    const telemetryAlerts = adaptTelemetryAlertsSnapshot(payloads.telemetryAlerts);
    const runtimeCommunications = adaptRuntimeCommunications(logs, agents, tasks, mockCommunications);
    const communications = adaptCommunicationsSnapshot(payloads.communications, runtimeCommunications);
    const hasRuntimePayloads = Boolean(payloads.channels || payloads.agents || payloads.decisions || payloads.dashboard || payloads.topology);
    const runtimeMemories = hasRuntimePayloads
      ? adaptRuntimeMemories(agents, tasks, channels, logs, mockMemories)
      : mockMemories;
    const memories = adaptMemorySnapshot(payloads.memory, runtimeMemories);
    const trainingSessions = adaptTrainingSnapshot(payloads.training, []);
    const rules = adaptRulesSnapshot(payloads.rules, []);
    return {
      channels,
      buses,
      agents,
      tasks,
      decisionTraces,
      safetyMonitor,
      replayRuns,
      channelIntelligence,
      overviewMetrics,
      devices,
      logs,
      moduleStatuses,
      telemetryAlerts,
      communications,
      memories,
      trainingSessions,
      rules,
      topologyNodes,
      topologyLinks,
      routingSegments,
      dashboardRuntime,
      directorRuntime,
      performanceRecords,
      readModelSources: {
        groups: payloadSource(payloads.groups, payloads.channels ? 'runtime_channel_inventory' : 'mock'),
        memory: payloadSource(payloads.memory, payloads.dashboard ? 'runtime_derived' : 'mock'),
        performance: payloadSource(payloads.agents, payloads.agents ? 'runtime_derived' : 'mock'),
        training: payloadSource(payloads.training, hasRuntimePayloads ? 'runtime_derived' : 'mock'),
        rules: payloadSource(payloads.rules, hasRuntimePayloads ? 'runtime_derived' : 'mock'),
        telemetryAlerts: payloadSource(payloads.telemetryAlerts, hasRuntimePayloads ? 'telemetry_alert_artifacts' : 'mock'),
        communications: payloadSource(payloads.communications, payloads.dashboard ? 'runtime_derived' : 'mock')
      }
    };
  }, [backend, logs, payloads.agents, payloads.channelIntelligence, payloads.channels, payloads.communications, payloads.dashboard, payloads.decisionTraces, payloads.decisions, payloads.groups, payloads.memory, payloads.pipelinePreview, payloads.processing, payloads.replayRuns, payloads.rules, payloads.safetyMonitor, payloads.telemetryAlerts, payloads.topology, payloads.training]);

  const backendActions = useMemo(() => ({
    refreshBackend,
    toggleAudioPreview,
    connectWingReadOnly,
    runBackendAnalysis,
    startReplayRun,
    setBackendMode,
    acceptProposal,
    dismissProposal,
    applyProposal,
    startSoundcheck,
    stopSoundcheck,
    acceptSoundcheckReplan,
    reanalyzeSoundcheckStage,
    finalizeSoundcheckReadiness,
    triggerEmergencyStop
  }), [acceptProposal, acceptSoundcheckReplan, applyProposal, connectWingReadOnly, dismissProposal, finalizeSoundcheckReadiness, reanalyzeSoundcheckStage, refreshBackend, runBackendAnalysis, setBackendMode, startReplayRun, startSoundcheck, stopSoundcheck, toggleAudioPreview, triggerEmergencyStop]);

  return {
    backend,
    soundcheck,
    emergencyStop,
    modeFromBackend,
    viewData,
    backendActions,
    connected: connectionState === 'connected',
    connectionState
  };
}
