import { useCallback, useEffect, useMemo, useState } from 'react';
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
  Task,
  TelemetryAlertsSnapshot,
  TrainingSessionSnapshotRow,
  TopologyLink,
  TopologyNode
} from '../types';

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

function startsOnMockDashboard() {
  if (typeof window === 'undefined') return false;
  const route = window.location.hash.replace(/^#\/?/, '');
  return route === '' || route === 'dashboard';
}

function shouldAutoConnect() {
  try {
    const preference = globalThis.localStorage?.getItem(AUTOCONNECT_STORAGE_KEY);
    if (preference === 'on') return true;
    if (preference === 'off') return false;
    return !startsOnMockDashboard();
  } catch {
    return !startsOnMockDashboard();
  }
}

function connectionMessage(state: BackendConnectionState) {
  if (state === 'connected') return 'Product Layer connected.';
  if (state === 'connecting') return 'Connecting to Product Layer.';
  if (state === 'error') return 'Backend connection failed. UI stays on mock fallback.';
  return mockBackendState.message;
}

function socketIsOpen() {
  const socket = (websocketService as { ws?: { readyState?: number } }).ws;
  const openState = globalThis.WebSocket?.OPEN ?? 1;
  return socket?.readyState === openState;
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

function applyResultFromPayload(payload: unknown, fallbackStatus = 'updated'): ProposalApplyResult {
  const data = payload && typeof payload === 'object' ? payload as Record<string, unknown> : {};
  const rawResult = data.apply_result && typeof data.apply_result === 'object'
    ? data.apply_result as Record<string, unknown>
    : data;
  const status = String(data.status ?? rawResult.status ?? fallbackStatus);
  const reason = String(rawResult.reason ?? data.reason ?? '');
  const readbackStatus = String(rawResult.readback_status ?? (status === 'applied' ? 'readback verified' : 'not verified'));
  const rollbackAvailable = Boolean(rawResult.rollback_available);
  const rollbackResults = Array.isArray(rawResult.rollback_results) ? rawResult.rollback_results : [];
  const rollbackFailed = rollbackResults.some(item => {
    const row = item && typeof item === 'object' ? item as Record<string, unknown> : {};
    return row.status && row.status !== 'applied';
  });
  const rollbackStatus = rollbackFailed
    ? 'rollback failed/unavailable'
    : rollbackResults.length
      ? 'rollback executed'
      : rollbackAvailable
        ? 'rollback available'
        : 'rollback unavailable';
  const blocked = status.includes('blocked') || reason.includes('blocked') || reason.includes('failed');
  return {
    proposalId: proposalIdFromPayload(payload),
    status,
    reason,
    applyRequest: blocked ? 'apply blocked' : status === 'applied' ? 'apply requested' : status,
    readbackStatus,
    rollbackStatus,
    rollbackAvailable,
    tone: blocked || rollbackFailed ? 'red' : status === 'applied' ? 'green' : 'yellow',
    updatedAt: new Date().toLocaleTimeString()
  };
}

export function useOperatorBackend() {
  const [connectionState, setConnectionState] = useState<BackendConnectionState>(() => shouldAutoConnect() ? 'connecting' : 'mock');
  const [payloads, setPayloads] = useState<ProductLayerPayloads>({});
  const [modeFromBackend, setModeFromBackend] = useState<AutomixerMode | null>(null);
  const [logs, setLogs] = useState<LogRecord[]>(mockLogs);
  const [proposalApplyResult, setProposalApplyResult] = useState<ProposalApplyResult | null>(null);

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
      tone: 'yellow',
      updatedAt: new Date().toLocaleTimeString()
    });
    websocketService.applyOperatorProposal(proposalId, true, 'frontend_supervised_operator_approved_apply');
  }, []);

  useEffect(() => {
    const handleDashboard = (payload: unknown) => {
      setPayloads(current => ({ ...current, dashboard: payload }));
      setLogs(current => appendBackendLog(current, 'dashboard_snapshot', 'Dashboard snapshot received.', 'SUCCESS'));
    };
    const handleChannels = (payload: unknown) => {
      setPayloads(current => ({ ...current, channels: payload }));
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
      const emergencyStopped = Boolean(data.emergency_stopped ?? data.reset_required);
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
      refreshBackend();
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
    const handleDisconnected = () => {
      setConnectionState(previous => previous === 'mock' ? 'mock' : 'offline');
      setLogs(current => appendBackendLog(current, 'backend', 'Product Layer disconnected; using mock fallback.', 'WARN'));
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
    websocketService.on('operator_analysis_report', handleAnalysis);
    websocketService.on('operator_proposal_created', handleProposalMutation);
    websocketService.on('operator_proposal_accepted', handleProposalMutation);
    websocketService.on('operator_proposal_dismissed', handleProposalMutation);
    websocketService.on('operator_proposal_apply_result', handleProposalApplyResult);
    websocketService.on('operator_proposal_apply_blocked', handleProposalApplyBlocked);
    websocketService.on('operator_proposals_imported', handleProposalMutation);
    websocketService.on('disconnected', handleDisconnected);

    if (shouldAutoConnect()) {
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
      websocketService.off('operator_analysis_report', handleAnalysis);
      websocketService.off('operator_proposal_created', handleProposalMutation);
      websocketService.off('operator_proposal_accepted', handleProposalMutation);
      websocketService.off('operator_proposal_dismissed', handleProposalMutation);
      websocketService.off('operator_proposal_apply_result', handleProposalApplyResult);
      websocketService.off('operator_proposal_apply_blocked', handleProposalApplyBlocked);
      websocketService.off('operator_proposals_imported', handleProposalMutation);
      websocketService.off('disconnected', handleDisconnected);
    };
  }, [refreshBackend, requestSnapshot]);

  const backend: BackendUiState = useMemo(() => {
    const sourcePayload = payloads.dashboard ?? payloads.mode ?? {};
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
    runBackendAnalysis,
    startReplayRun,
    setBackendMode,
    acceptProposal,
    dismissProposal,
    applyProposal
  }), [acceptProposal, applyProposal, dismissProposal, refreshBackend, runBackendAnalysis, setBackendMode, startReplayRun]);

  return {
    backend,
    modeFromBackend,
    viewData,
    backendActions,
    connected: connectionState === 'connected',
    connectionState
  };
}
