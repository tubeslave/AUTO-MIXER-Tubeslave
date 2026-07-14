import {
  agents,
  buses,
  channels,
  communications,
  devices,
  logs,
  memories,
  overviewMetrics,
  performance as mockPerformance,
  rules as mockRules,
  trainingSessions as mockTrainingSessions,
  tasks
} from '../mock/automixerData';
import type { OperatorViewData } from '../services/useOperatorBackend';
import { createUnavailableDecisionQueueState } from '../services/decisionQueueV2.mjs';
import type { DashboardRuntime, DirectorRuntime, RouteId, UiActions, UiState, ViewModel } from '../types';
import { AgentsConsole, Communications } from './AgentViews';
import { ChannelList, Groups } from './ChannelViews';
import { Dashboard } from './DashboardView';
import { Memory, RulesLibrary, Training } from './KnowledgeViews';
import { Performance } from './PerformanceView';
import { Settings } from './SettingsView';
import { SoundcheckView } from './SoundcheckView';
import { AudioRouting, Devices, Logs, OscMidi, SystemConnection } from './SystemViews';
import { TaskQueue } from './TaskQueueView';

const fallbackDashboardRuntime: DashboardRuntime = {
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
  sourceLabel: 'Mock',
  masterAvailable: false,
  integratedLufsAvailable: false,
  truePeakAvailable: false,
  shortTermLufsAvailable: false,
  momentaryLufsAvailable: false,
  lraAvailable: false,
  spectrumAvailable: false,
  aiConfidenceAvailable: false,
  audioCaptureRunning: false,
  mainFaderAvailable: false,
  readCoverageStatus: 'mock',
  cacheFreshnessStatus: 'mock'
};

const fallbackDirectorRuntime: DirectorRuntime = {
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

function fallbackViewData(): OperatorViewData {
  return {
    channels,
    buses,
    agents,
    tasks,
    decisionQueue: createUnavailableDecisionQueueState() as OperatorViewData['decisionQueue'],
    decisionTraces: [],
    safetyMonitor: {
      operatorMode: 'mock',
      liveWritePolicy: 'mock_frontend_only',
      applyPath: 'none',
      emergencyStopped: false,
      resetRequired: false,
      ready: false,
      mixerConnected: false,
      approvedLiveWriteKinds: [],
      blockedSurfaces: ['eq', 'compressor', 'routing', 'snapshots', 'autonomous_apply'],
      lastProofStatus: 'mock',
      lastReadbackStatus: 'mock',
      traceCount: 0,
      blockedTraceCount: 0,
      goalCapabilityObjectiveReadinessScore: 0,
      goalCapabilityObjectiveReadinessGoalAreaPriorities: [],
      goalCapabilityObjectiveReadinessByGoalArea: {},
      goalCapabilityObjectiveReadinessHotspots: [],
      proofGoalCapabilityObjectiveReadinessGoalAreaPriorities: [],
      proofGoalCapabilityObjectiveReadinessByGoalArea: {},
      proofGoalCapabilityObjectiveReadinessHotspots: [],
      sourceLabel: 'Mock'
    },
    replayRuns: [],
    channelIntelligence: [],
    overviewMetrics,
    devices,
    logs,
    moduleStatuses: [],
    telemetryAlerts: {
      available: false,
      generatedAt: 'mock',
      sourceLabel: 'mock',
      storageDir: '',
      alertCount: 0,
      criticalCount: 0,
      warningCount: 0,
      latestSessionId: '',
      readOnly: true,
      liveWrite: false,
      generatesArtifacts: false,
      sessions: [],
      alerts: []
    },
    communications,
    memories,
    trainingSessions: mockTrainingSessions,
    rules: mockRules,
    topologyNodes: [],
    topologyLinks: [],
    routingSegments: [],
    dashboardRuntime: fallbackDashboardRuntime,
    directorRuntime: fallbackDirectorRuntime,
    performanceRecords: mockPerformance,
    readModelSources: {
      groups: 'mock',
      memory: 'mock',
      performance: 'mock',
      training: 'mock',
      rules: 'mock',
      telemetryAlerts: 'mock',
      communications: 'mock'
    }
  };
}

export function buildView(route: RouteId, ui: UiState, actions: UiActions, data: OperatorViewData = fallbackViewData()): ViewModel {
  switch (route) {
    case 'dashboard': return Dashboard(ui, actions, data);
    case 'soundcheck': return SoundcheckView(ui, actions, data);
    case 'agents': return AgentsConsole(ui, actions, data);
    case 'communications': return Communications(ui, actions, data);
    case 'task-queue': return TaskQueue(ui, actions, data);
    case 'memory': return Memory(ui, actions, data);
    case 'performance': return Performance(ui, actions, data);
    case 'settings': return Settings(ui, actions, data);
    case 'channels': return ChannelList(ui, actions, data);
    case 'groups': return Groups(ui, actions, data);
    case 'system': return SystemConnection(ui, actions, data);
    case 'routing': return AudioRouting(ui, actions, data);
    case 'osc-midi': return OscMidi(ui, actions, data);
    case 'devices': return Devices(ui, actions, data);
    case 'logs': return Logs(ui, actions, data);
    case 'training': return Training(ui, actions, data);
    case 'rules': return RulesLibrary(ui, actions, data);
    default: return Dashboard(ui, actions, data);
  }
}
